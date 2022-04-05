import os
from typing import Optional, Tuple, Union, Dict

import torch
import torch.nn.functional as F
import torchvision
from hydra.utils import instantiate
from kornia.augmentation import ColorJitter
from omegaconf import DictConfig
from tqdm import tqdm

from .base import NPBGBase
from .. import SphereAggregator
from ..feature_extraction.cropping import extract_regions
from ..feature_extraction.view_processing import align_views_vertically
from ..rasterizer.project import project_points
from ..rasterizer.scatter import compute_one_scale_visibility, project_features
from ...utils import comm
from ...utils.pytorch3d import get_ndc_positive_bounds


class NPBGPlusPlus(NPBGBase):

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.feature_extractor = instantiate(self.hparams.system.feature_extractor)
        self.aggregator = instantiate(self.hparams.system.aggregator)

        self.aug = None
        if self.hparams.system.color_aug_p > 0.0:
            self.aug = ColorJitter(saturation=0.5, hue=0.5, brightness=0.1, p=self.hparams.system.color_aug_p,
                                   same_on_batch=False)

    def update_descriptors(
            self,
            points: torch.Tensor,  # b, n, 3
            views_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            # views_data contains:
            #   images: torch.Tensor, # b, k, 3, H, W
            #   R_Rows: torch.Tensor,  # b, k, 3, 3
            #   Ts: torch.Tensor,  # b, k, 3
            #   focal_length: torch.Tensor,  # b, 2
            #   principal_point: torch.Tensor,  # b, 2
            interest_mask: Optional[torch.Tensor] = None,  # b, n,
            crop_max_size: Optional[Union[int, Tuple[int, int]]] = None,
            debug: bool = False,
            impose_visibility: bool = False,  # set to True if you want measure fps
            **agg_kwargs
    ) -> dict:
        forward_info = {}
        if points.shape[1] == 0:
            return forward_info

        with torch.no_grad():
            b, k, _3 = views_data[2].shape
            n = points.shape[1]

            # crop the views images as much as possible but ensuring that interesting points are visible
            views_images, views_R_row, views_T, views_fcl_ndc, views_prp_ndc = views_data
            views_points = points[:, None, :, :].expand(b, k, n, 3).contiguous().view(b * k, n, 3)
            views_images = views_images.view(b * k, *views_images.shape[2:])
            views_R_row = views_R_row.view(b * k, 3, 3)
            views_T = views_T.view(b * k, 3)
            # views_fcl_ndc = views_fcl_ndc.view(b * k, 2)
            # views_prp_ndc = views_prp_ndc.view(b * k, 2)
            views_fcl_ndc = views_fcl_ndc[:, None, :].expand(b, k, 2).contiguous().view(b * k, 2)
            views_prp_ndc = views_prp_ndc[:, None, :].expand(b, k, 2).contiguous().view(b * k,
                                                                                        2)

            if interest_mask is None:
                interest_mask = torch.ones(1, 1, 1, device=views_T.device, dtype=torch.bool).expand(b, k, n)
            else:
                interest_mask = interest_mask[:, None, :].expand(b, k, n).contiguous()

            vnp, visible_mask = project_points(views_points, views_R_row, views_T, views_fcl_ndc,
                                               views_prp_ndc, views_images.shape[-2:])  # (b*k, n, 3), (b*k, n)
            if self.hparams.system.visibility_scale is not None:
                visible_mask = compute_one_scale_visibility(
                    vnp,
                    visible_mask,
                    views_images.shape[-2:],
                    scale=self.hparams.system.visibility_scale,
                    variant=1,
                ).bool()
            visible_mask = visible_mask.view(b, k, n)  # b, k, n
            visible_mask = torch.logical_and(visible_mask, interest_mask)  # b, k, n
            del interest_mask

            if debug:
                o_images = views_images.detach().clone().cpu()
            if self.hparams.system.align_view_images:
                views_images, views_R_row, views_T, views_fcl_ndc, views_prp_ndc, _, _, _ = align_views_vertically(
                    views_images, views_R_row, views_T, views_fcl_ndc, views_prp_ndc)

            if crop_max_size is not None:
                views_images, views_fcl_ndc, views_prp_ndc, _ = extract_regions(
                    views_points,
                    views_images.view(b * k, *views_images.shape[-3:]),
                    views_R_row,
                    views_T,
                    views_fcl_ndc,
                    views_prp_ndc,
                    visible_mask.view(b * k, n),
                    max_size=crop_max_size,
                    # avoid_scaling_down=False
                )  # (b*k, c, h, w), (b*k, 2), (b*k, 2)

            # calculate point coordinates in local view coordinate systems
            views_ndc_points, post_visible_mask = project_points(
                views_points, views_R_row, views_T,
                views_fcl_ndc,
                views_prp_ndc,
                views_images.shape[-2:])  # (b*k, n, 3), (b*k, n)
            visible_mask = torch.logical_and(visible_mask, post_visible_mask.view(b, k, n))

        # get image features
        if debug:
            c_images = views_images.detach().clone().cpu()
            torch.cuda.empty_cache()

        views_images = self.feature_extractor(views_images)  # b*k, fc, fh, fw

        # sample descriptors
        scale_x, scale_y = get_ndc_positive_bounds(views_images.shape[-2:])
        grid = views_ndc_points[:, :, :2].unsqueeze(1)  # b*k, 1, n, 2
        grid[..., 0] /= scale_x
        grid[..., 1] /= scale_y
        descriptors = F.grid_sample(views_images, -grid, mode='bilinear', align_corners=False)  # b*k, fc, 1, n
        descriptors = descriptors.view(b, k, -1, n)  # b, k, fc, n

        if impose_visibility:
            visible_mask = torch.where(visible_mask, visible_mask, ~visible_mask)

        self.aggregator.update(descriptors, visible_mask, visible_mask,
                               views_points.view(b, k, n, 3), views_R_row.view(b, k, 3, 3),
                               views_T.view(b, k, 3), **agg_kwargs)

        if debug:
            with torch.no_grad():
                v_images = views_images.view(b, k, *views_images.shape[-3:])[0].detach()  # k, fc, fh, fw
                fh, fw = v_images.shape[-2:]
                v_images = self.rgb_converter(v_images.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).cpu().float()
                if self.postprocessing_transform is not None:
                    v_images, _ = self.postprocessing_transform(v_images, None)

                c_images = c_images.view(b, k, *c_images.shape[-3:])[0].float()  # k, 3, fh, fw

                b_images = project_features(
                    views_points.view(b, k, n, 3)[0],
                    self.rgb_converter(descriptors[0].detach().permute(0, 2, 1)).permute(0, 2, 1).float(),
                    views_R_row.view(b, k, 3, 3)[0],
                    views_T.view(b, k, 3)[0],
                    views_fcl_ndc.view(b, k, 2)[0],
                    views_prp_ndc.view(b, k, 2)[0],
                    (fh, fw),
                    visible_mask.view(b, k, n)[0],
                    channel_first=True,
                    bg_feature=c_images.to(views_T.device),
                    cat_mask=False
                )[0].detach().cpu()
                if self.postprocessing_transform is not None:
                    b_images, _ = self.postprocessing_transform(b_images, None)
                v_images = torchvision.utils.make_grid(v_images, nrow=k, padding=1)
                b_images = torchvision.utils.make_grid(b_images, nrow=k, padding=1)
                r_image = torch.cat([v_images, b_images], dim=1)
                forward_info['input_views'] = r_image

                o_images = o_images.view(b, k, *o_images.shape[-3:])[0]
                if self.postprocessing_transform is not None:
                    o_images, _ = self.postprocessing_transform(o_images, None)
                forward_info['input_oviews'] = o_images

        return forward_info

    def extract_forward_args_from_batch(self, batch, all_points_are_interesting: bool = False, aggregate=False,
                                        debug=False, index=None):
        device = batch['cache_idx'].device
        cache_idx = batch['cache_idx'].cpu()  # (b,)
        points = torch.index_select(input=self.cached_points, dim=0, index=cache_idx).to(device)  # (b, n, 3)
        valid_mask = torch.index_select(input=self.cached_valid_mask, dim=0, index=cache_idx).to(device)  # (b, n)
        forward_info = {}

        if aggregate:
            views_data = batch['views_data']
            interest_mask = valid_mask
            if not all_points_are_interesting:
                # consider only those points that are visible in target views
                vnp, interest_mask = project_points(points, batch['R_row'], batch['T'],
                                                    batch['fcl_ndc'], batch['prp_ndc'], batch['img'].shape[-2:])
                interest_mask = compute_one_scale_visibility(vnp, interest_mask, batch['img'].shape[-2:], scale=1.0,
                                                             variant=1).bool()
                interest_mask = torch.logical_and(interest_mask, valid_mask)

            if all_points_are_interesting:
                crop_max_size = None
            else:
                crop_max_size = [x * 2 for x in batch['img'].shape[-2:]]

            # if comm.get_rank() == 0:
            #     interest_mask = 0 * interest_mask
            maxn = interest_mask.sum(dim=1).max().item()  # b
            if maxn == 0:
                debug = True
            maxn = max(1, maxn)
            new_points = points.new_zeros(points.shape[0], maxn, 3)
            new_mask = interest_mask.new_zeros(points.shape[0], maxn)
            if maxn > 0:
                for i in range(points.shape[0]):
                    ni = interest_mask[i].sum().item()
                    if ni > 0:
                        new_points[i, :ni] = points[i, interest_mask[i], :]
                        new_mask[i, :ni] = 1
            points = new_points
            interest_mask = new_mask
            self.aggregator.reset()
            torch.cuda.empty_cache()
            forward_info = self.update_descriptors(points, views_data, interest_mask, crop_max_size, debug=debug)
            torch.cuda.empty_cache()

        # we need to provide index during validation, because batch consists of views from one scene only
        # i.e. batch size == 1
        # in the mean time the aggregator's states have batch dimension equal to number of scenes
        # index = None
        # if len(torch.unique(cache_idx)) == 1:
        #     index = cache_idx[0].item()

        descriptors, visible_mask = self.aggregator(points, batch['R_row'], batch['T'], batch['fcl_ndc'],
                                                    batch['prp_ndc'], index=index)
        mask = visible_mask
        # if self.training:
        #     # inject random noise
        #     descriptors = descriptors.where(visible_mask[:, None, :],
        #                                     torch.randn_like(descriptors, device=descriptors.device))
        #     mask = interest_mask

        torch.cuda.empty_cache()

        # print("NUM POINTS VISIBLE", mask.sum())

        return points, descriptors, batch['R_row'], batch['T'], batch['fcl_ndc'], \
               batch['prp_ndc'], batch['image_size'][0].cpu().tolist(), True, mask, forward_info

    def compute_loss(self, batch: Dict):
        loss = super(NPBGPlusPlus, self).compute_loss(batch)

        if self.hparams.system.style_weight == 0.0:
            target_img, target_binary_mask, empty = self.get_target_img_and_mask_and_empty(batch)

            self.feature_extractor.eval()
            self.refiner.eval()
            reg_target = self.refiner(
                self.rasterizer.get_multi_scale_img(self.feature_extractor(target_img),
                                                    target_binary_mask.bool() if target_binary_mask is not None else None)
            )[:, :3, :, :]
            self.feature_extractor.train()
            self.refiner.train()
            if self.hparams.system.use_masks:
                color_reg_loss = F.l1_loss(
                    reg_target.float().where(target_binary_mask.bool(), target_img),
                    target_img.where(target_binary_mask.bool(), self.bg_rgb)
                )
            else:
                color_reg_loss = F.l1_loss(reg_target.float(), target_img)
            self.log("rgb_reg_loss", color_reg_loss.item(), prog_bar=True, logger=True, sync_dist=True)

            loss = loss + 1000 * color_reg_loss

        return loss

    def compute_dummy_loss(self):
        loss = super(NPBGPlusPlus, self).compute_dummy_loss()
        return loss + 0 * sum(p.sum() for p in self.aggregator.parameters()) + \
               0 * sum(p.sum() for p in self.feature_extractor.parameters())

    def training_step(self, batch: Dict, batch_idx: int):
        if self.aug is not None:
            with torch.no_grad():
                rank = comm.get_rank()
                world_size = comm.get_world_size()
                batch_size_per_gpu = self.hparams.dataloader.train.total_batch_size // world_size
                global_indices_of_batch_elements = range(rank * batch_size_per_gpu, (rank + 1) * batch_size_per_gpu)
                k = 0
                for i in range(self.hparams.dataloader.train.total_batch_size):
                    _ = self.aug(torch.ones(1, 3, 1, 1, dtype=batch['img'].dtype, device=batch['img'].device))
                    if self.aug._params['batch_prob'].item() and i in global_indices_of_batch_elements:
                        batch['img'][[k]] = self.aug.apply_transform(batch['img'][[k]], self.aug._params)
                        batch['views_data'][0][k] = self.aug.apply_transform(batch['views_data'][0][k],
                                                                             self.aug._params)
                        k += 1
        super(NPBGPlusPlus, self).training_step(batch, batch_idx)

    def on_train_batch_end(self, *args, **kwargs):
        self.aggregator.reset()

    @torch.no_grad()
    def aggregate_from_all_views(self, stage):
        num_scenes, num_points = self.cached_points.shape[:2]
        self.aggregator.reset(num_scenes, num_points, self.descriptor_dim, self.device)
        for cache_idx, scene_idx in enumerate(self.cache_idx2scene_idx[stage]):
            torch.cuda.empty_cache()
            points = self.cached_points[cache_idx][None, :, :].to(self.device)  # (1, N, 3)
            rr, t, fl, pp, image_idxs = self.datasets[stage][scene_idx][2].get_input_view_cameras(
                expand_intrinsics=True)  # (k, 3, 3), ...
            num_views = rr.shape[0] if not self.hparams.trainer.fast_dev_run else 1
            for i in tqdm(range(num_views)):
                image = self.datasets[stage][scene_idx][2].read_image(image_idxs[i].item()).detach()[None, None, :, :,
                        :]  # (1, 1, 3, H, W)
                self.update_descriptors(
                    points,
                    (
                        image.to(points.device),
                        rr[i][None, None, :, :].to(self.device),
                        t[i][None, None, :].to(self.device),
                        fl[i][None, :].to(self.device),
                        pp[i][None, :].to(self.device)
                    ),
                    index=cache_idx
                )
                torch.cuda.empty_cache()

    def save_descriptive_coefficients(self, stage):
        self.create_folder(os.path.join(os.getcwd(), "descriptors"))
        if isinstance(self.aggregator, SphereAggregator):
            self.aggregator.calculate_beta_and_visible_mask()
            ym = self.aggregator.ym.transpose(-2, -1).unsqueeze(2)  # b, n, 1, c
            beta = self.aggregator.beta  # b, n, m, c
            coefficients = torch.cat([ym, beta], dim=2)  # b, n, 1+m, c
        else:
            features, mask = self.aggregator.forward(torch.Tensor([[1]]), *([None] * 4))
            coefficients = features  # b, n, c

        for cache_idx, scene_idx in enumerate(self.cache_idx2scene_idx[stage]):
            n_points = self.cached_valid_mask[cache_idx].long().sum().item()
            dataset_name, _, dataset = self.datasets[stage][scene_idx]
            scene_name = '_'.join(dataset_name.split('_')[:-1])
            save_path = os.path.join(os.getcwd(), f"descriptors/{scene_name}.pth")
            coefficients = coefficients[cache_idx, :n_points].view(n_points, -1).data.detach().cpu()
            torch.save(coefficients, save_path)
        del coefficients
        torch.cuda.empty_cache()

    def on_validation_epoch_start(self):
        super(NPBGPlusPlus, self).on_validation_epoch_start()
        self.aggregate_from_all_views('val')

    def on_test_epoch_start(self):
        super(NPBGPlusPlus, self).on_test_epoch_start()
        self.aggregate_from_all_views('test')
        self.save_descriptive_coefficients('test')

    def on_validation_epoch_end(self):
        self.aggregator.reset()
        super(NPBGPlusPlus, self).on_validation_epoch_end()

    def on_test_epoch_end(self):
        self.aggregator.reset()
        super(NPBGPlusPlus, self).on_test_epoch_end()

    def configure_optimizers(self):
        def param_num(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        param_sets = [
            {"params": self.rgb_converter.parameters(), "lr": 1e-4}
        ]
        if not self.hparams.system.freeze_feature_extractor:
            param_sets.append({"params": self.feature_extractor.parameters(), "lr": 1e-4})
        if not self.hparams.system.freeze_aggregator and param_num(self.aggregator) > 0:
            param_sets.append({"params": self.aggregator.parameters(), "lr": 1e-4})
        if param_num(self.rasterizer) > 0:
            param_sets.append({"params": self.rasterizer.parameters(), "lr": 1e-4})  # bg feature
        if not self.hparams.system.freeze_refiner:
            param_sets.append({"params": self.refiner.parameters(), "lr": 1e-4})

        return torch.optim.Adam(param_sets)
