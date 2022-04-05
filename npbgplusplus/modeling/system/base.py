import contextlib
import logging
import os
from functools import partial
from typing import Optional, List, Tuple, Dict, Union

import lpips
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from hydra.utils import instantiate
from kornia import get_rotation_matrix2d
from omegaconf import DictConfig
from torch.utils.data import Dataset

from ..feature_extraction.rgb_converter import RGBConverter
from ..feature_extraction.view_processing import align_views_vertically, warp_affine
from ..metrics import ImagePairMetric, SSIMMetric, PSNRMetric
from ..metrics.vgg_loss import VGGLoss
from ..rasterizer.scatter import project_features
from ...data.build import build_datasets, build_loaders
from ...utils import comm
from ...utils.comm import is_main_process, synchronize

log = logging.getLogger(__name__)


class NPBGBase(pl.LightningModule):

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.hparams.update(cfg)
        self.descriptor_dim = self.hparams.system.descriptor_dim
        self.automatic_optimization = False

        # setup rgb converter
        self.rgb_converter = RGBConverter(self.descriptor_dim)

        # setup rasterizer
        # self.rasterizer = torch.jit.script(instantiate(self.hparams.system.rasterizer))
        self.rasterizer = instantiate(self.hparams.system.rasterizer)

        # setup refiner
        self.refiner = instantiate(self.hparams.system.refiner)
        if self.hparams.system.freeze_refiner:
            for p in self.refiner.parameters():
                p.requires_grad = False

        self.postprocessing_transform = None
        if self.hparams.system.postprocessing_transform is not None:
            self.postprocessing_transform = instantiate(self.hparams.system.postprocessing_transform)

        # In our case one dataset represents one scene. The convention for the datasets is the following:
        # Dict[stage, Optional[List[Tuple[dataset_name, dataset_config, dataset_class]]]]
        self.datasets: Dict[str, Optional[List[Tuple[str, DictConfig, Dataset]]]] = {
            'train': None,  # is built when calling self.train_dataloader
            'val': None,  # is built when calling self.val_dataloader
            'test': None  # is built when calling self.test_dataloader
        }

        # At each epoch we consider only {self.max_scenes_per_train_epoch} scenes to avoid OOM issues and to avoid
        # fetching xyz for each view, since we can share xyz for the views
        # from the same scene. Therefore we define cached points (xyz), which we update every epoch.

        self.max_scenes_per_train_epoch = self.hparams.system.max_scenes_per_train_epoch
        assert self.max_scenes_per_train_epoch > 0

        self.cached_points = None
        self.cached_valid_mask = None

        # We set {self.cache_idx2scene_idx} to know how to update self.cached_{points, valid_mask}
        # on {train/val/test}_epoch_start. Len(self.cache_idx2scene_idx[stage]) is less or equal self.max_scenes_per_train_epoch.
        self.cache_idx2scene_idx = {'train': [],
                                    'val': [],
                                    'test': []}  # set when calling self.{train/val/test}_dataloader

        # We set {self.cached_scene_idxs} to know which point clouds and images preload actually
        self.cached_train_scene_idxs = []

        # We choose first {self.max_scenes_per_train_epoch} from {self.unseen_train_scenes_idxs} during each train epoch.
        # On epoch end we mark those scenes as seen. When the list is empty we refill it.
        self.unseen_train_scenes_idxs = []  # set when calling self.train_dataloader

        self.register_buffer('bg_rgb', torch.tensor(list(self.hparams.system.bg_rgb_color)).view(1, 3, 1, 1))

        if self.hparams.system.style_weight > 0.0:
            self.vgg_loss = VGGLoss(style_img_path=self.hparams.system.style_img_path, optimized=True)
        else:
            self.vgg_loss = VGGLoss()
        self.vgg_loss.eval()
        self.loss_factors = {
            'm_reg_loss': 10.0,
            'style_loss': self.hparams.system.style_weight,
            'vgg_loss': 1.0,
            'mask_loss': 1000.0,
            'l1_d4_loss': 2500.0,
            'dimred_loss': 1.0,
        }
        if 'loss_factors' in self.hparams.system and self.hparams.system.loss_factors is not None:
            self.loss_factors.update(self.hparams.system.loss_factors)

        # Define metrics
        stage = 'test' if self.hparams.eval_only else 'val'

        if self.hparams.datasets[stage] is not None:
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    self.lpips_model = lpips.LPIPS(net='vgg')
            for p in self.lpips_model.parameters():
                p.requires_grad = False
            self.lpips_model.eval()

            self.lpips = self.create_per_stage_metric(partial(ImagePairMetric,
                                                              fun=partial(self.lpips_model, normalize=True)), stage)
            self.ssim = self.create_per_stage_metric(SSIMMetric, stage)
            self.psnr = self.create_per_stage_metric(PSNRMetric, stage)
            self.pairwise_metrics = {"LPIPS": self.lpips,
                                     'SSIM': self.ssim, "PSNR": self.psnr}

    def create_per_stage_metric(self, metric, stage):
        """
        Creates an instance of `metric` for each dataset in the specified stage
        Args:
            metric: constructor function for the desired metric
            stage: the stage for computing the metric (`val` | `test`)

        Returns:
            nn.ModuleDict
        """
        stage_dict: Dict[str, nn.ModuleList] = {stage: nn.ModuleList(  # Dict[stage, [metric_for_dataset_i,...]]
            [metric() for _ in range(len(self.hparams.datasets[stage]))])}
        return nn.ModuleDict(stage_dict)

    def forward(
            self,
            points: torch.Tensor,  # b, n, 3
            descriptors: Optional[torch.Tensor],  # b, c, n (if channel_first else: b, n, c)
            R_row: torch.Tensor,  # b, 3, 3 (R_row is applied to row vectors, i.e. x_view = x_world @ R_row + T)
            T: torch.Tensor,  # b, 3
            fcl_ndc: torch.Tensor,  # b, 2
            prp_ndc: torch.Tensor,  # b, 2
            image_size: Tuple[int, int],  # h, w
            channel_first: bool,  # if true use b, c, n convention for descriptors else b, n, c
            valid_mask: Optional[torch.Tensor] = None,  # b, n (if None, all points are valid),
            forward_info: Optional[dict] = None,  # additional info to output
            clamp: bool = False,  # clamp rgb values to [0, 1] range
            background: Optional[Union[float, torch.Tensor]] = None,  # b, 3, h, w or b, 3, 1, 1
            return_raster_result: bool = True,  # return raster result before alignment if any!
            debug: bool = False
    ) -> Tuple[torch.Tensor, dict]:
        if forward_info is None:
            forward_info = {}
        if background is not None:
            assert background.ndim == 4
            assert background.shape[1] == 3

        if descriptors is None:  # the case with no visible points
            forward_info['no_points'] = True
            value = points.new_zeros(1, 1, 1, 1)
            out = value.expand(points.shape[0], 3, *image_size)
            if self.postprocessing_transform is not None:
                out, _ = self.postprocessing_transform(out, None)
            if return_raster_result and self.training:
                forward_info['dimred_loss'] = out.new_zeros(1)
            return out, forward_info

        if return_raster_result:
            inp_descriptors = descriptors.detach()
            if channel_first:
                inp_descriptors = inp_descriptors.transpose(1, 2)
            # inp_descriptors: b, n, c
            rgbs = self.rgb_converter(inp_descriptors)  # b, n, 3
            with torch.no_grad():
                bg_feature = rgbs.new_ones(1, 3, 1, 1) if background is None else background
                forward_info['raster_result'] = project_features(
                    points, rgbs, R_row, T, fcl_ndc, prp_ndc, image_size, valid_mask,
                    scales=[1.0], ss_scale=[1], cat_mask=False, bg_feature=bg_feature,
                    channel_first=False)[0].cpu()
            if self.training:
                reconstructed = self.rgb_converter.reconstruct(rgbs)
                forward_info['dimred_loss'] = F.l1_loss(
                    reconstructed.where(valid_mask[:, :, None], reconstructed.new_zeros(1, 1, 1)).float(),  # b, n, c
                    inp_descriptors.where(valid_mask[:, :, None], inp_descriptors.new_zeros(1, 1, 1)),  # b, n, c
                    reduction='sum'
                )
            else:
                del rgbs
            del inp_descriptors
            torch.cuda.empty_cache()

        raster_img_size = image_size
        if self.hparams.system.align_during_rendering:
            _, R_row, T, fcl_ndc, prp_ndc, p2d, degrees, _ = align_views_vertically(
                None,
                R_row,
                T,
                fcl_ndc,
                prp_ndc,
                pad=True,
                img_size=image_size
            )
            if p2d is not None:
                raster_img_size = (image_size[0] + p2d[2] + p2d[3], image_size[1] + p2d[0] + p2d[1])

        raster_out = self.rasterizer(points, descriptors, R_row, T, fcl_ndc, prp_ndc,
                                     raster_img_size,
                                     valid_mask, channel_first=channel_first)

        if debug:
            forward_info['input_raster_raw'] = raster_out
            with torch.no_grad():
                zs = []
                for i, x in enumerate(raster_out):
                    z = self.rgb_converter(x[0, :self.descriptor_dim][None, ...].permute(0, 2, 3, 1)).cpu().permute(0,
                                                                                                                    3,
                                                                                                                    1,
                                                                                                                    2).float()
                    if self.postprocessing_transform is not None:
                        z, _ = self.postprocessing_transform(z, None)
                    if i > 0:
                        zh, zw = zs[0].shape[-2:]
                        zh, zw = zh - 4, zw - 4  # mind padding
                        lr = zw - z.shape[3]
                        z = F.pad(z, (lr // 2, lr - lr // 2, 0, 0), value=1.0)
                    z = F.pad(z, (2, 2, 2, 2), value=1.0)
                    zs.append(z)
                zs = torch.cat(zs, dim=2)[0]
                forward_info['input_raster'] = zs

        out = self.refiner(raster_out)

        if debug:
            sb = out[:1, :3].clamp(0.0, 1.0)
            if out.shape[1] == 4:
                sb = sb.where(torch.sigmoid(out[:1, 3:4]) >= 0.5, self.bg_rgb.type(sb.dtype))
            if self.postprocessing_transform is not None:
                sb, _ = self.postprocessing_transform(sb, None)
            forward_info['input_out'] = sb[0].detach().cpu()

        if self.hparams.system.align_during_rendering:
            current_height, current_width = out.shape[-2:]
            cx, cy = (current_width - 1) / 2, (current_height - 1) / 2
            M_inv = get_rotation_matrix2d(
                out.new_tensor((cx, cy)).expand(out.shape[0], 2),
                angle=degrees,
                scale=out.new_ones(fcl_ndc.shape)
            )
            out = warp_affine(out.float(), M_inv.float(), (current_height, current_width))
            l, r, t, b = p2d[0], current_width - p2d[1], p2d[2], current_height - p2d[3]
            out = out[:, :, t:b, l:r]

        mask = None
        if out.shape[1] == 4:
            mask = torch.sigmoid(out[:, [3], :, :])
            out = out[:, :3]

        if self.postprocessing_transform is not None:
            out, mask = self.postprocessing_transform(out, mask)
            if return_raster_result:
                forward_info['raster_result'], _ = self.postprocessing_transform(forward_info['raster_result'].float(),
                                                                                 None)
        if clamp:
            out = out.clamp(0.0, 1.0)

        if mask is not None:
            forward_info['mask'] = mask

            if background is not None and self.hparams.system.use_masks:
                assert not self.training
                forward_info['raw'] = out
                out = out.where(mask >= 0.5, background)

        if debug:
            if mask is not None:
                forward_info['input_fout'] = out.where(mask >= 0.5, self.bg_rgb)[0].detach().cpu()
            else:
                forward_info['input_fout'] = out[0].detach().cpu()

        return out, forward_info

    def extract_forward_args_from_batch(self, batch, **kwargs):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError

    def get_target_img_and_mask_and_empty(self, batch):
        target_img = batch['img']
        target_binary_mask: Optional[torch.Tensor] = None
        b, c, h, w = target_img.shape
        empty = batch['empty']

        if self.hparams.system.use_masks:
            target_binary_mask = batch['mask'] >= 0.5
            target_binary_mask = target_binary_mask.type(target_img.dtype)

        if self.postprocessing_transform is not None:
            target_img, target_binary_mask = self.postprocessing_transform(target_img, target_binary_mask)
            if empty is not None:
                empty, _ = self.postprocessing_transform(empty.long(), None)
                empty = empty.bool()

        return target_img, target_binary_mask, empty

    @staticmethod
    def down_scale(image: torch.Tensor, times: int):
        _, _, h, w = image.shape
        return F.interpolate(image, size=(int(h / times), int(w / times)), mode='bilinear', align_corners=False)

    def compute_loss(self, batch: Dict):
        # torch.autograd.set_detect_anomaly(True)

        target_img, target_binary_mask, empty = self.get_target_img_and_mask_and_empty(batch)

        debug = self.global_step % 500 == 0
        rendered_img, forward_info = self.forward(
            # the last args are used by npbgpp
            *self.extract_forward_args_from_batch(batch, all_points_are_interesting=False, aggregate=True, debug=debug),
            debug=debug
        )

        losses = {}
        if self.hparams.system.use_masks and self.refiner.out_channels == 4:
            empty_img = self.rasterizer.get_multiscale_empty_img(64, 64)
            self.refiner.eval()
            empty_mask = torch.sigmoid(self.refiner(empty_img)[:, 3, :, :])[:, None, :, :]
            self.refiner.train()
            losses["m_reg_loss"] = F.l1_loss(empty_mask, empty_mask.new_zeros(1, 1, 1, 1).expand_as(empty_mask))

        # save debug
        if 'input_oviews' in forward_info and 'input_views' in forward_info and 'input_raster' in forward_info and 'input_out' in forward_info and 'input_fout' in forward_info:
            with torch.no_grad():
                i1 = forward_info['input_oviews']
                i2 = forward_info['input_views']
                i3 = forward_info['input_raster']
                i4 = forward_info['input_out']
                i5 = forward_info['input_fout']
                i6 = target_img[0].detach().cpu()
                maxh = max([x.shape[1] for x in [i2, i3, i4, i5, i6]])
                i1 = F.interpolate(i1, size=(maxh, int(maxh * i1.shape[3] / i1.shape[2])), mode='bilinear',
                                   align_corners=False)
                i1 = torchvision.utils.make_grid(i1, nrow=i1.shape[0], padding=0)
                i1 = F.pad(i1, (1, 1, 1, 1))
                i2 = F.pad(i2, (1, 1, 1, maxh - i2.shape[1] + 1))
                i3 = F.pad(i3, (1, 1, 1, maxh - i3.shape[1] + 1))
                i4 = F.pad(i4, (1, 1, 1, maxh - i4.shape[1] + 1))
                i5 = F.pad(i5, (1, 1, 1, maxh - i5.shape[1] + 1))
                i6 = F.pad(i6, (1, 1, 1, maxh - i6.shape[1] + 1))
                final = torch.cat([i1, i2, i3, i4, i5, i6], dim=2)
                dir_path = os.path.join(os.getcwd(), f"train_images")
                # torchvision.utils.save_image(forward_info['input_out'], os.path.join(dir_path,
                #                                                                      f"step{self.global_step}_rank{comm.get_rank()}_io.png"))
                # torchvision.utils.save_image(forward_info['input_fout'], os.path.join(dir_path,
                #                                                                       f"step{self.global_step}_rank{comm.get_rank()}_ifo.png"))
                torchvision.utils.save_image(final, os.path.join(dir_path,
                                                                 f"step{self.global_step}_rank{comm.get_rank()}.png"))

        interesting_rgb_mask = ~empty
        if self.hparams.system.use_masks:
            target_img = target_img.where(target_binary_mask.bool(), self.bg_rgb)
            interesting_rgb_mask = torch.logical_and(interesting_rgb_mask, target_binary_mask.bool())
            if 'mask' in forward_info:
                forward_info['mask'] = forward_info['mask'].where(~empty, target_binary_mask)

        rendered_img = rendered_img.where(interesting_rgb_mask, target_img)

        losses["vgg_loss"] = 0
        style_loss = 0
        if 'no_points' not in forward_info:
            self.vgg_loss.eval()
            if self.loss_factors['style_loss'] > 0.0:
                losses["vgg_loss"], style_loss = self.vgg_loss(rendered_img, target_img, mask=target_binary_mask,
                                                               compute_style_loss=True)
            else:
                losses["vgg_loss"] = self.vgg_loss(rendered_img, target_img)

        if self.loss_factors['style_loss'] > 0.0:
            losses["style_loss"] = style_loss
        else:
            for key in filter(lambda x: x.startswith("l1_d"), self.loss_factors.keys()):
                if 'no_points' in forward_info:
                    losses[key] = 0
                else:
                    down = int(key[4])
                    losses[key] = F.l1_loss(self.down_scale(rendered_img, down), self.down_scale(target_img, down))

        if self.hparams.system.use_masks and self.refiner.out_channels == 4:
            if 'no_points' in forward_info:
                self.log("dice", 1.0, prog_bar=True, logger=True, sync_dist=True)
                losses["mask_loss"] = 0
            else:
                intersection = torch.sum(forward_info['mask'] * target_binary_mask, dim=(1, 2, 3))
                cardinality = torch.sum(forward_info['mask'] ** 2 + target_binary_mask ** 2, dim=(1, 2, 3))
                dice_score = 2 * intersection / (cardinality + 1e-10)
                self.log("dice", dice_score.mean().item(), prog_bar=True, logger=True, sync_dist=True)
                losses["mask_loss"] = torch.mean(-torch.log(dice_score + 1e-10))

        if 'dimred_loss' in forward_info:
            losses['dimred_loss'] = forward_info['dimred_loss']

        for key, loss in losses.items():
            self.log(key, loss.item(), prog_bar=True, logger=True, sync_dist=True)
        weighted_losses = {key: loss * self.loss_factors[key] for key, loss in losses.items()}
        loss = sum(weighted_losses.values())

        return loss

    def compute_dummy_loss(self):
        return 0 * sum(p.sum() for p in self.rgb_converter.parameters()) + \
               0 * sum(p.sum() for p in self.rasterizer.parameters()) + \
               0 * sum(p.sum() for p in self.refiner.parameters())

    def training_step(self, batch: Dict, batch_idx: int):
        loss = self.compute_loss(batch) + self.compute_dummy_loss()
        self.manual_backward(loss)

        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]

        for opt in optimizers:
            opt.step()
            opt.zero_grad()

    def validation_step(self, batch, batch_idx: int, *args):
        dataloader_idx = args[0] if len(args) == 1 else 0
        return self._shared_eval_step(batch, batch_idx, dataloader_idx, 'val')

    def test_step(self, batch, batch_idx: int, *args):
        dataloader_idx = args[0] if len(args) == 1 else 0
        return self._shared_eval_step(batch, batch_idx, dataloader_idx, 'test')

    @torch.no_grad()
    def _shared_eval_step(self, batch: Dict, batch_idx: int, dataloader_idx: Optional[int], stage: str):
        target_img, target_binary_mask, empty = self.get_target_img_and_mask_and_empty(batch)
        if self.hparams.system.use_masks:
            target_img = target_img.where(target_binary_mask.bool(), self.bg_rgb)

        rendered_img, forward_info = self.forward(
            *self.extract_forward_args_from_batch(batch, all_points_are_interesting=True, aggregate=False,
                                                  index=dataloader_idx), clamp=True,
            background=self.bg_rgb, return_raster_result=True)

        margin = None
        if 'image_margin' in self.hparams.datasets and self.hparams.datasets.image_margin is not None:
            margin = int(self.hparams.datasets.image_margin)
            target_img = target_img[:, :, margin:-margin, margin:-margin]
            rendered_img = rendered_img[:, :, margin:-margin, margin:-margin]

        cache_idx = dataloader_idx
        scene_idx = self.cache_idx2scene_idx[stage][cache_idx]
        assert batch['cache_idx'][0].item() == cache_idx

        # Save rendered images
        if self.hparams.system.save_rendered_eval_images:
            def img_path(suffix) -> str:
                dataset_name, _, dataset = self.datasets[stage][scene_idx]
                dir_path = os.path.join(os.getcwd(), f"rendered/{dataset_name}/{stage}_epoch{self.current_epoch}")
                return os.path.join(dir_path, f"{dataset.names[batch['idx'][i].item()]}_{suffix}.png")

            for i in range(rendered_img.size(0)):
                raster_img = None
                if 'raster_result' in forward_info:
                    raster_img = forward_info['raster_result'][i].cpu()
                    if margin is not None:
                        raster_img = raster_img[:, margin:-margin, margin:-margin]
                raw_img = None
                if 'raw' in forward_info:
                    raw_img = forward_info['raw_result'][i].cpu()
                    if margin is not None:
                        raw_img = raw_img[:, margin:-margin, margin:-margin]

                if self.hparams.eval_only:
                    torchvision.utils.save_image(rendered_img[i].cpu(), img_path('rendered'))
                    torchvision.utils.save_image(target_img[i].cpu(), img_path('gt'))
                    if raster_img is not None:
                        torchvision.utils.save_image(raster_img, img_path('raster'))
                else:
                    debug_img = torch.cat([rendered_img[i].cpu(), target_img[i].cpu()], dim=2)
                    if raw_img is not None:
                        debug_img = torch.cat([raw_img, debug_img], dim=2)
                    if raster_img is not None:
                        debug_img = torch.cat([raster_img, debug_img], dim=2)
                    torchvision.utils.save_image(debug_img, img_path('debug'))

        # Calculate metrics for current batch
        for name, metric in self.pairwise_metrics.items():
            metric[stage][scene_idx].update(rendered_img, target_img)

    def get_scene(self, stage, scene_idx):
        return self.datasets[stage][scene_idx][2]

    def update_cache(self, stage: str):
        points_list = [self.get_scene(stage, scene_idx).point_cloud['points'] for scene_idx in
                       self.cache_idx2scene_idx[stage]]
        max_points_num = max([len(points) for points in points_list])
        self.cached_points = torch.zeros(len(points_list), max_points_num, 3)
        self.cached_valid_mask = torch.zeros(len(points_list), max_points_num, dtype=torch.bool)
        for cache_idx in range(len(points_list)):
            points = points_list[cache_idx]
            self.cached_points[cache_idx, :points.shape[0]] = points
            self.cached_valid_mask[cache_idx, :points.shape[0]] = True

    def _shared_epoch_start(self, stage):
        for scene_idx in self.cache_idx2scene_idx[stage]:
            self.get_scene(stage, scene_idx).load_point_cloud()
            if self.hparams.datasets.cache_images:
                if stage == 'train':
                    if scene_idx in self.cached_train_scene_idxs:
                        self.get_scene(stage, scene_idx).load_images()
                else:
                    self.get_scene(stage, scene_idx).load_images()
        self.update_cache(stage)

    def on_train_epoch_start(self):
        self._shared_epoch_start('train')
        self.create_folder(os.path.join(os.getcwd(), f"train_images"))

    def on_validation_epoch_start(self):
        self.our_train_epoch_end()
        self._shared_epoch_start('val')
        if self.datasets['val'] != None and self.hparams.system.save_rendered_eval_images:
            for dataset_name, _, _ in self.datasets['val']:
                self.create_folder(os.path.join(os.getcwd(), f"rendered/{dataset_name}/val_epoch{self.current_epoch}"))

    def on_test_epoch_start(self):
        self._shared_epoch_start('test')
        if self.datasets['test'] != None and self.hparams.system.save_rendered_eval_images:
            for dataset_name, _, _ in self.datasets['test']:
                self.create_folder(os.path.join(os.getcwd(), f"rendered/{dataset_name}/test_epoch{self.current_epoch}"))

    def _shared_epoch_end(self, stage):
        self.cached_points = None
        self.cached_valid_mask = None
        for scene_idx in self.cache_idx2scene_idx[stage]:
            self.get_scene(stage, scene_idx).unload_point_cloud()
            self.get_scene(stage, scene_idx).unload_images()

    @torch.no_grad()
    def _shared_eval_epoch_end(self, stage: str):
        # Report metrics for each scene in cache
        for scene_idx in self.cache_idx2scene_idx[stage]:
            dataset_name = self.datasets[stage][scene_idx][0]
            for name, metric in self.pairwise_metrics.items():
                self.log(f"{name}/{dataset_name}", metric[stage][scene_idx].compute(), prog_bar=True, logger=True)
                metric[stage][scene_idx].reset()

    def on_train_epoch_end(self, *args, **kwargs):
        if self.datasets['val'] is None or self.hparams.trainer.limit_val_batches == 0:
            self.our_train_epoch_end()

    def our_train_epoch_end(self, *args, **kwargs):
        # called in the beginning on_validation_epoch_start or
        # inside on_train_epoch_end if there is no validation
        # we assume that on_train_epoch_end should be always called before on_validation_epoch_start
        # in contrast to what happens in pytorch-lightning
        self._shared_epoch_end('train')

    def validation_epoch_end(self, outputs):
        self._shared_eval_epoch_end('val')
        self._shared_epoch_end('val')

    def test_epoch_end(self, outputs):
        self._shared_eval_epoch_end('test')
        self._shared_epoch_end('test')

    def train_dataloader(self):
        if self.datasets['train'] is None:
            self.datasets['train'] = build_datasets(self.hparams, 'train')
            # for name, _, d in self.datasets['train']:
            #     d.load_point_cloud()
            #     d.unload_point_cloud()
            log.info(f"Total number of train scenes: {len(self.datasets['train'])}")

        # Choose current subset of scenes
        if len(self.unseen_train_scenes_idxs) == 0:
            num_scenes = len(self.datasets['train'])
            if num_scenes > self.max_scenes_per_train_epoch:
                cached_scene_idxs = torch.randperm(num_scenes,
                                                   generator=torch.Generator().manual_seed(123 + self.current_epoch))
                cached_scene_idxs = cached_scene_idxs.tolist()

                # to avoid situation (when during one epoch we have the less number of scenes than usually
                # otherwise could lead to overfitting
                remainder = num_scenes % self.max_scenes_per_train_epoch
                if remainder > 0:
                    _inds = cached_scene_idxs[:remainder]
                    cached_scene_idxs = cached_scene_idxs + _inds
            else:
                cached_scene_idxs = torch.arange(num_scenes)
                cached_scene_idxs = cached_scene_idxs.tolist()
            self.unseen_train_scenes_idxs = cached_scene_idxs
        current_train_scenes_idxs = self.unseen_train_scenes_idxs[:self.max_scenes_per_train_epoch]
        self.unseen_train_scenes_idxs = self.unseen_train_scenes_idxs[self.max_scenes_per_train_epoch:]

        self.cache_idx2scene_idx['train'] = current_train_scenes_idxs

        # Current worker (gpu) works with its own sub-subset of scenes
        rank = comm.get_rank()

        if self.hparams.dataloader.train_data_mode == "common":
            self.cached_train_scene_idxs = self.cache_idx2scene_idx['train']
        else:
            # Change sampler so that there is no need to cache everything

            # the logic below is tightly coupled with SceneDistributedSampler
            # see the code of SceneDistributedSampler first
            world_size = comm.get_world_size()
            if len(current_train_scenes_idxs) >= world_size:
                assert len(current_train_scenes_idxs) % world_size == 0
                num_scenes_per_gpu = len(current_train_scenes_idxs) // world_size
                self.cached_train_scene_idxs = current_train_scenes_idxs[
                                               num_scenes_per_gpu * rank:num_scenes_per_gpu * (rank + 1)]
            else:
                assert world_size % len(current_train_scenes_idxs) == 0
                num_gpus_per_scene = world_size // len(current_train_scenes_idxs)
                self.cached_train_scene_idxs = current_train_scenes_idxs[rank // num_gpus_per_scene]

            if not isinstance(self.cached_train_scene_idxs, list):
                self.cached_train_scene_idxs = [self.cached_train_scene_idxs]
            if not isinstance(self.cache_idx2scene_idx['train'], list):
                self.cache_idx2scene_idx['train'] = [self.cache_idx2scene_idx['train']]

        log.info(f"rank: {rank}, self.cache_idx2scene_idx['train']: {self.cache_idx2scene_idx['train']}, "
                 f"image_caching_for: {self.cached_train_scene_idxs}")

        # Set cache idx
        for cache_idx, scene_idx in enumerate(self.cache_idx2scene_idx['train']):
            # we set attribute cache_idx for the dataset in order for items in batch have cache_idx,
            # so we can perform torch.index_select from cached points
            self.datasets['train'][scene_idx][2].cache_idx = cache_idx

        datasets = [self.datasets['train'][scene_idx][2] for scene_idx in current_train_scenes_idxs]
        if self.hparams.dataloader.train_data_mode == "common":
            loaders = build_loaders(self.hparams, [torch.utils.data.ConcatDataset(datasets)], 'train')
        else:
            loaders = build_loaders(self.hparams, [torch.utils.data.ConcatDataset(datasets)], 'train',
                                    sampler_name="scene", seed=self.current_epoch)
        return loaders[0]

    def eval_dataloader(self, stage):
        if self.datasets[stage] is None:
            self.datasets[stage] = build_datasets(self.hparams, stage)
        self.cache_idx2scene_idx[stage] = torch.arange(len(self.datasets[stage])).tolist()
        for cache_idx, scene_idx in enumerate(self.cache_idx2scene_idx[stage]):
            self.get_scene(stage, scene_idx).cache_idx = cache_idx
        loaders = build_loaders(self.hparams, [d for _, _, d in self.datasets[stage]], stage)
        return loaders

    def val_dataloader(self):
        return self.eval_dataloader('val')

    def test_dataloader(self):
        return self.eval_dataloader('test')

    def state_dict(self, *args, **kwargs):
        out = super().state_dict(*args, **kwargs)
        keys_to_delete = []
        for key in out.keys():
            if 'cached' in key or '_inception' in key or 'vgg_loss' in key or 'bg_rgb' in key or 'lpips' in key:
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del out[key]
        return out

    def create_folder(self, dir_path):
        if not os.path.exists(dir_path):
            if is_main_process():
                os.makedirs(dir_path, exist_ok=True)
        synchronize()
