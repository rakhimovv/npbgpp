import gc
import logging
import math
import os
from typing import Optional, Tuple, Union, List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import trimesh
from PIL import Image
from pytorch3d.renderer.mesh.rasterize_meshes import pix_to_non_square_ndc
from torch.utils.data import Dataset
from torchvision.datasets.folder import pil_loader

__all__ = ['BaseScene']

from tqdm import tqdm

from ..modeling import sample_points, project_points, calculate_view_selection_scores, compute_one_scale_visibility
from ..utils.pytorch3d import warp_images, center_crop, create_screen_intrinsic_matrix

log = logging.getLogger(__name__)


class BaseScene(Dataset):

    def __init__(
            self,
            scene_root: os.PathLike,
            images_root: os.PathLike,
            masks_root: Optional[os.PathLike] = None,
            pc_path: Optional[os.PathLike] = None,
            num_samples: Optional[int] = None,
            random_zoom: Optional[Tuple[float, float]] = None,
            random_shift: bool = False,
            image_size: Optional[Union[int, Tuple[int, int]]] = None,
            target_views_indices: Optional[List[int]] = None,
            exclude_indices: Optional[List[int]] = None,
            old_image_size: Tuple[int, int] = None,
            noise_sigma: Optional[float] = None,
            n_point: Optional[int] = None,
            image_margin: Optional[int] = None
    ):
        self.n_point = n_point
        self.noise_sigma = noise_sigma
        self.num_samples = num_samples
        self.random_zoom = random_zoom
        self.random_shift = random_shift
        self.scene_root = scene_root
        self.pc_path = pc_path
        self.image_margin = image_margin
        assert self.num_samples is None or self.num_samples > 0

        self.fx_ndc, self.fy_ndc, self.px_ndc, self.py_ndc, self.old_image_size = self.get_intrinsics_ndc()
        self.old_image_size = old_image_size or self.old_image_size
        self.names, self.R_cols, self.Ts, self.cam_poses = self.get_extrinsics()
        assert len(self.names) == len(self.R_cols) == len(self.Ts) == len(self.cam_poses)

        if target_views_indices is not None:
            for ti in target_views_indices:
                assert exclude_indices is None or ti not in exclude_indices
                assert 0 <= ti < len(self.names), f"{ti}, {len(self.names)}"
            self.target_views_indices_mask = torch.zeros(len(self.names), dtype=torch.bool)
            self.target_views_indices_mask[target_views_indices] = 1
            self.input_views_indices_mask = ~self.target_views_indices_mask.detach()
        else:
            self.input_views_indices_mask = torch.ones(len(self.names), dtype=torch.bool)
            self.target_views_indices_mask = torch.ones(len(self.names), dtype=torch.bool)

        if exclude_indices is not None:
            for ei in exclude_indices:
                assert 0 <= ei < len(self.names), f"{ei}, {len(self.names)}"
            self.target_views_indices_mask[exclude_indices] = 0
            self.input_views_indices_mask[exclude_indices] = 0

        self.input_views_indices = torch.arange(len(self.names))[self.input_views_indices_mask]
        self.target_views_indices = torch.arange(len(self.names))[self.target_views_indices_mask]
        self.exclude_indices = exclude_indices

        self.imgs_paths = self.get_paths(images_root, self.names)
        self.masks_paths = None if masks_root is None else self.get_paths(masks_root, self.names)

        image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.image_size = image_size if image_size is not None else self.old_image_size
        assert 0 < self.image_size[0] <= self.old_image_size[0] and 0 < self.image_size[1] <= self.old_image_size[1]
        if self.random_shift:
            assert self.image_size[0] < self.old_image_size[0] or self.image_size[1] < self.old_image_size[1]

        self.point_cloud = None
        self.cache_idx = None
        self.images = []
        self.masks = []

    def get_intrinsics_ndc(self) -> (float, float, float, float, (int, int)):
        """
           Args:
           Returns:
               Tuple: fx_ndc, fy_ndc, px_ndc, py_ndc, viewport_height, viewport_width
           """
        raise NotImplementedError()

    def get_extrinsics(self, verbose=True) -> (List[str], torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Args:
            verbose
        Returns:
            Tuple: [list of names, R_cols, Ts, cam_poses]
        """
        raise NotImplementedError()

    def get_paths(self, root: os.PathLike, names: List[str], ext: str = 'JPG'):
        paths = [os.path.join(root, f"{name}.{ext}") for name in names]
        for path in paths:
            assert os.path.exists(path), f"{path}"
        return paths

    def load_point_cloud(self, include_rgb: bool = False) -> dict:
        assert self.pc_path is not None
        if self.point_cloud is not None:
            if include_rgb:
                if 'features' in self.point_cloud:
                    return self.point_cloud
            else:
                return self.point_cloud

        if self.pc_path.endswith(".pcd"):
            pc = o3d.io.read_point_cloud(self.pc_path)
            vertices = np.asarray(pc.points)
            colors = np.asarray(pc.colors)
        else:
            pc = trimesh.load(self.pc_path)
            vertices = pc.vertices.view(np.ndarray)
            colors = pc.colors / 255

        self.point_cloud = {'points': torch.tensor(vertices, dtype=torch.float32)}
        if include_rgb:
            self.point_cloud['features'] = torch.tensor(colors, dtype=torch.float32)[:, :3]

        _, _, self.v = torch.pca_lowrank(self.point_cloud['points'], q=3)
        if self.n_point is not None:
            self.n_point = int(self.n_point)
            if self.n_point < self.point_cloud['points'].shape[0]:
                # todo use p3d farthest point sampling ?
                self.point_cloud['points'] = sample_points(self.point_cloud['points'], self.n_point, 123, 'random')[0]
        if self.noise_sigma is not None:
            self.point_cloud['points'] += self.noise_sigma * torch.randn_like(self.point_cloud['points'])
        log.info(f"{self.scene_root} #points={self.point_cloud['points'].shape}")
        return self.point_cloud

    def unload_point_cloud(self):
        del self.point_cloud
        self.point_cloud = None
        gc.collect()

    def load_images(self):
        log.info(f"Start image caching for {self.scene_root}...")
        # todo parallelize image reading
        if self.masks_paths is not None:
            self.masks = [self.read_mask(i) for i in range(len(self.masks_paths))]
        self.images = [self.read_image(i) for i in range(len(self.imgs_paths))]
        log.info(f"End image caching for {self.scene_root}.")

    def unload_images(self):
        del self.images, self.masks
        self.images = []
        self.masks = []
        gc.collect()

    def read_image(self, idx: int, ignore_exclude_indices: bool = False) -> Optional[torch.Tensor]:
        if not ignore_exclude_indices and self.exclude_indices is not None and idx in self.exclude_indices:
            return None
        if len(self.images) > 0:
            return self.images[idx]
        image: Image = pil_loader(self.imgs_paths[idx])
        image = image.resize(self.old_image_size[::-1], Image.BILINEAR)
        image = torch.tensor(np.array(image)).permute(2, 0, 1).contiguous()
        image = image / 255
        return image

    def read_mask(self, idx: int, ignore_exclude_indices: bool = False) -> Optional[torch.Tensor]:
        if (
                not ignore_exclude_indices and self.exclude_indices is not None and idx in self.exclude_indices) or self.masks_paths is None:
            return None
        if len(self.masks) > 0:
            return self.masks[idx]
        mask: Image = pil_loader(self.masks_paths[idx])
        mask = mask.resize(self.old_image_size[::-1], Image.BILINEAR)
        mask = torch.tensor(np.array(mask), dtype=torch.float32)
        mask = mask.mean(dim=2)[None, ...] / 255
        return mask

    def __len__(self):
        if self.num_samples is not None:
            return self.num_samples
        return len(self.target_views_indices)

    def __getitem__(self, item_idx: int):
        item_idx = item_idx % len(self.target_views_indices)
        idx = self.target_views_indices[item_idx].item()

        image = self.read_image(idx)
        mask = self.read_mask(idx)

        fx_ndc, fy_ndc, px_ndc, py_ndc = self.fx_ndc, self.fy_ndc, self.px_ndc, self.py_ndc

        # random shift
        center_pix = None
        # todo consider zoom ?
        if self.random_shift:
            if mask is None:
                _mask = torch.ones(1, self.old_image_size[0], self.old_image_size[1])
            else:
                _mask = mask
            _, foregr_i, foregr_j = torch.nonzero(_mask[:, ::8, ::8], as_tuple=True)
            foregr_i, foregr_j = foregr_i * 8, foregr_j * 8

            left_j = torch.tensor([self.image_size[1] // 2])
            right_j = torch.tensor([self.old_image_size[1] - 1 - self.image_size[1] // 2])
            top_i = torch.tensor([self.image_size[0] // 2])
            bottom_i = torch.tensor([self.old_image_size[0] - 1 - self.image_size[0] // 2])
            # left top
            foregr_j = torch.cat([foregr_j, left_j])
            foregr_i = torch.cat([foregr_i, top_i])
            # right top
            foregr_j = torch.cat([foregr_j, right_j])
            foregr_i = torch.cat([foregr_i, top_i])
            # left bottom
            foregr_j = torch.cat([foregr_j, left_j])
            foregr_i = torch.cat([foregr_i, bottom_i])
            # right bottom
            foregr_j = torch.cat([foregr_j, right_j])
            foregr_i = torch.cat([foregr_i, bottom_i])
            # filter the centers of crops that go beyond the border

            _m = (foregr_j <= right_j) & (foregr_j >= left_j) & (foregr_i <= bottom_i) & (foregr_i >= top_i)
            foregr_j = foregr_j[_m]
            foregr_i = foregr_i[_m]

            pnt_idx = np.random.choice(len(foregr_i))
            H, W = self.old_image_size
            x_mask_ndc = pix_to_non_square_ndc(W - 1 - foregr_j[pnt_idx], W, H)
            y_mask_ndc = pix_to_non_square_ndc(H - 1 - foregr_i[pnt_idx], H, W)
            center_pix = torch.tensor([foregr_i[pnt_idx], foregr_j[pnt_idx]])

            # f_ndc * center_world + p_ndc_old = center_ndc
            # f_ndc * center_world + p_ndc_new = 0
            px_ndc, py_ndc = px_ndc - x_mask_ndc, py_ndc - y_mask_ndc

        # crop according to image size
        if self.image_size[0] != self.old_image_size[0] or self.image_size[1] != self.old_image_size[1]:
            fx_ndc, fy_ndc, px_ndc, py_ndc = center_crop(fx_ndc, fy_ndc, px_ndc, py_ndc, self.old_image_size,
                                                         self.image_size)

        # random zoom
        if self.random_zoom:
            s = rand_(*self.random_zoom)
            fx_ndc, fy_ndc, px_ndc, py_ndc = fx_ndc * s, fy_ndc * s, px_ndc * s, py_ndc * s

        item = {
            "idx": idx,
            "cam_pos": self.cam_poses[idx],
            "R_row": self.R_cols[idx].t(),
            "T": self.Ts[idx],
            "fcl_ndc": torch.tensor((fx_ndc, fy_ndc), dtype=torch.float32),
            "prp_ndc": torch.tensor((px_ndc, py_ndc), dtype=torch.float32),
            "image_size": torch.tensor(self.image_size, dtype=torch.long)
        }

        empty = torch.zeros((1, *image.shape[1:]), dtype=torch.bool, device=image.device)
        if self.image_margin is not None and self.image_margin > 0:
            margin = self.image_margin
            empty[:, :margin, :] = 1
            empty[:, -margin:, :] = 1
            empty[:, :, :margin] = 1
            empty[:, :, -margin:] = 1

        # modify image and mask
        if self.image_size[0] != self.old_image_size[0] or self.image_size[1] != self.old_image_size[1] \
                or self.random_zoom or self.random_shift:
            src_fcl_ndc = torch.tensor((self.fx_ndc, self.fy_ndc), dtype=torch.float32)
            src_prp_ndc = torch.tensor((self.px_ndc, self.py_ndc), dtype=torch.float32)

            valid_mask = (~empty).float()
            hom_input = torch.cat([image, valid_mask], dim=0)
            if mask is not None:
                hom_input = torch.cat([hom_input, mask], dim=0)

            result = warp_images(hom_input[None, ...],
                                 src_fcl_ndc[None, :], src_prp_ndc[None, :], self.old_image_size,
                                 item["fcl_ndc"][None, :], item["prp_ndc"][None, :],
                                 self.image_size, cuda_sync=False)[0]

            image, valid_mask = result[:3], result[[3]]
            if mask is not None:
                mask = result[[4]]

            empty = ~(valid_mask >= 0.5)

        item["img"] = image
        if center_pix is not None:
            item["center_pix"] = center_pix
        if mask is not None:
            item["mask"] = mask
        if mask is not None:
            empty = torch.logical_and(empty, ~(mask >= 0.5))
        item['empty'] = empty
        if getattr(self, 'cache_idx', None) is not None:
            # we may set it during training
            item['cache_idx'] = self.cache_idx
        return item

    def calculate_near_far(self):
        loaded = True
        if self.point_cloud is None:
            loaded = False
            points = self.load_point_cloud()['points']  # (n, 3)

        fcl_ndc = torch.tensor((self.fx_ndc, self.fy_ndc), dtype=torch.float32)
        prp_ndc = torch.tensor((self.px_ndc, self.py_ndc), dtype=torch.float32)

        min_z, max_z = 1e10, 0

        log.info("Calculating near and far...")
        for i in tqdm(range(len(self.R_cols))):
            ndc_points, v_mask = project_points(points[None], self.R_cols[i].T[None], self.Ts[i][None],
                                                fcl_ndc[None], prp_ndc[None], self.old_image_size)  # (1, n, 3), (1, n)
            if v_mask.sum().item() != 0:
                zs = ndc_points[0][v_mask[0]][:, 2]
                min_z = min(min_z, zs.min().item())
                max_z = max(max_z, zs.max().item())
        log.info(f"{self.scene_root} near={min_z}, far={max_z}")

        if not loaded:
            self.unload_point_cloud()

        return min_z, max_z

    def get_scene_radius(self):
        max_radius = 0.0

        for cam_pos in self.cam_poses:
            max_radius = max(max_radius, cam_pos.norm())

        loaded = True
        if self.point_cloud is None:
            loaded = False
            self.load_point_cloud()

        points = self.point_cloud['points']  # (n, 3)
        max_radius = max(max_radius, points.norm(dim=1).max().item())

        if not loaded:
            self.unload_point_cloud()

        return max_radius

    @torch.no_grad()
    def prepare_patchmatch_format(self, save_path: str, scale_factor=1.0):
        # https://github.com/FangjinhuaWang/PatchmatchNet#reproducing-results

        h = int(math.ceil(self.old_image_size[0] * scale_factor / 8) * 8)
        w = int(math.ceil(self.old_image_size[1] * scale_factor / 8) * 8)
        print(f"IMAGE SIZE: {(h, w)}")

        os.makedirs(save_path, exist_ok=True)
        os.makedirs(os.path.join(save_path, 'cams_1'), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'masks'), exist_ok=True)

        # prepare cam.txt
        pairs_file = open(os.path.join(save_path, 'pair.txt'), 'w')
        print(f"Total images: {len(self.input_views_indices)}")
        pairs_file.write(f'{len(self.input_views_indices)}\n')

        # prepare common info
        fcl_ndc = torch.tensor([[self.fx_ndc, self.fy_ndc]], dtype=torch.float32)
        prp_ndc = torch.tensor([[self.px_ndc, self.py_ndc]], dtype=torch.float32)
        K = create_screen_intrinsic_matrix(fcl_ndc, prp_ndc, (h, w))[0]
        depth_min, depth_max = self.calculate_near_far()
        depth_min, depth_max = max(1, depth_min - 1), depth_max + 1
        unload_point_cloud = self.point_cloud is None
        points = self.load_point_cloud()['points']

        for i in tqdm(range(len(self.input_views_indices))):
            view_name = str(i).zfill(8)
            idx = self.input_views_indices[i].item()

            ndc_points, v_mask = project_points(points[None], self.R_cols[idx].T[None], self.Ts[idx][None],
                                                fcl_ndc, prp_ndc, (h, w))  # (1, n, 3), (1, n)
            interest_mask = compute_one_scale_visibility(
                ndc_points,
                v_mask,
                (h, w), scale=0.125, variant=1
            ).bool()[0]
            if interest_mask.sum().item() != 0:
                masked_points = points[interest_mask]
                scores = calculate_view_selection_scores(
                    masked_points,
                    self.R_cols[self.input_views_indices_mask].transpose(1, 2),
                    self.Ts[self.input_views_indices_mask],
                    fcl_ndc,
                    prp_ndc,
                    (h, w),
                    self.cam_poses[idx][None, :],
                    use_dist=False,
                    sigma_lower_baseline_angle=5,
                    sigma_higher_baseline_angle=15,
                )  # (1, k, n)
                scores = scores.sum(dim=2)  # (1, k)
                scores = scores[0]  # (k,)
            else:
                scores = 100 * torch.ones(len(self.input_views_indices))
            for k, t in enumerate(self.input_views_indices):
                if t.item() == idx:
                    scores[k] = 0.0
                    break
            del ndc_points, v_mask, interest_mask

            # write view selection info
            pairs_file.write(f'{i}\n')
            pairs_file.write('10')
            top_views = torch.topk(scores, 10)
            for j in range(10):
                pairs_file.write(f' {top_views.indices[j].item()} {top_views.values[j].item()}')
            pairs_file.write('\n')
            del scores

            # write camera params info
            R_col, T = self.R_cols[idx].numpy(), self.Ts[idx].numpy()
            R_col[:2] *= -1
            T[:2] *= -1
            with open(os.path.join(save_path, 'cams_1', f'{view_name}_cam.txt'), 'w') as f:
                f.write('extrinsic\n')
                f.write(f'{R_col[0][0]} {R_col[0][1]} {R_col[0][2]} {T[0]}\n')
                f.write(f'{R_col[1][0]} {R_col[1][1]} {R_col[1][2]} {T[1]}\n')
                f.write(f'{R_col[2][0]} {R_col[2][1]} {R_col[2][2]} {T[2]}\n')
                f.write('0.0 0.0 0.0 1.0\n\n')
                f.write('intrinsic\n')
                f.write(f'{K[0][0]} {K[0][1]} {K[0][2]}\n')
                f.write(f'{K[1][0]} {K[1][1]} {K[1][2]}\n')
                f.write(f'{K[2][0]} {K[2][1]} {K[2][2]}\n\n')
                f.write(f'{depth_min} {depth_max}\n')

            img = self.read_image(idx)
            mask = self.read_mask(idx)
            if scale_factor != 1.0:
                img = F.interpolate(img[None], size=(h, w), mode='bilinear', align_corners=False)[0]
                if mask is not None:
                    mask = F.interpolate(mask[None], size=(h, w), mode='bilinear', align_corners=False)[0]

            if mask is not None:
                # img = img.where(mask >= 0.5, torch.zeros(3, 1, 1))
                torchvision.utils.save_image(mask, os.path.join(save_path, 'masks', f'{view_name}.jpg'))
            torchvision.utils.save_image(img, os.path.join(save_path, 'images', f'{view_name}.jpg'))

        pairs_file.close()

        if unload_point_cloud:
            self.unload_point_cloud()


def rand_(min_: float, max_: float, *args):
    return min_ + (max_ - min_) * np.random.rand(*args)
