import os
from typing import Optional, Tuple, Union, List

import numpy as np
import torch
import trimesh

__all__ = ['ScannetScene']

from .base import BaseScene
from ..utils.pytorch3d import convert_screen_intrinsics_to_ndc


class ScannetScene(BaseScene):

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
            **kwargs):
        super().__init__(scene_root, images_root, masks_root, pc_path, num_samples, random_zoom, random_shift,
                         image_size, target_views_indices, exclude_indices, None, **kwargs)
        assert self.image_margin is not None and self.image_margin > 0

    def get_paths(self, root: os.PathLike, names: List[str], ext: str = 'jpg'):  # Changes default value for ext
        return super().get_paths(root, names, ext)

    # https://github.com/facebookresearch/pytorch3d/blob/main/docs/notes/cameras.md
    def get_intrinsics_ndc(self):
        camera = np.load(os.path.join(self.scene_root, 'intrinsics.npy'), allow_pickle=True).item()
        image_height, image_width = camera['height'], camera['width']
        K = camera['K']
        return convert_screen_intrinsics_to_ndc((image_height, image_width), K[0, 0], K[1, 1])  # , K[0, 2], K[1, 2])

    def get_extrinsics(self, verbose=True) -> (List[str], torch.Tensor, torch.Tensor, torch.Tensor):
        extrinsics = np.load(os.path.join(self.scene_root, 'extrinsics.npy'))
        labels = sorted([name.split('.')[0] for name in os.listdir(os.path.join(self.scene_root, 'images'))], key=int)
        R_cols, Ts, cam_poses = [], [], []
        for RT in extrinsics:
            R_col = RT[:3, :3]
            T = RT[:3, 3]
            R_col[:2, :] *= -1
            T[:2] *= -1
            cam_pos = -R_col.T @ T

            R_cols.append(R_col)
            Ts.append(T)
            cam_poses.append(cam_pos)
        return labels, torch.tensor(R_cols, dtype=torch.float32), torch.tensor(Ts, dtype=torch.float32), \
               torch.tensor(cam_poses, dtype=torch.float32)

    def _load_point_cloud(self, include_rgb: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        pc = trimesh.load(os.path.join(self.scene_root, "full.ply"))
        vertices = torch.tensor(pc.vertices.view(np.ndarray), dtype=torch.float32)
        if include_rgb:
            rgb = torch.tensor(pc.colors, dtype=torch.float32)[:, :3] / 255
            return vertices, rgb
        return vertices
