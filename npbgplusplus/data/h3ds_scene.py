import os
from pathlib import Path
from typing import Optional, Tuple, Union, List

import numpy as np
import torch

from ..utils.pytorch3d import convert_screen_intrinsics_to_ndc

try:
    from h3ds.dataset import H3DS
except ModuleNotFoundError:
    pass

__all__ = ['H3DSScene']

from .base import BaseScene


class H3DSScene(BaseScene):

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
            views_config_id: str = '32',
            **kwargs):

        self.labels, self.K, self.RT = self.load_data(scene_root, views_config_id)
        super().__init__(scene_root, images_root, masks_root, pc_path, num_samples, random_zoom, random_shift,
                         image_size, target_views_indices, exclude_indices, None, **kwargs)

    def read_image(self, idx: int, ignore_exclude_indices: bool = False) -> Optional[torch.Tensor]:
        image = super().read_image(idx, ignore_exclude_indices)
        mask = super().read_mask(idx, ignore_exclude_indices)
        if image is not None and mask is not None:
            image = image.where(mask >= 0.5, torch.ones(1, 1, 1).expand(*image.shape))
        return image

    def get_paths(self, root: os.PathLike, names: List[str], ext: str = 'jpg'):  # Changes default value for ext
        if 'image' in Path(root).name:
            return super().get_paths(root, [f"img_{name}" for name in names], ext)
        elif 'mask' in Path(root).name:
            return super().get_paths(root, [f"mask_{name}" for name in names], ext)
        else:
            raise Exception("Invalid dir " + str(root))

    def get_intrinsics_ndc(self):
        return convert_screen_intrinsics_to_ndc((512, 512), self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2])

    def get_extrinsics(self, verbose=True) -> (List[str], torch.Tensor, torch.Tensor, torch.Tensor):
        R_cols, Ts = self.RT[:, :3, :3], self.RT[:, :3, -1]
        cam_poses = -np.einsum('bij,bi->bj', R_cols, Ts)

        return self.labels, torch.tensor(R_cols, dtype=torch.float32), torch.tensor(Ts, dtype=torch.float32), \
               torch.tensor(cam_poses, dtype=torch.float32)

    def load_data(self, scene_path, views_config_id):
        scene_path = Path(scene_path)
        h3ds_dir = scene_path.parent
        scene_id = scene_path.name
        h3ds = H3DS(path=h3ds_dir)
        views_idx = h3ds.helper._config['scenes'][scene_id]['default_views_configs'][views_config_id]
        labels = ['{0:04}'.format(idx) for idx in views_idx]
        _, images, masks, cameras = h3ds.load_scene(scene_id=scene_id, views_config_id=views_config_id)
        K = np.array([camera[0] for camera in cameras]).mean(axis=0)
        RT = np.array([np.linalg.inv(camera[1]) for camera in cameras])
        RT[:, :2, :] *= -1

        return labels, K, RT
