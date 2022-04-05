import os
from typing import Optional, Tuple, Union, List

import torch

__all__ = ['ColmapScene']

from .base import BaseScene
from .colmap_read_model import read_cameras_binary, read_images_binary
from ..utils.pytorch3d import convert_screen_intrinsics_to_ndc


class ColmapScene(BaseScene):

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
                         image_size, target_views_indices, exclude_indices, None)

    def get_paths(self, root: os.PathLike, names: List[str], ext: str = 'png'):  # Changes default value for ext
        return super().get_paths(root, names, ext)

    def get_intrinsics_ndc(self):
        cameras_file = os.path.join(self.scene_root, 'sparse/cameras.bin')
        cameras = read_cameras_binary(cameras_file)
        camera = next(iter(cameras.values()))  # Images share intrinsics
        fx, fy, cx, cy = camera.params
        return convert_screen_intrinsics_to_ndc((camera.height, camera.width), fx, fy, cx, cy)

    def get_extrinsics(self, verbose=True) -> (List[str], torch.Tensor, torch.Tensor, torch.Tensor):
        images_file = os.path.join(self.scene_root, 'sparse/images.bin')
        img_data = read_images_binary(images_file)
        labels = []
        R_cols, Ts, cam_poses = [], [], []
        for img in img_data.values():
            R_col = img.qvec2rotmat()
            T = img.tvec
            R_col[:2, :] *= -1
            T[:2] *= -1
            cam_pos = -R_col.T @ T

            labels.append(img.name.split('.')[0])
            R_cols.append(R_col)
            Ts.append(T)
            cam_poses.append(cam_pos)
        return labels, torch.tensor(R_cols, dtype=torch.float32), torch.tensor(Ts, dtype=torch.float32), \
               torch.tensor(cam_poses, dtype=torch.float32)
