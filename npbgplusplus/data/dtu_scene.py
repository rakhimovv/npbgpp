import os
from pathlib import Path
from typing import Optional, Tuple, Union, List

import imageio
import numpy as np
import torch
from kornia.geometry.epipolar import KRt_from_projection

from .base import BaseScene

__all__ = ['DTUScene']

from ..utils.pytorch3d import convert_screen_intrinsics_to_ndc


class DTUScene(BaseScene):

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

        self.ids, self.K, self.RT, self.hw = self.load_data(scene_root)
        super().__init__(scene_root, images_root, masks_root, pc_path, num_samples, random_zoom, random_shift,
                         image_size, target_views_indices, exclude_indices, None, **kwargs)

    def read_image(self, idx: int, ignore_exclude_indices: bool = False) -> Optional[torch.Tensor]:
        image = super().read_image(idx, ignore_exclude_indices)
        mask = super().read_mask(idx, ignore_exclude_indices)
        if image is not None and mask is not None:
            image = image.where(mask >= 0.5, torch.ones(1, 1, 1).expand(*image.shape))
        return image

    def get_paths(self, root: os.PathLike, names: List[str], ext: str = 'png'):
        if 'image' in Path(root).name:
            return super().get_paths(root, [f"{name:06}" for name in names], ext)
        elif 'mask' in Path(root).name:
            return super().get_paths(root, [f"{name:03}" for name in names], ext)
        else:
            raise Exception("Invalid dir " + str(root))

    def get_intrinsics_ndc(self):
        return convert_screen_intrinsics_to_ndc(self.hw, self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2])

    def get_extrinsics(self, verbose=True) -> (List[str], torch.Tensor, torch.Tensor, torch.Tensor):
        R_cols, Ts = self.RT[:, :3, :3], self.RT[:, :3, -1]
        cam_poses = -torch.einsum('bij,bi->bj', R_cols, Ts)

        return self.ids, R_cols.clone().detach().type(torch.float32), Ts.clone().detach().type(torch.float32), \
               cam_poses.clone().detach().type(torch.float32)

    @staticmethod
    def load_data(scene_path):
        img_dir = os.path.join(scene_path, 'image')
        for p in sorted(os.listdir(img_dir)):
            if p.startswith('._'):
                os.remove(os.path.join(img_dir, p))
        img_file = os.path.join(img_dir, sorted(os.listdir(img_dir))[0])
        h, w, _ = imageio.imread(img_file).shape
        num = len(os.listdir(img_dir))

        all_cam = np.load(os.path.join(scene_path, "cameras.npz"))

        RT = []
        Ks = []
        for i in range(num):
            P = all_cam["world_mat_" + str(i)]
            P = P[:3]

            K, R, t = KRt_from_projection(torch.tensor(P)[None])
            K = K / K[:, 2, 2]

            RT.append(torch.cat([R, t], dim=-1))
            Ks.append(K)

        K = torch.cat(Ks).mean(dim=0)
        RT = torch.cat(RT)
        RT[:, :2, :] *= -1

        return torch.arange(num), K, RT, [h, w]
