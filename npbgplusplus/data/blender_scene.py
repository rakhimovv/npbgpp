__all__ = ['BlenderScene']

import json
import os
from typing import Optional, Tuple, Union, List

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .base import BaseScene
from ..utils.pytorch3d import convert_screen_intrinsics_to_ndc


class BlenderScene(BaseScene):
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
            white_bg=True,  # If the image has alpha channel, replace empty space with white
            **kwargs):
        self.split_paths, self.poses, self.intrinsic_data, self.i_split = load_blender_data(scene_root)
        self.white_bg = white_bg
        super().__init__(scene_root, images_root, masks_root, pc_path, num_samples, random_zoom, random_shift,
                         image_size, target_views_indices, exclude_indices, None, **kwargs)

    def get_paths(self, root: os.PathLike, names: List[str], ext: str = 'png'):  # Changes default value for ext
        return super().get_paths(root, self.split_paths, ext)

    def read_image(self, idx: int, ignore_exclude_indices: bool = False) -> Optional[torch.Tensor]:
        if not ignore_exclude_indices and self.exclude_indices is not None and idx in self.exclude_indices:
            return None
        if len(self.images) > 0:
            return self.images[idx]
        image = np.array(imageio.imread(self.imgs_paths[idx])).astype(np.float32) / 255.0
        if self.white_bg:
            image = image[..., :3] * image[..., -1:] + (1.0 - image[..., -1:])
        else:
            image = image[..., :3]
        image: Image = Image.fromarray((image * 255.0).astype(np.uint8))
        image = image.resize(self.old_image_size[::-1], Image.BILINEAR)
        image = torch.tensor(np.array(image)).permute(2, 0, 1).contiguous()
        image = image / 255
        return image

    def read_mask(self, idx: int, ignore_exclude_indices: bool = False) -> Optional[torch.Tensor]:
        if (not ignore_exclude_indices and self.exclude_indices is not None and idx in self.exclude_indices):
            return None
        if len(self.masks) > 0:
            return self.masks[idx]
        image = np.array(imageio.imread(self.imgs_paths[idx])).astype(np.float32) / 255.0
        mask = image[..., -1][None]
        mask = torch.tensor(mask, dtype=torch.float32)
        mask = F.interpolate(mask[None], size=self.old_image_size, mode='bilinear', align_corners=False)[0]
        return mask

    def get_intrinsics_ndc(self):
        image_height, image_width, focal = self.intrinsic_data
        return convert_screen_intrinsics_to_ndc((image_height, image_width), focal, focal)

    def get_extrinsics(self, verbose=True):
        labels = ['-'.join(path.split('/')[1:]) for path in self.split_paths]
        R_cols, Ts, cam_poses = [], [], []
        for RT in self.poses:
            RT = np.linalg.inv(RT)
            RT[[0, 2], :] *= -1

            R_col = RT[:3, :3]
            T = RT[:3, 3]
            cam_pos = -R_col.T @ T

            R_cols.append(R_col)
            Ts.append(T)
            cam_poses.append(cam_pos)
        return labels, torch.tensor(R_cols, dtype=torch.float32), torch.tensor(Ts, dtype=torch.float32), \
               torch.tensor(cam_poses, dtype=torch.float32)


def load_blender_data(basedir):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    split_paths = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        poses = []
        for frame in meta['frames'][::1]:
            split_paths.append(frame['file_path'])
            poses.append(np.array(frame['transform_matrix']))
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + poses.shape[0])
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    poses = np.concatenate(all_poses, 0)

    h, w = imageio.imread(os.path.join(basedir, split_paths[0] + '.png')).shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * w / np.tan(.5 * camera_angle_x)

    return split_paths, poses, [h, w, focal], i_split
