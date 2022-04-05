import logging
from typing import Tuple

import torch
from torch.utils.data import Dataset

from .base import BaseScene
from ..modeling.feature_extraction.view_selection import sample_views_indices, calculate_view_selection_scores, \
    sample_points
from ..modeling.rasterizer.project import project_points
from ..modeling.rasterizer.scatter import compute_one_scale_visibility

__all__ = ['ViewSceneWrapper']

log = logging.getLogger(__name__)


class ViewSceneWrapper(Dataset):
    """
    Wrapper for PeopleScene dataset to include neighboring views based on baseline angle score and distance to cam pos
    """

    def __init__(
            self,
            scene_dataset: BaseScene = None,
            selection_method: str = "multinomial",
            selection_count: int = 3,
            num_iterations: int = 500
    ):
        assert selection_method in ["multinomial", "best", ""]
        self.scene = scene_dataset
        self.selection_method = selection_method
        self.selection_count = selection_count
        self.num_iterations = num_iterations
        assert selection_count <= self.scene.input_views_indices_mask.sum()

    def load_point_cloud(self, include_rgb: bool = False):
        return self.scene.load_point_cloud(include_rgb)

    def unload_point_cloud(self):
        return self.scene.unload_point_cloud()

    def load_images(self):
        return self.scene.load_images()

    def unload_images(self):
        return self.scene.unload_images()

    def read_image(self, idx: int, ignore_exclude_indices: bool = False):
        return self.scene.read_image(idx, ignore_exclude_indices)

    def read_mask(self, idx: int, ignore_exclude_indices: bool = False):
        return self.scene.read_mask(idx, ignore_exclude_indices)

    def __getattr__(self, item):
        return self.scene.__getattribute__(item)

    @property
    def cache_idx(self):
        return self.scene.cache_idx

    @cache_idx.setter
    def cache_idx(self, value):
        self.scene.cache_idx = value

    def __len__(self):
        return self.scene.__len__()

    def __getitem__(self, idx: int):
        item = self.scene.__getitem__(idx)

        if self.selection_count > 0 and self.selection_method != "":
            idx = item['idx']
            scores = self.get_selection_scores(idx, item)  # (?, n)
            mask = self.scene.input_views_indices_mask.detach().clone()
            mask[idx] = 0  # the target view must not use the same view as input
            view_idxs = torch.arange(len(self.scene.names))[mask]
            assert len(scores) == len(view_idxs)
            selection_idx = view_idxs[
                sample_views_indices(scores, self.selection_method, min(self.selection_count, len(scores)))
            ]
            images = torch.stack([self.scene.read_image(i) for i in selection_idx], dim=0)
            R_rows = self.scene.R_cols[selection_idx].transpose(1, 2)
            Ts = self.scene.Ts[selection_idx]
            fcl_ndc = torch.tensor([self.scene.fx_ndc, self.scene.fy_ndc], dtype=torch.float32)
            prp_ndc = torch.tensor([self.scene.px_ndc, self.scene.py_ndc], dtype=torch.float32)
            item['views_data'] = (images, R_rows, Ts, fcl_ndc, prp_ndc)

        return item

    def get_selection_scores(self, idx: int, item: dict) -> torch.Tensor:
        target_points = self.scene.load_point_cloud()['points'].view(1, -1, 3)  # (1, N, 3)
        target_ndc_points, target_visible_mask = project_points(
            target_points,
            item['R_row'][None, :, :],
            item['T'][None, :],
            item['fcl_ndc'][None, :],
            item['prp_ndc'][None, :],
            self.image_size
        )
        if target_visible_mask.sum().item() == 0:
            return torch.full((self.scene.input_views_indices_mask.sum() - 1, 0), fill_value=1.0, dtype=torch.float32)
        target_points = target_points.view(-1, 3)[target_visible_mask.view(-1)]  # (n, 3)
        target_ndc_points = target_ndc_points.view(-1, 3)[target_visible_mask.view(-1)]  # (n, 3)
        del target_visible_mask
        interest_mask = compute_one_scale_visibility(
            target_ndc_points.view(1, -1, 3),
            None,
            self.image_size, scale=0.125, variant=1
        ).bool()
        if interest_mask.sum().item() == 0:
            return torch.full((self.scene.input_views_indices_mask.sum() - 1, 0), fill_value=1.0, dtype=torch.float32)
        target_points = target_points[interest_mask.view(-1)]
        target_ndc_points = target_ndc_points[interest_mask.view(-1)]
        del interest_mask
        if self.selection_method == "multinomial":
            sample_num_points = min(self.num_iterations, target_ndc_points.shape[0])
            _, points_idx = sample_points(target_ndc_points, sample_num_points, selection_method="farthest")
            del target_ndc_points
            target_points = target_points[points_idx]
            del points_idx
        scores = calculate_view_selection_scores(
            target_points,
            *self.get_input_view_cameras(idx)[:-1],
            self.scene.old_image_size,
            item['cam_pos'][None, :],
            use_dist=False,
            sigma_lower_baseline_angle=5,
            sigma_higher_baseline_angle=15,
        )  # (1, k, n)
        scores = scores[0]
        return scores

    def get_input_view_cameras(
            self,
            idx=None,
            dtype=torch.float32,
            device: torch.device = 'cpu',
            expand_intrinsics: bool = False) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = self.scene.input_views_indices_mask.detach().clone()
        if idx is not None:
            mask[idx] = 0  # the target view must not use the same view as input
        view_idxs = torch.arange(len(self.scene.names))[mask]
        R_row = self.scene.R_cols[mask].type(dtype).transpose(1, 2).to(device)  # (k, 3, 3)
        T = self.scene.Ts[mask].type(dtype).to(device)  # (k, 3)
        k = R_row.shape[0] if expand_intrinsics else 1
        fcl_ndc = torch.tensor([[self.scene.fx_ndc, self.scene.fy_ndc]], dtype=dtype, device=device).expand(k, 2)
        prp_ndc = torch.tensor([[self.scene.px_ndc, self.scene.py_ndc]], dtype=dtype, device=device).expand(k, 2)
        return R_row, T, fcl_ndc, prp_ndc, view_idxs
