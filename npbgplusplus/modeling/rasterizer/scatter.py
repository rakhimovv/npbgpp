from typing import Tuple, Optional, Union, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_scatter
except ImportError:
    pass

from .project import project_points

__all__ = ['NearestScatterRasterizer', 'project_features', 'compute_one_scale_visibility']


class NearestScatterRasterizer(nn.Module):

    def __init__(self, descriptor_dim: int, cat_mask: bool = True,
                 scales: Union[float, List[float]] = (1.0,), ss_scale: List[int] = (1,),
                 invert_mask_values: bool = False, learnable: bool = True):
        super().__init__()
        self.cat_mask = cat_mask  # concatenate visible mask as last channels
        self.scales = scales
        self.ss_scale = ss_scale
        self.descriptor_dim = descriptor_dim
        if learnable:
            self.bg_feature = nn.Parameter(torch.zeros(1, descriptor_dim, 1, 1), requires_grad=True)
        else:
            self.register_buffer('bg_feature', torch.zeros(1, descriptor_dim, 1, 1))
        self.invert_mask_values = invert_mask_values

    @torch.jit.export
    def get_multiscale_empty_img(self, h: int, w: int) -> List[torch.Tensor]:
        if self.cat_mask:
            invalid_mask_value = 1.0 if self.invert_mask_values else 0.0
            empty_img = torch.cat((self.bg_feature.detach(),
                                   invalid_mask_value * torch.ones(1, 1, 1, 1, device=self.bg_feature.device,
                                                                   dtype=self.bg_feature.dtype)),
                                  dim=1)
        else:
            empty_img = self.bg_feature.detach()

        out = []
        for scale in self.scales:
            out.append(empty_img.expand(1, empty_img.size(1), int(scale * h), int(scale * w)))
        return out

    @torch.jit.export
    def _get_masked_img(self, img: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, _, h, w = img.shape
        if mask is not None:
            bg: torch.Tensor = self.bg_feature
            if img.dtype == torch.float16 and bg.dtype == torch.float32:
                bg = bg.half()
            img = img.where(mask, bg)
        if self.cat_mask:
            mask_value = 0.0 if self.invert_mask_values else 1.0
            invalid_mask_value = 1.0 if self.invert_mask_values else 0.0
            m = mask_value * torch.ones(1, 1, 1, 1, device=img.device, dtype=img.dtype).expand(b, 1, h, w)
            if mask is not None:
                m = m.where(mask, invalid_mask_value * torch.ones(1, 1, 1, 1, device=img.device, dtype=img.dtype))
            img = torch.cat([img, m], dim=1)
        return img

    @torch.jit.export
    def get_multi_scale_img(self, img: torch.Tensor, mask: Optional[torch.Tensor] = None, corrupt: bool = True) -> List[
        torch.Tensor]:

        b, _, h, w = img.shape
        # one_scale_img = self._get_masked_img(img, mask).detach()

        # if corrupt:
        #     valid_mask = torch.rand(b, 1, h, w, device=img.device) > 0.5
        #     if mask is not None:
        #         corrupted_mask = torch.logical_and(mask, valid_mask)
        #     else:
        #         corrupted_mask = valid_mask
        #     corrupted_img = self._get_masked_img(img, corrupted_mask).detach()
        # else:
        #     corrupted_img = one_scale_img

        img = img.detach()
        if mask is not None:
            mask = mask.float()

        out = []
        for scale in self.scales:
            scaled_img = F.interpolate(img, size=(int(scale * h), int(scale * w)), mode='nearest')
            scaled_mask = F.interpolate(mask, size=(int(scale * h), int(scale * w)),
                                        mode='nearest') if mask is not None else None
            if corrupt:
                valid_mask = torch.rand(b, 1, int(scale * h), int(scale * w), device=img.device) > scale / 2
                scaled_mask = torch.logical_and(scaled_mask, valid_mask) if scaled_mask is not None else valid_mask
            scaled_img = self._get_masked_img(scaled_img, scaled_mask).detach()
            out.append(scaled_img)
        return out

    def forward(
            self,
            points: torch.Tensor,  # b, n, 3
            descriptors: torch.Tensor,  # (b, c, n) if channel_first else (b, n, c)
            R_row: torch.Tensor,  # b, 3, 3 (R_row is applied to row vectors, i.e. x_view = x_world @ R_row + T)
            T: torch.Tensor,  # b, 3
            fcl_ndc: torch.Tensor,  # b, 2
            prp_ndc: torch.Tensor,  # b, 2
            image_size: Tuple[int, int],  # height and width
            valid_mask: Optional[torch.Tensor] = None,  # b, n (if None all points are valid),
            channel_first: bool = False,
            remove_non_unique: bool = True
    ) -> List[torch.Tensor]:
        return project_features(points, descriptors, R_row, T, fcl_ndc, prp_ndc, image_size,
                                valid_mask, self.scales, self.ss_scale, self.cat_mask, self.bg_feature, channel_first,
                                self.invert_mask_values, remove_non_unique)


class NearestScatterFilterRasterizer(NearestScatterRasterizer):

    def __init__(self, descriptor_dim: int, cat_mask: bool = True, invert_mask_values: bool = False,
                 init_error_thr=0.0, learnable: bool = True):
        super().__init__(descriptor_dim, cat_mask=True, scales=[1.0], learnable=learnable,
                         invert_mask_values=invert_mask_values)
        self.cat_mask = cat_mask
        self.register_buffer('errors_median', init_error_thr * torch.ones(descriptor_dim))
        self.register_buffer('errors_mean', init_error_thr * torch.ones(descriptor_dim))
        self.register_buffer('errors_q80', init_error_thr * torch.ones(descriptor_dim))
        self.register_buffer('errors_q90', init_error_thr * torch.ones(descriptor_dim))

    def forward(
            self,
            points: torch.Tensor,  # b, n, 3
            descriptors: torch.Tensor,  # (b, c, n) if channel_first else (b, n, c)
            R_row: torch.Tensor,  # b, 3, 3 (R_row is applied to row vectors, i.e. x_view = x_world @ R_row + T)
            T: torch.Tensor,  # b, 3
            fcl_ndc: torch.Tensor,  # b, 2
            prp_ndc: torch.Tensor,  # b, 2
            image_size: Tuple[int, int],  # height and width
            valid_mask: Optional[torch.Tensor] = None,  # b, n (if None all points are valid),
            channel_first: bool = False,
            remove_non_unique: bool = True
    ) -> List[torch.Tensor]:
        scales = (1.0, 0.5)
        ss_scale = (1, 1)
        out0, out1 = project_features(points, descriptors, R_row, T, fcl_ndc, prp_ndc, image_size,
                                      valid_mask, scales, ss_scale, self.cat_mask, self.bg_feature, channel_first,
                                      self.invert_mask_values, remove_non_unique)
        with torch.no_grad():
            out1 = F.interpolate(out1, size=out0.shape[-2:], mode='nearest')
            errors = F.l1_loss(out0[:, :-1], out1[:, :-1], reduction='none')
            if self.invert_mask_values:
                mask = (1.0 - out0[:, -1:]) * (1.0 - out1[:, -1:])
            else:
                mask = out0[:, -1:] * out1[:, -1:]
            if self.training:
                mask_bool = (mask == 1.0)  # fixme
                if torch.any(mask_bool):
                    for i in range(self.descriptor_dim):
                        valid_errors = torch.masked_select(errors[:, i:i + 1], mask_bool)
                        self.errors_median[i] = 0.99 * self.errors_median[i] + 0.01 * torch.median(valid_errors)
                        self.errors_mean[i] = 0.99 * self.errors_mean[i] + 0.01 * torch.mean(valid_errors)
                        self.errors_q80[i] = 0.99 * self.errors_q80[i] + 0.01 * torch.quantile(valid_errors, 0.80)
                        self.errors_q90[i] = 0.99 * self.errors_q90[i] + 0.01 * torch.quantile(valid_errors, 0.90)
            else:
                errors = errors * mask  # (b, c, h, w)
                out0[:, :-1] = out0[:, :-1].where(
                    torch.any(errors < self.errors_q90[None, :, None, None], dim=1, keepdim=True), self.bg_feature)
        if self.cat_mask:
            return [out0]
        return [out0[:, :-1]]


def project_features(
        points: torch.Tensor,  # b, n, 3
        features: torch.Tensor,  # (b, c, n) if channel_first else (b, n, c)
        R_row: torch.Tensor,  # b, 3, 3 (R_row is applied to row vectors, i.e. x_view = x_world @ R_row + T)
        T: torch.Tensor,  # b, 3
        fcl_ndc: torch.Tensor,  # b, 2
        prp_ndc: torch.Tensor,  # b, 2
        image_size: Tuple[int, int],  # height and width
        valid_mask: Optional[torch.Tensor] = None,  # b, n (if None all points are valid)
        scales: List[float] = (1.0,),
        ss_scale: List[int] = (1,),  # super sampling scale
        cat_mask: bool = True,
        bg_feature: torch.Tensor = torch.zeros(1, 1, 1, 1),
        channel_first: bool = False,
        invert_mask_values: bool = False,
        remove_non_unique: bool = True
) -> List[torch.Tensor]:
    assert isinstance(ss_scale, list) or isinstance(ss_scale, tuple), f"{ss_scale}, {type(ss_scale)}"
    assert isinstance(scales, list) or isinstance(scales, tuple), f"{scales}, {type(scales)}"
    assert len(ss_scale) == len(scales), f"{scales}, {ss_scale}"
    assert bg_feature.ndim == 4 and (bg_feature.shape[-2:] == (1, 1) or bg_feature.shape[-2:] == image_size)
    bg_feature = bg_feature.to(features.device)  # (b, c, 1 or height, 1 or width)

    ndc_points, v_mask = project_points(points, R_row, T, fcl_ndc, prp_ndc, image_size)  # (b, n, 3), (b, n)
    assert v_mask is not None  # make jit.script happy
    # ndc_points = ndc_points.float()
    if valid_mask is not None:
        v_mask = torch.logical_and(valid_mask, v_mask)

    if not channel_first:
        features = features.permute(0, 2, 1).contiguous()
    assert features.shape[1] == bg_feature.shape[1], f"{features.shape}, {bg_feature.shape}"

    # append ones to track empty pixels
    b, c, n = features.shape
    descriptors = torch.cat(
        (features, torch.ones(1, 1, 1, dtype=features.dtype, device=features.device).expand(b, 1, n)), dim=1)

    batch_size, descriptor_dim, _ = descriptors.shape
    image_height, image_width = image_size

    projected_features_list = []
    for i, scale in enumerate(scales):
        ss = ss_scale[i]
        height, width = ss * int(scale * image_height), ss * int(scale * image_width)

        with torch.no_grad():
            pixel_id = points_to_pixels(ndc_points, (height, width))
            pixel_id = pixel_id.where(v_mask,
                                      torch.tensor(height * width, device=pixel_id.device, dtype=pixel_id.dtype))

            # Calculate z-buffer
            z_buffer = ndc_points.new_full((points.shape[0], height * width + 1), np.inf)
            torch_scatter.scatter(ndc_points[:, :, 2], pixel_id, 1, reduce='min', out=z_buffer)

            # Mask out occluded points
            z_buffer[:, -1] = 0

            backprojected_min_depth = z_buffer.gather(1, pixel_id)

            not_occluded = (ndc_points[:, :, 2] == backprojected_min_depth) & (backprojected_min_depth < np.inf)

            pixel_id = pixel_id.where(not_occluded,
                                      torch.tensor(height * width, dtype=pixel_id.dtype, device=pixel_id.device))

            if remove_non_unique:
                z_buffer = ndc_points.new_full((points.shape[0], height * width + 1), np.inf)
                z_index = torch.arange(points.shape[1], device=points.device).expand(points.shape[0], -1).float()
                torch_scatter.scatter(z_index, pixel_id, 1, reduce='min', out=z_buffer)
                first = (z_index == z_buffer.gather(1, pixel_id))
                pixel_id = pixel_id.where(first,
                                          torch.tensor(height * width, dtype=pixel_id.dtype, device=pixel_id.device))

            # Project features
            pixel_id = pixel_id.unsqueeze(1).expand(-1, descriptor_dim, -1)

        # Simple scatter will work as duplicate points have been eliminated
        projected_features = descriptors.new_zeros((batch_size, descriptor_dim, height * width + 1))
        projected_features.scatter_(2, pixel_id, descriptors)

        # Crop the extra pixel and reshape
        projected_features = projected_features[:, :, :height * width].view(batch_size, descriptor_dim, height, width)

        if ss > 1:
            projected_features = F.avg_pool2d(projected_features, (ss, ss), (ss, ss))
            with torch.no_grad():
                m = projected_features[:, -1:, :, :]
                projected_features = projected_features / m.where(m != 0, torch.ones(1, 1, 1, 1, device=points.device))

        projected_features[:, :-1, :, :] = projected_features[:, :-1, :, :].where(
            projected_features[:, -1:] != 0,
            bg_feature
        )

        if not cat_mask:
            projected_features = projected_features[:, :-1, :, :]
        else:
            if invert_mask_values:
                projected_features[:, :-1, :, :] = 1.0 - projected_features[:, :-1, :, :]

        projected_features_list.append(projected_features)

    return projected_features_list


def ndc_to_screen(ndc_points: torch.Tensor, image_size: Tuple[int, int]):
    """
    Convert ndc coordinates to screen coordinates
    Args:
        ndc_points: torch.Tensor of shape [b, n, 3]
        image_size: Tuple[int, int] height and width in pixels

    Returns:
        screen_points: torch.Tensor of shape [b, n, 2]
    """
    image_height, image_width = image_size
    s = (min(image_size) - 1) / 2
    cx = - (image_width - 1) / 2
    cy = - (image_height - 1) / 2

    b, n, _3 = ndc_points.shape
    screen_points = ndc_points.new_zeros(b, n, 2)

    screen_points[:, :, 0] = s * ndc_points[:, :, 0] + cx
    screen_points[:, :, 1] = s * ndc_points[:, :, 1] + cy  # b, n

    # flip_xy
    screen_points *= -1

    return screen_points


def points_to_pixels(ndc_points: torch.Tensor, image_size: Tuple[int, int]):
    """
    Return pixels' ids in a frame of given size
    Args:
        ndc_points: torch.Tensor of shape [b, n, 3]
        image_size: Tuple[int, int] height and width in pixels

    Returns:
        pixel_ids: torch.Tensor of shape [b, n]
    """
    screen_points = ndc_to_screen(ndc_points, image_size)

    # round (e.g. screen_x=0.5 corresponds to pixel_x=0 and screen_x=0.6 corresponds to pixel_x=1)
    screen_x = (screen_points[:, :, 0] - 1e-6 + 0.5).long()
    screen_y = (screen_points[:, :, 1] - 1e-6 + 0.5).long()

    # compute pixel id
    image_height, image_width = image_size
    pixel_id = screen_y * image_width + screen_x  # b, n

    out_of_bounds_mask = ~((0 <= screen_x) & (screen_x < image_width) & (0 <= screen_y) & (screen_y < image_height))
    dummy = torch.tensor(image_height * image_width, dtype=pixel_id.dtype, device=pixel_id.device)
    pixel_id = pixel_id.where(~out_of_bounds_mask, dummy)

    return pixel_id


@torch.no_grad()
def compute_one_scale_visibility(
        ndc_points: torch.Tensor,
        valid_mask: Optional[torch.Tensor],
        image_size: Tuple[int, int],
        scale: float = 0.25,
        variant: int = 0):
    """
    Args:
        ndc_points: Tensor of shape [batch_size, points_n, 3]
        valid_mask: Tensor of shape [batch_size, points_n]
        image_size: (height, width)
        scale: float
        variant: int
    Returns:
        visibility_scores: Tensor of shape [batch_size, points_n]
    """
    height, width = int(scale * image_size[0]), int(scale * image_size[1])
    pixel_id = points_to_pixels(ndc_points, (height, width))  # b, n
    if valid_mask is not None:
        pixel_id = pixel_id.where(valid_mask, pixel_id.new_full([], height * width))
    closest = z_buffer_stat(pixel_id, ndc_points[:, :, 2], 'min')  # b, n

    if variant == 0:
        score = closest - ndc_points[:, :, 2]  # Negative number, where higher is better
        score = torch.exp(score)
    elif variant == 1:
        score = (ndc_points[:, :, 2] == closest).type(closest.dtype)
        if valid_mask is not None:
            score[~valid_mask] = 0
    else:
        raise ValueError

    return score


def z_buffer_stat(pixel_id, z, reduce):
    z_buffer = torch_scatter.scatter(z, pixel_id, dim=1, reduce=reduce)
    return torch.gather(z_buffer, 1, pixel_id)
