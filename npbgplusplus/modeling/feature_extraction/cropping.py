import math

import torch

from ..rasterizer.project import project_points
from ...utils.pytorch3d import get_ndc_positive_bounds, warp_images, create_ndc_intrinsic_matrix

__all__ = ['calculate_center_ndc_and_pix_size', 'extract_regions']


@torch.no_grad()
def calculate_center_ndc_and_pix_size(
        ndc_points: torch.Tensor,  # (b, n, 3)
        mask: torch.Tensor,  # (b, n)
        h: int,
        w: int,
        margin: int = 30
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    ndc_points = ndc_points[:, :, :2]  # we do not need depth here
    scale_x, scale_y = get_ndc_positive_bounds((h, w))

    left_border = masked_max_2d(ndc_points[:, :, 0], mask, -scale_x, scale_x)  # (b,)
    right_border = masked_min_2d(ndc_points[:, :, 0], mask, -scale_x, scale_x)  # (b,)
    bottom_border = masked_min_2d(ndc_points[:, :, 1], mask, -scale_y, scale_y)  # (b,)
    top_border = masked_max_2d(ndc_points[:, :, 1], mask, -scale_y, scale_y)  # (b,)

    h_ndc = top_border - bottom_border  # (b,)
    w_ndc = left_border - right_border  # (b,)

    h_pix = (h_ndc / (2.0 * scale_y)) * h  # (b,)
    w_pix = (w_ndc / (2.0 * scale_x)) * w  # (b,)

    h_pix = (h_pix + margin + 0.5).long().clamp(1, h).type(w_pix.dtype)  # (b,)
    w_pix = (w_pix + margin + 0.5).long().clamp(1, w).type(w_pix.dtype)  # (b,)

    center_ndc = torch.stack([(left_border + right_border) / 2, (top_border + bottom_border) / 2], dim=1)  # (b, 2)

    return center_ndc, h_pix, w_pix


@torch.no_grad()
def extract_regions(points, images, R_row, T, fcl_ndc, prp_ndc, original_interest_mask,
                    max_size=None, avoid_scaling_down=True):
    """
    Note: if there is no interesting point inside an image, the original view is returned
    Args:
        points: torch.Tensor
            of shape [batch_size, points_n, 3]
        images: torch.Tensor
            of shape [batch_size, channels_n, H, W]
        R_row: torch.Tensor
            of shape [batch_size, 3, 3]
        T: torch.Tensor
            of shape [batch_size, 3]
        fcl_ndc: torch.Tensor
            of shape [batch_size, 2]
        prp_ndc: torch.Tensor
            of shape [batch_size, 2]
        interest_mask: torch.Tensor
            of shape [batch_size, points]
        max_size: Optional[Union[int, Tuple[int, int]]
        avoid_scaling_down: bool
    Returns:
        (image_crops [b, *target_size], new_fcl_ndc [b, 2], new_prp_ndc [b, 2])
    """
    assert max_size is None or isinstance(max_size, int) or (
            (isinstance(max_size, tuple) or isinstance(max_size, list)) and len(
        max_size) == 2), f"{max_size}, {type(max_size)}"
    b, _, h, w = images.shape

    # compute center_ndc, h_pix, w_pix
    ndc_points, mask = project_points(points, R_row, T, fcl_ndc, prp_ndc, (h, w))  # (b, n, 3), (b, n)
    mask = torch.logical_and(mask, original_interest_mask)  # (b, n)
    center_ndc, h_pix, w_pix = calculate_center_ndc_and_pix_size(ndc_points, mask, h, w)  # (b, 2), (b, 2), (b, 2)
    # del ndc_points, mask
    center_ndc = torch.cat([center_ndc, center_ndc.new_ones(1, 1).expand(b, 1)], dim=1)  # (b, 3)
    if avoid_scaling_down and max_size is not None:
        clamp_size = (max_size, max_size) if isinstance(max_size, int) else max_size  # Tuple[int, int]
        h_pix, w_pix = h_pix.clamp(max=clamp_size[0]), w_pix.clamp(max=clamp_size[1])  # (b, 2), (b, 2)

    # compute K_src_inv
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    K_src_inv = torch.inverse(create_ndc_intrinsic_matrix(fcl_ndc, prp_ndc))  # (b, 3, 3)

    # compute center_world
    center_world = torch.bmm(K_src_inv, center_ndc[:, :, None])[:, :, 0]  # (b, 3)

    # compute max_world
    scale_x, scale_y = get_ndc_positive_bounds((h, w))  # float, float
    left_border = center_ndc[:, 0] + (w_pix / w) * scale_x  # (b,)
    top_border = center_ndc[:, 1] + (h_pix / h) * scale_y  # (b,)
    max_ndc = torch.stack([left_border, top_border], dim=1)  # (b, 2)
    max_ndc = torch.cat([max_ndc, max_ndc.new_ones(b, 1)], dim=1)  # (b, 3)
    max_world = torch.bmm(K_src_inv, max_ndc[:, :, None])[:, :, 0]  # (b, 3)
    # del scale_x, scale_y, left_border, top_border, max_ndc

    # compute h_pix_max, w_pix_max
    if isinstance(max_size, tuple) or isinstance(max_size, list):
        assert max_size[0] >= 2 and max_size[1] >= 2
        h_pix_max, w_pix_max = max_size  # int, int
    else:
        h_pix_max, w_pix_max = int(h_pix.max().item()), int(w_pix.max().item())  # int, int
        if isinstance(max_size, int):
            assert max_size >= 2
            a = w_pix_max / h_pix_max  # float
            h_pix_max, w_pix_max = (max_size, math.ceil(max_size * a)) if a < 1 else (math.ceil(max_size / a), max_size)

    # compute new_max_ndc
    s = torch.stack([w_pix_max / w_pix, h_pix_max / h_pix], dim=1).min(dim=1).values  # (b,)
    s = s.clamp(0.0, 1.0)  # (b,) avoid upsampling the region of interest to match crop size
    h_pix, w_pix = s * h_pix, s * w_pix  # (b,), (b,)
    new_scale_x, new_scale_y = get_ndc_positive_bounds((h_pix_max, w_pix_max))  # float, float
    new_left_border = (w_pix / w_pix_max) * new_scale_x  # (b,)
    new_top_border = (h_pix / h_pix_max) * new_scale_y  # (b,)
    new_max_ndc = torch.stack([new_left_border, new_top_border], dim=1)  # (b, 2)
    # del s, h_pix, w_pix, new_left_border, new_top_border, new_scale_x, new_scale_y

    # compute new_fcl_ndc and new_prp_ndc
    # solve:
    # new_fcl_ndc * center_world + new_prp_ndc = 0
    # new_fcl_ndc * max_world + new_prp_ndc = new_max_ndc
    new_fcl_ndc = new_max_ndc / (max_world[:, :2] - center_world[:, :2] + 1e-8)  # (b, 2)
    new_prp_ndc = - new_fcl_ndc * center_world[:, :2]  # (b, 2)

    # crops: (b, 3, h_pix_max, w_pix_max)
    crops = warp_images(images, fcl_ndc, prp_ndc, (h, w), new_fcl_ndc, new_prp_ndc, (h_pix_max, w_pix_max))

    return crops, new_fcl_ndc, new_prp_ndc, center_ndc


def masked_min_2d(tensor: torch.Tensor, mask: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    return torch.where(
        torch.any(mask, dim=1),
        torch.min(tensor.where(mask, tensor.new_tensor(upper).view(1, 1)), dim=1).values,
        tensor.new_tensor(lower),
    )


def masked_max_2d(tensor: torch.Tensor, mask: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    return torch.where(
        torch.any(mask, dim=1),
        torch.max(tensor.where(mask, tensor.new_tensor(lower).view(1, 1)), dim=1).values,
        tensor.new_tensor(upper),
    )
