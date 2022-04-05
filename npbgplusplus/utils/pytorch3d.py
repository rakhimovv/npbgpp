from typing import Tuple

import torch
from kornia import HomographyWarper


def get_ndc_positive_bounds(image_size: Tuple[int, int]) -> Tuple[float, float]:
    min_size, max_size = min(image_size), max(image_size)
    scale = max_size / min_size
    if min_size == image_size[1]:
        return 1.0, scale
    return scale, 1.0


def convert_screen_intrinsics_to_ndc(image_size, fx_screen, fy_screen, px_screen=None, py_screen=None):
    image_height, image_width = image_size
    image_height, image_width = int(image_height), int(image_width)
    s = min(image_height, image_width)
    fx_ndc = fx_screen * 2.0 / (s - 1)
    fy_ndc = fy_screen * 2.0 / (s - 1)
    px_ndc = - (px_screen - (image_width - 1) / 2.0) * 2.0 / (s - 1) if px_screen is not None else 0.0
    py_ndc = - (py_screen - (image_height - 1) / 2.0) * 2.0 / (s - 1) if px_screen is not None else 0.0
    return fx_ndc, fy_ndc, px_ndc, py_ndc, (image_height, image_width)


def center_crop(fx_ndc, fy_ndc, px_ndc, py_ndc, image_size, crop_size):
    """
    use the following facts:
    fx * cw + px = 0
    new_fx * cw + new_px = 0
    fx * left_world + px = (wn / w) * sx
    new_fx * left_world + new_px = nsx
    """
    h, w = image_size
    hn, wn = crop_size
    sx, sy = get_ndc_positive_bounds(image_size)
    nsx, nsy = get_ndc_positive_bounds(crop_size)
    left_world, top_world = ((wn / w) * sx - px_ndc) / fx_ndc, ((hn / h) * sy - py_ndc) / fy_ndc
    new_fx_ndc, new_fy_ndc = nsx / (left_world + px_ndc / fx_ndc), nsy / (top_world + py_ndc / fy_ndc)
    new_px_ndc, new_py_ndc = new_fx_ndc * px_ndc / fx_ndc, new_fy_ndc * py_ndc / fy_ndc
    return new_fx_ndc, new_fy_ndc, new_px_ndc, new_py_ndc


@torch.no_grad()
def create_ndc_intrinsic_matrix(
        fcl_ndc: torch.Tensor,  # (b, 2)
        prp_ndc: torch.Tensor  # (b, 2)
):
    K = fcl_ndc.new_zeros(fcl_ndc.shape[0], 3, 3)
    K[:, 0, 0] = fcl_ndc[:, 0]
    K[:, 1, 1] = fcl_ndc[:, 1]
    K[:, 0, 2] = prp_ndc[:, 0]
    K[:, 1, 2] = prp_ndc[:, 1]
    K[:, 2, 2] = 1
    return K


@torch.no_grad()
def create_screen_intrinsic_matrix(
        fcl_ndc: torch.Tensor,  # (b, 2)
        prp_ndc: torch.Tensor,  # (b, 2)
        image_size: Tuple[int, int]
):
    K = create_ndc_intrinsic_matrix(fcl_ndc, prp_ndc)
    image_height, image_width = image_size
    s = min(image_height, image_width)
    # check convert_screen_intrinsics_to_ndc to understand these manipulations
    K[:, 0, 0] = K[:, 0, 0] * (s - 1) / 2
    K[:, 1, 1] = K[:, 1, 1] * (s - 1) / 2
    K[:, 0, 2] = (image_width - 1) / 2.0 - K[:, 0, 2] * (s - 1) / 2
    K[:, 1, 2] = (image_height - 1) / 2.0 - K[:, 1, 2] * (s - 1) / 2
    return K


@torch.no_grad()
def create_kornia_intrinsic_matrix(
        fcl_ndc: torch.Tensor,  # (b, 2)
        prp_ndc: torch.Tensor,  # (b, 2)
        image_size: Tuple[int, int]
):
    K = create_ndc_intrinsic_matrix(fcl_ndc, prp_ndc)

    # NDC coordinates: min_size_ndc \in [-1, 1], max_size_ndc \in [-s, s] where s = max_size / min_size
    # Kornia coordinates: normalized_x \in [-1, 1], normalized_y \in [-1, 1]

    scale_x, scale_y = get_ndc_positive_bounds(image_size)  # float, float
    K[:, 0, :] /= scale_x
    K[:, 1, :] /= scale_y

    # in Kornia convention x-axis is pointing right, y-axis is pointing bottom => put minus before principal point
    K[:, 0, 2] *= -1
    K[:, 1, 2] *= -1

    return K


def warp_images(
        images,  # (b, c, *src_image_size)
        src_fcl_ndc: torch.Tensor,  # (b, 2)
        src_prp_ndc: torch.Tensor,  # (b, 2)
        src_image_size: Tuple[int, int],
        dst_fcl_ndc: torch.Tensor,  # (b, 2)
        dst_prp_ndc: torch.Tensor,  # (b, 2)
        dst_image_size: Tuple[int, int],
        cuda_sync: bool = True
) -> torch.Tensor:
    with torch.no_grad():
        homography_warper = HomographyWarper(*dst_image_size, align_corners=False)
        K_src = create_kornia_intrinsic_matrix(src_fcl_ndc, src_prp_ndc, src_image_size)  # (b, 3, 3)
        K_dst = create_kornia_intrinsic_matrix(dst_fcl_ndc, dst_prp_ndc, dst_image_size)  # (b, 3, 3)
        if torch.cuda.is_available() and cuda_sync:
            torch.cuda.synchronize()
        K_dst_inv = torch.inverse(K_dst)  # (b, 3, 3)
        H = torch.bmm(K_src, K_dst_inv)  # (b, 3, 3)
    # homography_warper.precompute_warp_grid(H)
    # homography_warper._warped_grid = homography_warper._warped_grid.float()
    # return homography_warper(images)
    return homography_warper(images, H)
