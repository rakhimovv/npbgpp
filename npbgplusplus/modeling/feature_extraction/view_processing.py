from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from kornia import get_rotation_matrix2d, warp_affine

from ..rasterizer.project import project_points
from ...utils.pytorch3d import get_ndc_positive_bounds, create_ndc_intrinsic_matrix


def align_views_vertically(views_images, views_R_row, views_T, views_fcl_ndc, views_prp_ndc,
                           views_points=None, pad=True, v=(0, 1, 0), img_size=None, pad_value=0.0,
                           warp_padding_mode='zeros'):
    """
    Aligns views images so that the principal component lies vertically in the image plane.
    Args:
        views_images: Optional[torch.Tensor]
            of shape [k, 3, H, W].
        views_R_row: torch.Tensor
            of shape [k, 3, 3].
        views_T: torch.Tensor
            of shape [k, 3]
        views_fcl_ndc: torch.Tensor
            of shape [k, 2]
        views_prp_ndc: torch.Tensor
            of shape [k, 2]
        views_points: torch.Tensor
            of shape [k, n, 3]. XYZ coordinates of keypoints in the world coordinate system
        pad: bool; Whether to pad the images before rotating to preserve the full image
        v: tuple (x, y, z); Vector in world coordinates to be aligned vertically.
            If None, the first principal component of the point cloud is used.
        img_size: tuple (h, w); if views_images is not provided use it
    Returns:
        views_images, new_R_row, new_T, new_fcl_ndc, new_prp_ndc
    """

    if views_images is not None:
        height, width = views_images.shape[-2:]
    else:
        height, width = img_size

    if pad:
        p2d, views_fcl_ndc, views_prp_ndc = complete_padding(height, width, views_fcl_ndc, views_prp_ndc)
        height += p2d[2] + p2d[3]
        width += p2d[0] + p2d[1]
        if views_images is not None:
            views_images = torch.nn.functional.pad(views_images, p2d, 'constant', pad_value)
    else:
        p2d = None

    with torch.no_grad():
        # Get direction in image plane
        if v is None:
            assert views_points is not None
            _, _, v = torch.pca_lowrank(views_points[:, ::1000], q=1)  # can estimate direction without using all points
            ref_vector_origin = views_points.mean(axis=1)
        else:
            v = views_R_row.new_tensor(v)  # (1, 3)
            optical_axis = views_R_row[:, :, 2]  # (b, 3)
            cam_pos = -torch.einsum('bzl, bl -> bz', views_R_row, views_T)  # (b, 3)
            ref_vector_origin = cam_pos + optical_axis * 10  # (b, 3)

        ref_vector_world = torch.stack([ref_vector_origin, ref_vector_origin + v.view(-1, 3)], dim=1)
        ref_vector_ndc, _ = project_points(ref_vector_world, views_R_row, views_T, views_fcl_ndc, views_prp_ndc)
        ref_vector_dir_ndc = ref_vector_ndc[:, 1, :2] - ref_vector_ndc[:, 0, :2]
        degrees = 90 - torch.rad2deg(torch.atan2(ref_vector_dir_ndc[:, 1], ref_vector_dir_ndc[:, 0]))
        cx, cy = (width - 1) / 2, (height - 1) / 2  # Subtract 1 to be consistent with rasterization

        batch_size = views_points.shape[0] if views_points is not None else views_R_row.shape[0]

        M = get_rotation_matrix2d(degrees.new_tensor((cx, cy)).expand(batch_size, 2),
                                  angle=-degrees,
                                  scale=degrees.new_ones(views_fcl_ndc.shape))

        aligned_images = None
        if views_images is not None:
            aligned_images = warp_affine(views_images.float(), M.float(), (height, width),
                                         padding_mode=warp_padding_mode)

        ext_rot = torch.zeros_like(views_R_row)
        ext_rot[:, :2, :2] = M[:, :2, :2]
        ext_rot[:, 2, 2] = 1

        # K^T R^T K^-T to compensate for intrinsic transform
        Kt = create_ndc_intrinsic_matrix(views_fcl_ndc, views_prp_ndc).transpose(-2, -1)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        Kt_inv = torch.inverse(Kt)
        ext_rot = Kt @ ext_rot.transpose(-2, -1) @ Kt_inv

        return aligned_images, views_R_row @ ext_rot, (views_T.view(-1, 1, 3) @ ext_rot).view(-1, 3), \
               views_fcl_ndc, views_prp_ndc, p2d, degrees, ref_vector_world


@torch.no_grad()
def pad_to_size(height, width, fcl_ndc, prp_ndc, new_size) -> (Tuple[int, int, int, int], torch.Tensor, torch.Tensor):
    scale_x, scale_y = get_ndc_positive_bounds((height, width))
    bound = prp_ndc.new_tensor((scale_x, scale_y)).view(1, 2)

    # use the fact that:
    # fcl_ndc * top_left_corner_view + prp_ndc = bound
    # fcl_ndc * bottom_right_corner_view + prp_ndc = -bound
    top_left_corner_view = (bound - prp_ndc) / fcl_ndc
    bottom_right_corner_view = (-bound - prp_ndc) / fcl_ndc

    new_h, new_w = new_size
    dh, dw = (new_h - height), (new_w - width)
    p2d = (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2)

    device = fcl_ndc.device
    scale_x, scale_y = get_ndc_positive_bounds((new_h, new_w))
    new_top_left_corner_ndc = torch.tensor([(width / 2) / (width / 2 + p2d[0]) * scale_x,
                                            (height / 2) / (height / 2 + p2d[2]) * scale_y], device=device).view(1, 2)
    new_bottom_right_corner_ndc = -torch.tensor([(width / 2) / (width / 2 + p2d[1]) * scale_x,
                                                 (height / 2) / (height / 2 + p2d[3]) * scale_y], device=device).view(1,
                                                                                                                      2)
    # use the fact that:
    # new_fcl_ndc * top_left_corner_view + new_prp_ndc = new_top_left_corner_ndc
    # new_fcl_ndc * bottom_right_corner_view + new_prp_ndc = new_bottom_right_corner_ndc
    new_fcl_ndc = (new_top_left_corner_ndc - new_bottom_right_corner_ndc) / (
            top_left_corner_view - bottom_right_corner_view + 1e-8)
    new_prp_ndc = new_bottom_right_corner_ndc - new_fcl_ndc * bottom_right_corner_view

    return p2d, new_fcl_ndc, new_prp_ndc


@torch.no_grad()
def complete_padding(height, width, fcl_ndc, prp_ndc) -> (Tuple[int, int, int, int], torch.Tensor, torch.Tensor):
    diag = int(np.ceil(np.linalg.norm((height, width))))
    return pad_to_size(height, width, fcl_ndc, prp_ndc, (diag, diag))
