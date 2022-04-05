from typing import Optional, Tuple

import torch


# from pytorch3d.renderer import PerspectiveCameras

@torch.no_grad()
def project_points(
        points: torch.Tensor,  # b, n, 3
        R_row: torch.Tensor,  # b, 3, 3 (R_row is applied to row vectors, i.e. x_view = x_world @ R_row + T)
        T: torch.Tensor,  # b, 3
        fcl_ndc: torch.Tensor,  # b, 2
        prp_ndc: torch.Tensor,  # b, 2
        image_size: Optional[Tuple[int, int]] = None
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # cameras = PerspectiveCameras(fcl_ndc, prp_ndc, R_row, T, device=points.device)
    # ndc_points = cameras.transform_points(points, eps=1e-7)

    view_points = torch.bmm(points, R_row) + T.view(-1, 1, 3)
    K = fcl_ndc.new_zeros(len(fcl_ndc), 3, 3)
    K[:, 0, 0] = fcl_ndc[:, 0]
    K[:, 1, 1] = fcl_ndc[:, 1]
    K[:, 0, 2] = prp_ndc[:, 0]
    K[:, 1, 2] = prp_ndc[:, 1]
    K[:, 2, 2] = 1
    K = K.transpose(-2, -1).expand(view_points.shape[0], K.shape[1], K.shape[2])
    ndc_points = torch.bmm(view_points, K)
    ndc_points[:, :, :2] /= (ndc_points[:, :, 2:] + 1e-7)

    v_mask: Optional[torch.Tensor] = None

    if image_size is not None:
        min_size, max_size = min(image_size), max(image_size)
        min_size_index = 0 if min_size == image_size[1] else 1
        max_size_index = 1 - min_size_index
        s = max_size / min_size
        # 1/z>0 <=> z > 1
        v_mask = (ndc_points[:, :, 2] > 1) & \
                 (-1 <= ndc_points[:, :, min_size_index]) & (ndc_points[:, :, min_size_index] <= 1) & \
                 (-s <= ndc_points[:, :, max_size_index]) & (ndc_points[:, :, max_size_index] <= s)

    # uncomment if using PerspectiveCameras
    # z = torch.where(ndc_points[:, :, 2] != 0, ndc_points[:, :, 2], points.new_ones(1, 1))
    # ndc_points[:, :, 2] = ndc_points[:, :, 2].where(ndc_points[:, :, 2] == 0, 1 / z)

    return ndc_points, v_mask
