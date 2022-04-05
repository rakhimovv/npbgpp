from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..rasterizer.project import project_points
from ..rasterizer.scatter import compute_one_scale_visibility


@torch.no_grad()
def sample_points(points: torch.Tensor, num_iterations: int, seed: Optional[int] = None,
                  selection_method='farthest') -> Tuple[torch.Tensor, torch.Tensor]:
    # points: (n, 3)
    g = None
    if seed is not None:
        g = torch.Generator()
        g.manual_seed(123)
    if selection_method == 'farthest':
        selection = farthest_point_sample(points[None, :, :], npoint=num_iterations, gen=g)[0]
    elif selection_method == 'random':
        selection = torch.randperm(len(points), generator=g)[:num_iterations]
    else:
        raise ValueError
    return points[selection], selection


# from https://www.programmersought.com/article/8737853003/#12_farthest_point_sample_28
@torch.no_grad()
def farthest_point_sample(xyz, npoint, gen=None):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    assert N >= npoint
    centroids = xyz.new_zeros(B, npoint, dtype=torch.long)
    distance = xyz.new_ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), generator=gen, dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        # Update the i-th farthest point
        centroids[:, i] = farthest
        # Take the xyz coordinate of the farthest point
        centroid = xyz[batch_indices, farthest, :][:, None, :]
        # Calculate the Euclidean distance from all points in the point set to this farthest point
        dist = torch.sum((xyz - centroid) ** 2, -1)
        # Update distances to record the minimum distance of each point in the sample from all existing sample points
        mask = dist < distance
        distance[mask] = dist[mask]
        # Find the farthest point from the updated distances matrix, and use it as the farthest point for the next iteration
        farthest = torch.max(distance, -1)[1]
    return centroids


@torch.no_grad()
def sample_views_indices(scores: torch.Tensor, selection_method: str, selection_count: int) -> List[int]:
    """
    Samples #selection_count indexes based on selection_method using scores as weights
    Args:
        scores: torch.Tensor of shape (k, n)
        selection_method: 'best' | 'multinomial'
        selection_count: how many items are selected

    Returns:
        torch.Tensor of shape [selection_count]
    """
    assert selection_count >= 0
    assert selection_count <= scores.shape[0]
    if selection_count == 0:
        return []

    out = []
    for i in range(selection_count):
        normed_scores = scores / (scores.sum(dim=0, keepdim=True) + 1e-6)  # (k, n)
        normed_scores = normed_scores.sum(dim=1)  # (k,)
        if normed_scores.sum().item() == 0:
            normed_scores += 1e-1
        normed_scores = normed_scores / normed_scores.sum()  # (k,)

        if selection_method == 'best':
            ind = normed_scores.argmax().item()
        else:
            assert selection_method == 'multinomial'
            ind = torch.multinomial(normed_scores, 1, replacement=False).item()
        while ind in out:
            ind = torch.multinomial(normed_scores, 1, replacement=False).item()
        out.append(ind)
        scores = scores[:, scores[ind] == 0]

    return out


@torch.no_grad()
def calculate_view_selection_scores(points, R_row, T, fcl_ndc, prp_ndc,
                                    image_size: Tuple[int, int],
                                    cam_poses=None,
                                    optimal_baseline_angle=15, sigma_lower_baseline_angle=1,
                                    sigma_higher_baseline_angle=20, max_dist_ratio=1.5, use_angle=True,
                                    use_dist=True,
                                    ) -> torch.Tensor:
    r"""Calculates view selection scores as described in Section 4.1 of
      Yao et al (2018). MVSNet: Depth inference for unstructured multi-view stereo. ECCV.
    additionally taking into account the ratio of distances from the keypoint to the camera.
    The score of each point is multiplied by exp[-beta * (dist1/dist2 - 1)^2], where dist1 >= dist2,
    and exp[-beta * (max_dist_ratio - 1)^2] = 0.1.
    Parameters
    ----------
    points : torch.Tensor
        of shape [n, 3]. XYZ coordinates of keypoints in the world coordinate system
    R_row : torch.Tensor
        of shape [k, 3, 3].
    T : torch.Tensor
        of shape [k, 3]
    fcl_ndc: torch.Tensor
        of shape [k, 2]
    prp_ndc: torch.Tensor
        of shape [k, 2]
    image_size: Tuple[int, int]: height and width
    cam_poses: Optional[torch.Tensor]
        of shape [t, 3]. XYZ coordinates of the cameras in the world coordinate system
    optimal_baseline_angle : float
    sigma_lower_baseline_angle : float
    sigma_higher_baseline_angle : float
    max_dist_ratio : float
    Returns
    -------
    score : torch.Tensor
        of shape [t, k, n]
    """
    n = points.shape[0]
    k, _3 = T.shape

    data_cam_poses = (- R_row @ T[:, :, None]).squeeze(2)  # (k, 3)
    to_data_cam_poses = points.view(1, 1, n, 3) - data_cam_poses.view(1, k, 1, 3)  # (1, k, n, 3)

    if cam_poses is None:
        to_cam_poses = to_data_cam_poses.transpose(0, 1)  # (t=k, 1, n, 3)
    else:
        t, _3 = cam_poses.shape
        to_cam_poses = points.view(1, 1, n, 3) - cam_poses.view(t, 1, 1, 3)  # (t, 1, n, 3)

    assert use_angle or use_dist

    log_score: torch.Tensor = 0  # (t, k, n)

    # compute angle score
    if use_angle:
        eps = 1e-8
        l, r = -1 + eps, 1 - eps
        baseline_angle_cos = F.cosine_similarity(to_cam_poses, to_data_cam_poses, dim=3).clamp(l, r)  # (t, k, n)
        baseline_angle = baseline_angle_cos.acos()  # (t, k, n)
        del baseline_angle_cos
        baseline_angle = baseline_angle.rad2deg()  # (t, k, n)
        twice_var_angle = torch.where(
            baseline_angle <= optimal_baseline_angle,
            2 * sigma_lower_baseline_angle ** 2,
            2 * sigma_higher_baseline_angle ** 2
        )
        log_score_angle = - (baseline_angle - optimal_baseline_angle) ** 2 / twice_var_angle
        del baseline_angle, twice_var_angle
        log_score += log_score_angle

    # compute distance score
    # if use_dist:
    #     dist_to_data_cam_poses = to_data_cam_poses.norm(dim=3)  # (1, k, n)
    #     if cam_poses is None:
    #         dist_to_cam_poses = dist_to_data_cam_poses.transpose(0, 1)  # (t=k, 1, n)
    #     else:
    #         dist_to_cam_poses = to_data_cam_poses.norm(dim=3)  # (t, 1, n)
    #
    #     dist_ratio = dist_to_cam_poses / dist_to_data_cam_poses  # (t, k, n)
    #     del dist_to_data_cam_poses, dist_to_cam_poses
    #     dist_ratio = dist_ratio.where(dist_ratio > 1, dist_ratio.reciprocal())
    #     beta = np.log(10) / (max_dist_ratio - 1) ** 2
    #     log_score_dist = - beta * (dist_ratio - 1) ** 2
    #     del dist_ratio
    #     log_score += log_score_dist

    score = log_score.exp()  # (t, k, n)

    ndc_points, tracks = project_points(points[None, :, :].expand(k, n, 3), R_row, T, fcl_ndc, prp_ndc, image_size)
    tracks = compute_one_scale_visibility(
        ndc_points,
        tracks,
        image_size,
        scale=0.125,
        variant=1,
    ).bool()
    score = score.where(tracks[None, :, :], score.new_zeros(1, 1, 1))  # (t, k, n)
    if cam_poses is None:
        score[torch.arange(k), torch.arange(k), :] = 0.0
    return score
