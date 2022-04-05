from typing import Union, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

__all__ = ['MeanAggregator', 'SphereAggregator']


class MeanAggregator(nn.Module):

    # Note (careful!):
    # when calling update or compute between reset calls
    # batch indexes must correspond to the same scenes every time

    def __init__(self, weighted: bool):
        super(MeanAggregator, self).__init__()
        self.weighted = weighted
        self.reset()

    def reset(self, b: Optional[int] = None, n: Optional[int] = None, c: Optional[int] = None,
              device: torch.device = 'cpu'):
        if b is not None and n is not None and c is not None:
            self.features_sum = torch.zeros(b, c, n, device=device)
            self.features_total = torch.zeros(b, n, device=device)
        else:
            self.features_sum: Union[float, torch.Tensor] = 0.0  # [b, channels, n]
            self.features_total: Union[float, torch.Tensor] = 0.0  # [b, n]
        self.aggregated_features: Optional[torch.Tensor] = None  # [b, channels, n]
        self.visible_mask: Optional[torch.Tensor] = None  # [b, n]
        torch.cuda.empty_cache()

    def get_num_scenes(self) -> int:
        if self.features_sum is not None and isinstance(self.features_sum, torch.Tensor):
            return self.features_sum.shape[0]
        return 0

    def update(self, features: torch.Tensor, valid_mask: torch.Tensor, scores: torch.Tensor, points, R_row, T,
               index=None):
        """
        Aggregates features using mean reduction
        Note (careful!): batch indexes should correspond to the same scene every time update is called between reset calls
        Args:
            features: Tensor of shape [b, views, channels, n]
            valid_mask: Tensor of shape [b, views, n]
            scores: Tensor of shape [b, views, n]
            points: Tensor of shape [b, views, n, 3]
            R_row: Tensor of shape [b, views, 3, 3]
            T: Tensor of shape [b, views, 3]
            index: Optional[int] if provided update part of the state corresponding to the particular scene
        """
        self.aggregated_features = None
        self.visible_mask = None
        if self.weighted:
            weights = torch.where(valid_mask, scores, scores.new_zeros(1, 1, 1))  # e^weights for softmax
        else:
            weights = valid_mask.type(features.dtype)

        if index is None:
            self.features_sum += (features * weights[:, :, None, :]).sum(dim=1)
            self.features_total += weights.sum(dim=1)
        else:
            self.features_sum[index] = self.features_sum[index] + (features * weights[:, :, None, :]).sum(dim=1)[0]
            self.features_total[index] = self.features_total[index] + weights.sum(dim=1)[0]

    def forward(self, points, R_row: torch.Tensor, T: torch.Tensor, fcl_ndc: torch.Tensor,
                prp_ndc: torch.Tensor, index: Optional[int] = None):
        """
            args are provided to possibly compute and use such things as directions or distances
            from points to target camera pos

            Returns:
                descriptors: Tensor of shape [b, descriptors_dim, n]
                valid_mask: Tensor of shape [b, n]
        """
        # todo to save memory and time it makes sense to compute features only for the points of interest
        # todo but ndc_points would be calculated twice in this case, both in rasterizer and here
        # todo in this case we need to pass already calculated those to rasterizer to save memory and time
        if points.shape[1] == 0 or self.get_num_scenes() == 0:
            return None, None

        assert isinstance(self.features_sum, torch.Tensor) and isinstance(self.features_total, torch.Tensor)

        if self.aggregated_features is None:
            # since mean aggregation does not use target view information we can cache results
            self.visible_mask: torch.Tensor = self.features_total > 1e-8  # Epsilon threshold
            self.aggregated_features = self.features_sum / torch.where(self.visible_mask[:, None, :],
                                                                       self.features_total[:, None, :],
                                                                       self.features_sum.new_ones(1, 1, 1))
        if index is None:
            return self.aggregated_features, self.visible_mask

        agg_f = self.aggregated_features[index:index + 1]
        vm = self.visible_mask[index:index + 1]
        if index is not None:
            agg_f = agg_f.expand(points.shape[0], -1, -1)
            vm = vm.expand(points.shape[0], -1)

        return agg_f, vm


class SphereAggregator(nn.Module):

    def __init__(self, type='learnable_v', m=6, learnable_alpha=False,
                 n_harmonic_functions=3, width=16, depth=1, tanh=False, learnable_freq: bool = False):
        # in nex: hidden=64
        # omega_0 = pi / 2
        super().__init__()

        self.type = type
        self.m = m

        if 'learnable' in self.type:
            if self.type == 'learnable_v':
                dim = 3
                embedding_dim = n_harmonic_functions * 2 * dim
            elif self.type == 'learnable_vd':
                dim = 3
                embedding_dim = n_harmonic_functions * 2 * dim + 1
            elif self.type == 'learnable_vo':
                dim = 6
                embedding_dim = n_harmonic_functions * 2 * dim
            else:
                assert self.type == 'learnable_vdo'
                dim = 6
                embedding_dim = n_harmonic_functions * 2 * dim + 1

            self.harmonic_embedding = HarmonicEmbedding(n_harmonic_functions, learnable_freq)
            self.mlp = [
                nn.Linear(embedding_dim, width),
                nn.LeakyReLU(inplace=True)
            ]
            for i in range(depth):
                self.mlp += [nn.Linear(width, width), nn.LeakyReLU(inplace=True)]
            self.mlp += [nn.Linear(width, m)]
            if tanh:
                self.mlp += [nn.Tanh()]
            self.mlp = nn.Sequential(*self.mlp)

        self.reset()

        self.learnable_alpha = learnable_alpha
        if learnable_alpha:
            self.alpha = nn.Parameter(np.log(np.exp(1) - 1) * torch.ones((self.m, 1)))
        else:
            self.register_buffer("alpha", torch.ones((self.m, 1)))

    def reset(self, b: Optional[int] = None, n: Optional[int] = None, c: Optional[int] = None,
              device: torch.device = 'cpu'):
        if b is not None and n is not None and c is not None:
            self.xt_sum: Optional[torch.Tensor] = torch.zeros(b, n, self.m, device=device)
            self.y_sum: Optional[torch.Tensor] = torch.zeros(b, c, n, device=device)
            self.xt_dot_x_sum = torch.zeros(b, n, self.m, self.m, device=device)
            self.xt_dot_y_sum = torch.zeros(b, n, self.m, c, device=device)
            self.total = torch.zeros(b, n, 1, 1, device=device)
        else:
            self.xt_sum: Optional[torch.Tensor] = 0.0  # [b, n, m]
            self.y_sum: Optional[torch.Tensor] = 0.0  # [b, c, n]
            self.xt_dot_x_sum: Optional[torch.Tensor] = 0.0  # [b, n, m, m]
            self.xt_dot_y_sum: Optional[torch.Tensor] = 0.0  # [b, n, m, c]
            self.total: Union[float, torch.Tensor] = 0.0  # [b, n, 1, 1]

        self.visible_mask: Optional[torch.Tensor] = None  # [b, n]
        self.beta = None
        self.ym = None
        torch.cuda.empty_cache()

    def get_num_scenes(self) -> int:
        if self.xt_sum is not None and isinstance(self.xt_sum, torch.Tensor):
            return self.xt_sum.shape[0]
        return 0

    def update(self, features: torch.Tensor, valid_mask: torch.Tensor, scores: torch.Tensor, points, R_row, T,
               index: Optional[int] = None):
        """
        Aggregates features using mean reduction
        Note (careful!): batch indexes should correspond to the same scene every time update is called between reset calls
        Args:
            features: Tensor of shape [b, views, channels, n]
            valid_mask: Tensor of shape [b, views, n]
            scores: Tensor of shape [b, views, n]
            points: Tensor of shape [b, views, n, 3]
            R_row: Tensor of shape [b, views, 3, 3]
            T: Tensor of shape [b, views, 3]
            index: Optional[int] if provided update part of the state corresponding to the particular scene
        """
        b, k, c, n = features.shape
        if index is not None:
            assert b == 1

        self.beta, self.visible_mask, self.ym = None, None, None

        with torch.no_grad():
            mask = torch.logical_and(valid_mask, scores > 0)  # (b, k, n)
            cam_pos = -torch.einsum('bkzl, bkl -> bkz', R_row, T)  # (b, k, 3)
            view_dirs = points - cam_pos[:, :, None, :]  # (b, k, n, 3)
            del cam_pos
            distances = view_dirs.norm(dim=3, keepdim=True)  # (b, k, n, 1)
            view_dirs /= (distances + 1e-7)  # (b, k, n, 3)
            optical_axis = R_row[:, :, :, 2].view(b, k, 1, 3).expand(b, k, n, 3)
            view_dirs = view_dirs.view(b * k * n, 3)[mask.view(-1), :]
            distances = distances.view(b * k * n, 1)[mask.view(-1), :]
            optical_axis = optical_axis.contiguous().view(b * k * n, 3)[mask.view(-1), :]
            basis = features.new_zeros(b, k, n, self.m)

        basis.view(b * k * n, self.m)[mask.view(-1), :] = self.calculate_basis(view_dirs, distances, optical_axis)
        del view_dirs, distances, optical_axis
        torch.cuda.empty_cache()
        features = features.where(mask[:, :, None, :], features.new_zeros(1, 1, 1, 1))  # (b, k, c, n)

        xt_sum = basis.sum(dim=1)  # (b, n, m)
        y_sum = features.sum(dim=1)  # (b, c, n)

        xt_dot_x_sum = torch.einsum('bknt, bknm -> bntm', basis, basis)  # t=m
        xt_dot_y_sum = torch.einsum('bknm, bkcn -> bnmc', basis, features)
        view_num = mask.type(features.dtype).sum(dim=1).view(b, n, 1, 1)

        if index is None:
            self.xt_sum = self.xt_sum + xt_sum
            self.y_sum = self.y_sum + y_sum
            self.xt_dot_x_sum = self.xt_dot_x_sum + xt_dot_x_sum
            self.xt_dot_y_sum = self.xt_dot_y_sum + xt_dot_y_sum
            self.total = self.total + view_num
        else:
            self.xt_sum[index] = self.xt_sum[index] + xt_sum[0]
            self.y_sum[index] = self.y_sum[index] + y_sum[0]
            self.xt_dot_x_sum[index] = self.xt_dot_x_sum[index] + xt_dot_x_sum[0]
            self.xt_dot_y_sum[index] = self.xt_dot_y_sum[index] + xt_dot_y_sum[0]
            self.total[index] = self.total[index] + view_num[0]

    def calculate_beta_and_visible_mask(self):
        if self.beta is not None and self.visible_mask is not None and self.ym is not None:
            return
        assert self.total.shape[1] > 0
        self.visible_mask = self.total[:, :, 0, 0] > 0  # (b, n, 1, 1)
        device, dtype = self.xt_dot_x_sum.device, self.xt_dot_x_sum.dtype
        total = torch.where(self.total > 0, self.total, self.total.new_ones(1, 1, 1, 1))  # (b, n, 1, 1)
        # alpha = 1 / total  # torch.pow(total, np.log(0.02) / np.log(50))
        alpha = self.alpha if not self.learnable_alpha else F.softplus(self.alpha)
        reg_term = (alpha * torch.eye(self.m, device=device, dtype=dtype)).view(1, 1, self.m, self.m)
        b, n = total.shape[:2]
        xt_dot_x_mean_inv = self.total.new_ones(b, n, self.m, self.m)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        xt_dot_x_mean_inv.view(b * n, self.m, self.m)[self.visible_mask.view(b * n), :, :] = torch.inverse(
            self.xt_dot_x_sum.view(b * n, self.m, self.m)[self.visible_mask.view(b * n), :, :] / total.view(b * n, 1,
                                                                                                            1)[
                                                                                                 self.visible_mask.view(
                                                                                                     b * n), :,
                                                                                                 :] + reg_term.view(1,
                                                                                                                    self.m,
                                                                                                                    self.m) / total.view(
                b * n, 1, 1)[self.visible_mask.view(b * n), :, :]
        )
        xt_dot_y_mean = self.xt_dot_y_sum / total
        b, n, _1, _1 = total.size()
        self.ym = self.y_sum / total.view(b, 1, n)  # (b, c, n)
        xt_dot_ym_mean = torch.einsum('bnmr, bcnr -> bnmc', self.xt_sum[..., None],
                                      self.ym[..., None]) / total  # r=1

        self.beta = torch.where(
            self.visible_mask[:, :, None, None],
            torch.einsum('bntm, bnmc -> bntc', xt_dot_x_mean_inv, xt_dot_y_mean - xt_dot_ym_mean).float(),  # t=m
            xt_dot_y_mean.new_zeros(1, 1, 1, 1)
        )

    def forward(self, points, R_row: torch.Tensor, T: torch.Tensor, fcl_ndc: Optional[torch.Tensor],
                prp_ndc: Optional[torch.Tensor], index: Optional[int] = None) -> Tuple[
        Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
            args are provided to possibly compute and use such things as directions or distances
            from points to target camera pos

            Returns:
                descriptors: Tensor of shape [b, descriptors_dim, n]
                valid_mask: Tensor of shape [b, n]
        """
        if points.shape[1] == 0 or self.get_num_scenes() == 0:
            return None, None

        if self.beta is None:
            self.calculate_beta_and_visible_mask()

        with torch.no_grad():
            cam_pos = -torch.einsum('bzl, bl -> bz', R_row, T)  # (b, 3)
            view_dirs = points - cam_pos[:, None, :]  # (b, n, 3)
            distances = view_dirs.norm(dim=2, keepdim=True)  # (b, n, 1)
            view_dirs /= (distances + 1e-7)  # (b, n, 3)
            b, n, _3 = view_dirs.shape
            optical_axis = None
            if self.type == 'learnable_vo' or self.type == 'learnable_vdo':
                optical_axis = R_row[:, :, 2].view(b, 1, 3).expand(b, n, 3)
            basis = self.calculate_basis(view_dirs, distances, optical_axis).detach()  # b, n, m

        beta = self.beta if index is None else self.beta[index:index + 1]  # b, n, m, c
        ym = self.ym if index is None else self.ym[index:index + 1]
        visible_mask = self.visible_mask if index is None else self.visible_mask[index:index + 1]
        if index is not None:
            beta = beta.expand(basis.shape[0], -1, -1, -1)
            ym = ym.expand(basis.shape[0], -1, -1)
            visible_mask = visible_mask.expand(basis.shape[0], -1)

        features = torch.einsum('bnmc, bnm -> bcn', beta, basis) + ym

        return features, visible_mask

    def calculate_basis(self, view_dirs: torch.Tensor, distances: torch.Tensor,
                        optical_axis: torch.Tensor) -> torch.Tensor:
        # view_dirs: (..., 3)
        # distances(..., 1)
        # optical_axis_dirs: (..., 3)
        # return: (..., m)

        if self.type == 'spherical':
            assert self.m == 4
            x = view_dirs[..., 0]
            y = view_dirs[..., 1]
            z = view_dirs[..., 2]
            out = torch.stack([
                torch.full_like(x, 0.282095, device=x.device, dtype=x.dtype),
                0.488603 * x,
                0.488603 * y,
                0.488603 * z,
                # 1.092548 * x * z,
                # 1.092548 * y * z,
                # 1.092548 * x * y,
                # 0.315392 * (3 * z ** 2 - 1),
                # 0.546274 * (x ** 2 - y ** 2)
            ], dim=-1)  # (..., m)
        elif self.type == 'learnable_v':
            out = self.mlp(self.harmonic_embedding(view_dirs))
        elif self.type == 'learnable_vd':
            out = self.mlp(torch.cat([distances, self.harmonic_embedding(view_dirs)], dim=-1))
        elif self.type == 'learnable_vo':
            out = self.mlp(
                torch.cat([self.harmonic_embedding(view_dirs), self.harmonic_embedding(optical_axis)], dim=-1))
        else:
            assert self.type == 'learnable_vdo'
            out = self.mlp(
                torch.cat([distances, self.harmonic_embedding(view_dirs), self.harmonic_embedding(optical_axis)],
                          dim=-1))

        return out


class HarmonicEmbedding(torch.nn.Module):
    def __init__(self, n_harmonic_functions=3, learnable_freq: bool = True):
        """
        Given an input tensor `x` of shape [minibatch, ... , dim],
        the harmonic embedding layer converts each feature
        in `x` into a series of harmonic features `embedding`
        as follows:
            embedding[..., i*dim:(i+1)*dim] = [
                sin(x[..., i]),
                sin(2*x[..., i]),
                sin(4*x[..., i]),
                ...
                sin(2**(self.n_harmonic_functions-1) * x[..., i]),
                cos(x[..., i]),
                cos(2*x[..., i]),
                cos(4*x[..., i]),
                ...
                cos(2**(self.n_harmonic_functions-1) * x[..., i])
            ]

        Note that `x` is also premultiplied by `omega0` before
        evaluating the harmonic functions.
        """
        super().__init__()
        self.n_harmonic_functions = n_harmonic_functions
        self.learnable_freq = learnable_freq
        if self.learnable_freq:
            self.omega0 = nn.Parameter(torch.log(torch.exp(torch.tensor(0.1)) - 1), requires_grad=True)
        else:
            self.register_buffer(
                'frequencies',
                (np.pi / 2) * (2.0 ** torch.arange(n_harmonic_functions)),
            )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: tensor of shape [..., dim]
        Returns:
            embedding: a harmonic embedding of `x`
                of shape [..., n_harmonic_functions * dim * 2]
        """
        if self.learnable_freq:
            frequencies = F.softplus(self.omega0) * (
                    2.0 ** torch.arange(self.n_harmonic_functions, device=x.device, dtype=x.dtype))
        else:
            frequencies = self.frequencies
        embed = (x[..., None] * frequencies).view(*x.shape[:-1], self.n_harmonic_functions * x.shape[-1])
        return torch.cat((embed.sin(), embed.cos()), dim=-1)
