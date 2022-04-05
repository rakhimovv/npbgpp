import logging

import numpy as np
import torch
from pytorch_lightning.metrics import Metric
from scipy import linalg

log = logging.getLogger(__name__)

__all__ = ['FID']


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the features of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over features, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over features for generated samples.
    -- sigma2: The covariance matrix over features, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        log.info(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calculate_feature_statistics(features):
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


class FID(Metric):
    def __init__(self, dims: int = 2048):
        super().__init__(dist_sync_on_step=False, compute_on_step=False)
        self.dims = dims
        self.add_state("fake_features", default=[], dist_reduce_fx="cat")
        self.add_state("real_features", default=[], dist_reduce_fx="cat")
        self.real_stats = None

    def update(self, inception_features: torch.Tensor, features_are_real: bool):
        if features_are_real:
            self.real_features.append(inception_features.detach().cpu())
        else:
            self.fake_features.append(inception_features.detach().cpu())
            if self.real_stats is not None and len(self.real_features) == 0:
                # append something dummy in order for sync op to work without errors
                self.real_features.append(inception_features.detach().cpu())

    def move_states_to_device(self, device: torch.device):
        for i in range(len(self.fake_features)):
            self.fake_features[i] = self.fake_features[i].to(device)
        for i in range(len(self.real_features)):
            self.real_features[i] = self.real_features[i].to(device)

    def compute(self):

        if isinstance(self.real_features, list):
            self.real_features = torch.cat(self.real_features, dim=0)

        if isinstance(self.fake_features, list):
            self.fake_features = torch.cat(self.fake_features, dim=0)

        if self.real_stats is None:
            self.real_features = self.real_features.cpu()
            m_real, s_real = calculate_feature_statistics(self.real_features.numpy())
            self.real_stats = (m_real, s_real)
        else:
            m_real, s_real = self.real_stats

        self.fake_features = self.fake_features.cpu()
        m_fake, s_fake = calculate_feature_statistics(self.fake_features.numpy())

        fid_value = calculate_frechet_distance(m_real, s_real, m_fake, s_fake)

        return fid_value
