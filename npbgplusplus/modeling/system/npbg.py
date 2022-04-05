import logging
import os
from typing import Dict

import hydra
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig

from .base import NPBGBase
from .. import SphereAggregator
from ...utils.comm import is_main_process

log = logging.getLogger(__name__)


class NPBG(NPBGBase):

    def __init__(self, cfg: DictConfig):
        super(NPBG, self).__init__(cfg)
        cache_descriptor_dim = self.descriptor_dim
        if 'aggregator' in self.hparams.system and self.hparams.system.aggregator is not None:
            self.aggregator = instantiate(self.hparams.system.aggregator)
            self.m = self.aggregator.m
            cache_descriptor_dim *= (self.m + 1)
            self.aggregator.xt_sum = torch.Tensor([1])

        self.cached_descriptors = nn.Parameter(
            torch.zeros(self.max_scenes_per_train_epoch, int(self.hparams.system.max_points), cache_descriptor_dim),
            requires_grad=True)

        # Define where to save or load from descriptors
        self.descriptors_save_dir = None
        if self.hparams.system.descriptors_save_dir is not None:
            self.descriptors_save_dir = hydra.utils.to_absolute_path(self.hparams.system.descriptors_save_dir)
            self.create_folder(self.descriptors_save_dir)
            # Remove existing descriptors from previous trainings
            if self.hparams.system.descriptors_save_dir != self.hparams.system.descriptors_pretrained_dir:
                [os.remove(os.path.join(self.descriptors_save_dir, f)) for f in os.listdir(self.descriptors_save_dir)]
            log.info(f"Descriptors if updated will be cached into {self.descriptors_save_dir}")

        self.descriptors_pretrained_dir = None
        if self.hparams.system.descriptors_pretrained_dir is not None:
            self.descriptors_pretrained_dir = hydra.utils.to_absolute_path(
                self.hparams.system.descriptors_pretrained_dir)
            assert os.path.exists(self.descriptors_pretrained_dir), f"{self.descriptors_pretrained_dir}"
            log.info(
                f"During training descriptors will be preloaded from {self.descriptors_pretrained_dir} if not cached yet in {self.descriptors_save_dir}")

    def get_descriptors(self, points, batch, cache_idx, valid_mask):
        cached = torch.index_select(input=self.cached_descriptors[:, :points.shape[1]], dim=0, index=cache_idx)
        if hasattr(self, 'aggregator'):
            y_beta = cached.view(*cached.shape[:2], self.m + 1, -1)  # b, n, 1+m, c
            self.aggregator.ym = y_beta[:, :, 0].transpose(-2, -1)  # b, c, n
            self.aggregator.beta = y_beta[:, :, 1:]
            descriptors = self.aggregator.forward(points, batch['R_row'], batch['T'], batch['fcl_ndc'],
                                                  batch['prp_ndc'], None)[0].transpose(-2, -1)
            invalid = (self.aggregator.beta.view(*y_beta.shape[:2], -1) == 0).all(dim=-1)
            return descriptors, valid_mask & ~invalid
        else:
            return cached, valid_mask

    def extract_forward_args_from_batch(self, batch, **kwargs):
        device = batch['cache_idx'].device
        cache_idx = batch['cache_idx']
        points = torch.index_select(input=self.cached_points, dim=0, index=cache_idx.cpu()).to(device)  # (b, n, 3)
        valid_mask = torch.index_select(input=self.cached_valid_mask, dim=0, index=cache_idx.cpu()).to(device)
        descriptors, valid_mask = self.get_descriptors(points, batch, cache_idx, valid_mask)
        channel_first = False
        return points, descriptors, batch['R_row'], batch['T'], batch['fcl_ndc'], \
               batch['prp_ndc'], batch['image_size'][0].cpu().tolist(), channel_first, valid_mask

    def training_step(self, batch: Dict, batch_idx: int, optimizer_idx: int):
        super(NPBG, self).training_step(batch, batch_idx)

    def get_scene_name_without_stage_suffix(self, dataset_name) -> str:
        suffix = dataset_name.split('_')[-1]
        if suffix in ['finetune', 'val', 'holdout']:
            scene_name = '_'.join(dataset_name.split('_')[:-1])
        else:
            scene_name = dataset_name
        return scene_name

    def update_cache(self, stage: str):
        super(NPBG, self).update_cache(stage)

        # load cached descriptors or load pretrained descriptors or init them
        for cache_idx, scene_idx in enumerate(self.cache_idx2scene_idx[stage]):
            dataset_name, _, scene = self.datasets[stage][scene_idx]
            scene_name = self.get_scene_name_without_stage_suffix(dataset_name)
            points = scene.point_cloud['points']
            n_points = points.shape[0]
            if self.descriptors_save_dir is not None and os.path.exists(
                    os.path.join(self.descriptors_save_dir, f"{scene_name}.pth")):
                descriptors = torch.load(
                    os.path.join(self.descriptors_save_dir, f"{scene_name}.pth"),
                    map_location=self.device,
                ).type(self.cached_descriptors.dtype)
            elif self.descriptors_pretrained_dir is not None:
                descriptors = torch.load(
                    os.path.join(self.descriptors_pretrained_dir, f"{scene_name}.pth"),
                    map_location=self.device,
                ).type(self.cached_descriptors.dtype)
            else:
                descriptors = torch.zeros(n_points, self.cached_descriptors.shape[-1],
                                          device=self.device, dtype=self.cached_descriptors.dtype)

            self.cached_descriptors.data[cache_idx, :n_points] = descriptors

    def on_train_epoch_start(self):
        super(NPBG, self).on_train_epoch_start()

        # reset the state of optimizer for descriptors
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]

        opt_descriptors = optimizers[0]
        for group in opt_descriptors.param_groups:
            for p in group['params']:
                opt_descriptors.state[p] = {}

    def our_train_epoch_end(self, *args, **kwargs):
        if is_main_process():
            for scene_idx in self.cached_train_scene_idxs:
                dataset_name, _, scene = self.datasets['train'][scene_idx]
                cache_idx = scene.cache_idx
                scene_name = self.get_scene_name_without_stage_suffix(dataset_name)
                n_points = self.cached_valid_mask[cache_idx].long().sum().item()
                descriptors = self.cached_descriptors[cache_idx, :n_points].data.detach().cpu()
                if self.descriptors_save_dir is not None:
                    torch.save(descriptors, os.path.join(self.descriptors_save_dir, f"{scene_name}.pth"))
        super(NPBG, self).our_train_epoch_end(*args, **kwargs)

    def configure_optimizers(self):
        optimizers = [torch.optim.RMSprop([
            {"params": [self.cached_descriptors], "lr": 1e-1},
            {"params": self.rasterizer.parameters(), "lr": 1e-1},  # bg feature
            {"params": self.rgb_converter.parameters(), "lr": 1e-4}
        ])]
        if not self.hparams.system.freeze_refiner:
            optimizers.append(torch.optim.Adam(self.refiner.parameters(), lr=1e-4))
        return optimizers

    def _shared_epoch_end(self, stage):
        super(NPBG, self)._shared_epoch_end(stage)
        self.cached_descriptors.data.zero_()

    # def val_dataloader(self):
    #     if self.datasets['val'] is None:
    #         self.datasets['val'] = build_datasets(self.hparams, 'val')
    #
    #     if len(self.datasets['val']) == 0:
    #         return None
    #
    #     # the scenes in val must coincide with train (because only last training scenes have up-to-date descriptors)
    #     # except hyperparams like image_size, num_samples, etc.
    #     assert len(self.datasets['train']) == len(self.datasets['val'])
    #     for i, (train_dataset_name, _, _) in enumerate(self.datasets['train']):
    #         val_dataset_name = self.datasets['val'][i][0]
    #         # assert train_dataset_name == val_dataset_name
    #     self.cache_idx2scene_idx['val'] = self.cache_idx2scene_idx['train']
    #
    #     datasets = []
    #     for cache_idx, scene_idx in enumerate(self.cache_idx2scene_idx['val']):
    #         # we set attribute cache_idx for the dataset in order for items in batch have cache_idx,
    #         # so we can perform torch.index_select from cached points
    #         self.datasets['val'][scene_idx][2].cache_idx = cache_idx
    #         datasets.append(self.datasets['val'][scene_idx][2])
    #
    #     loaders = build_loaders(self.hparams, datasets, 'val')
    #     return loaders
