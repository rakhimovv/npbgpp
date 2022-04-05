import logging
from typing import Optional, List
from typing import Tuple

import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

from npbgplusplus.utils import comm
from hydra.utils import instantiate

log = logging.getLogger(__name__)

__all__ = ['build_datasets', 'build_loaders']


def build_datasets(cfg: DictConfig, stage: str) -> List[Tuple[str, DictConfig, Dataset]]:
    assert stage in ['train', 'val', 'test']
    if stage not in cfg.datasets:
        return []

    datasets_params = cfg.datasets[stage]
    if datasets_params is None:
        return []

    assert isinstance(datasets_params, ListConfig)

    datasets = []

    for dataset_param in datasets_params:
        assert dataset_param.dataset_name not in [d[0] for d in datasets]
        dataset = instantiate(dataset_param.dataset_class)
        log.info(f"Dataset ({dataset_param.dataset_name}) in stage={stage} contains {len(dataset)} elements")
        datasets.append((dataset_param.dataset_name, dataset_param, dataset))

    return datasets


def build_loaders(cfg: DictConfig, datasets: List[Dataset], stage: str, sampler_name: str = "distributed", seed=None) -> \
        Optional[
            List[DataLoader]]:
    if len(datasets) == 0:
        return None

    assert stage in ['train', 'val', 'test']
    if stage == 'train':
        assert len(datasets) == 1

    num_workers = cfg.dataloader[stage].num_workers
    pin_memory = cfg.dataloader[stage].pin_memory
    batch_size = get_batch_size(cfg.dataloader[stage].total_batch_size)

    data_loaders = []
    shuffle = cfg.dataloader.train.shuffle if stage == 'train' else False
    drop_last = stage == 'train'

    for dataset in datasets:
        dataloader_kwargs = {}
        if torch.cuda.is_available():
            if sampler_name == "scene":
                assert isinstance(dataset, torch.utils.data.ConcatDataset)
                sampler = SceneDistributedSampler(sizes=[len(d) for d in dataset.datasets], shuffle=shuffle, seed=seed)
            else:
                assert sampler_name == "distributed"
                sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
            dataloader_kwargs['sampler'] = sampler
        else:
            dataloader_kwargs['shuffle'] = shuffle
        data_loaders.append(DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            # worker_init_fn=worker_init_reset_seed,
            **dataloader_kwargs
        ))

    return data_loaders


# def worker_init_reset_seed(worker_id):
#     # seed = np.random.randint(2 ** 31) + worker_id
#     seed = 100 + worker_id
#     np.random.seed(seed)
#     torch.set_rng_state(torch.manual_seed(seed).get_state())
#     random.seed(seed)


def get_batch_size(total_batch_size):
    world_size = comm.get_world_size()
    assert (total_batch_size > 0 and total_batch_size % world_size == 0), \
        f"Total batch size ({total_batch_size}) must be divisible by the number of gpus ({world_size})."
    batch_size = total_batch_size // world_size
    return batch_size


class SceneDistributedSampler(Sampler):

    def __init__(self, sizes: List[int], shuffle: bool = True, seed: Optional[int] = None):
        """
        Args:
            sizes (List[int]): list of scene sizes
            shuffle (bool): whether to shuffle the indices or not
            seed (int): the initial seed of the shuffle. Must be the same
                across all workers. If None, will use a random seed shared
                among workers (require synchronization among all workers).
        """
        assert isinstance(sizes, list)
        assert all([size > 0 for size in sizes])
        assert len(set(sizes)) == 1
        if seed is None:
            seed = comm.shared_random_seed()
        self._seed = int(seed)
        self._rank = comm.get_rank()
        self._world_size = comm.get_world_size()
        num_scenes = len(sizes)
        if self._world_size >= num_scenes:
            assert self._world_size % num_scenes == 0
            num_gpus_per_scene = self._world_size // num_scenes
            assert sizes[0] % num_gpus_per_scene == 0
        else:
            assert num_scenes % self._world_size == 0
        self.num_samples = sum(sizes) // self._world_size
        self._shuffle = shuffle
        if shuffle:
            self.g = torch.Generator()
            self.g.manual_seed(self._seed)

    def __iter__(self):
        inds = torch.arange(self.num_samples * self._rank, self.num_samples * (self._rank + 1))
        if self._shuffle:
            inds = inds[torch.randperm(self.num_samples, generator=self.g)]
        return iter(inds)

    def __len__(self):
        return self.num_samples
