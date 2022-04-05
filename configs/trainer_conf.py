from dataclasses import dataclass
from typing import Any, Optional

from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()


@dataclass
class LightningTrainerConf:
    accelerator: Optional[str] = None
    accumulate_grad_batches: Any = 1
    amp_backend: str = 'native'
    amp_level: str = 'O2'
    auto_lr_find: Any = False
    auto_scale_batch_size: Any = False
    auto_select_gpus: bool = False
    benchmark: bool = False
    checkpoint_callback: bool = True
    check_val_every_n_epoch: int = 1
    default_root_dir: Optional[str] = None
    deterministic: bool = False
    fast_dev_run: bool = False
    gpus: Optional[Any] = None
    gradient_clip_val: float = 0.0
    gradient_clip_algorithm: str = "norm"
    ipus: Optional[int] = None
    limit_train_batches: float = 1.0
    limit_val_batches: float = 1.0
    limit_test_batches: float = 1.0
    log_gpu_memory: Optional[str] = None
    log_every_n_steps: int = 50
    prepare_data_per_node: bool = True
    profiler: Any = None
    overfit_batches: float = 0.0
    plugins: Any = None
    precision: int = 32
    max_epochs: int = 1000
    min_epochs: int = 1
    max_steps: Optional[int] = None
    min_steps: Optional[int] = None
    max_time: Any = None
    num_nodes: int = 1
    num_sanity_val_steps: int = 2
    num_processes: int = 1
    reload_dataloaders_every_n_epochs: int = 0
    replace_sampler_ddp: bool = True
    resume_from_checkpoint: Optional[str] = None
    sync_batchnorm: bool = False
    terminate_on_nan: bool = False
    tpu_cores: Optional[Any] = None
    track_grad_norm: Any = -1
    val_check_interval: float = 1.0
    weights_summary: Optional[str] = 'top'
    weights_save_path: Optional[str] = None
    move_metrics_to_cpu: bool = False
    multiple_trainloader_mode: str = "max_size_cycle"


@dataclass
class LightningTrainerProjectConf(LightningTrainerConf):
    accelerator: Optional[str] = 'ddp'
    log_gpu_memory: Optional[str] = 'min_max'
    profiler: Any = 'simple'
    num_sanity_val_steps: int = 0
    replace_sampler_ddp: bool = False


cs.store(group="trainer", name="default", node=LightningTrainerConf)
cs.store(group="trainer", name="project", node=LightningTrainerProjectConf)
