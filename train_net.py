import logging
import os
from typing import List, Iterable, Optional

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.loggers import LightningLoggerBase

from hydra.utils import instantiate
from configs import trainer_conf

log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    log.info(f"Current working directory: {os.getcwd()}")
    log.info(f"Original working directory: {hydra.utils.get_original_cwd()}")
    pl.seed_everything(3407, workers=True)

    loggers: Iterable[LightningLoggerBase] = [instantiate(a) for a in cfg.loggers]
    callbacks_cfg_list = cfg.callbacks.fit if not cfg.eval_only else cfg.callbacks.test
    callbacks: Optional[List[pl.callbacks.Callback]] = None
    if callbacks_cfg_list is not None:
        callbacks: Optional[List[pl.callbacks.Callback]] = [instantiate(a) for a in callbacks_cfg_list]
    model: pl.LightningModule = instantiate(cfg.system.system_class, cfg)
    trainer: pl.Trainer = pl.Trainer(**cfg.trainer, logger=loggers, callbacks=callbacks)

    if cfg.weights_path is not None:
        weights_absolute_path = hydra.utils.to_absolute_path(cfg.weights_path)
        assert os.path.exists(weights_absolute_path), f"{weights_absolute_path} does not exist"
        try:
            model.load_state_dict(torch.load(weights_absolute_path, map_location=model.device)['state_dict'],
                                  strict=False)
            log.info(f"Loaded weights {weights_absolute_path} successfully!")
        except Exception as e:
            log.info(f"Exception encountered while loading weights: {e}")
            pass

    # train / test
    if not cfg.eval_only:
        trainer.fit(model)
    else:
        trainer.test(model)


if __name__ == "__main__":
    main()
