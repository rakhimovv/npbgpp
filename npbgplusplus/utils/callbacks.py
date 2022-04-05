import os

from pytorch_lightning import Callback, LightningModule


class CheckpointEveryEpoch(Callback):

    def on_epoch_end(self, trainer, pl_module: LightningModule) -> None:
        dirpath = './checkpoints'
        trainer.save_checkpoint(os.path.join(dirpath, f"epoch{trainer.current_epoch}.ckpt"))
        trainer.save_checkpoint(os.path.join(dirpath, f"last.ckpt"))
