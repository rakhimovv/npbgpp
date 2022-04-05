import torch
from pytorch_lightning.metrics import Metric


class MeanLoss(Metric):
    def __init__(self):
        super().__init__(dist_sync_on_step=False, compute_on_step=False)
        self.add_state("loss_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, loss: torch.Tensor):
        self.loss_sum += loss.sum()
        self.total += loss.shape[0]

    def compute(self):
        return self.loss_sum / self.total
