from functools import partial
from typing import Callable, Union

import torch
from torchmetrics import Metric
from torchmetrics.functional.image.psnr import psnr
from torchmetrics.functional.image.ssim import ssim


class AccumulatedLoss(Metric):
    def __init__(self):
        super().__init__(dist_sync_on_step=False, compute_on_step=False)
        self.add_state("loss_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    # noinspection PyMethodOverriding
    def update(self, loss: torch.Tensor, weight: Union[int, float]):
        self.loss_sum += loss.sum()
        self.total += weight

    def compute(self):
        return self.loss_sum / self.total


class ImagePairMetric(AccumulatedLoss):
    def __init__(self, fun: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        super().__init__()
        self.fun = fun

    # noinspection PyMethodOverriding
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        super().update(loss=self.fun(pred, target).sum(), weight=pred.shape[0])


class SSIMMetric(ImagePairMetric):
    def __init__(self):
        super().__init__(lambda pred, target: ssim(pred, target, reduction='none').mean(dim=(1, 2, 3)))


class PSNRMetric(ImagePairMetric):
    def __init__(self):
        super().__init__(partial(psnr, dim=(1, 2, 3), data_range=1.0, reduction='none'))
