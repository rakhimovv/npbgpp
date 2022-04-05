from typing import Optional

import torch
import torch.nn as nn


class Rotate90CCW(nn.Module):

    def forward(self, image: torch.Tensor, mask: Optional[torch.Tensor]):
        return rotate_90_ccw(image, mask)


def rotate_90_ccw(image: torch.Tensor, mask: Optional[torch.Tensor]):
    # image: b, c, h, w
    image = torch.flip(image.permute(0, 1, 3, 2), dims=(2,))
    if mask is not None:
        mask = torch.flip(mask.permute(0, 1, 3, 2), dims=(2,))
    return image, mask
