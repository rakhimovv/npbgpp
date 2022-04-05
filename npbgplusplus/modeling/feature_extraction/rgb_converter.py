import torch
import torch.nn as nn

__all__ = ['RGBConverter']


class RGBConverter(nn.Module):

    def __init__(self, feature_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(feature_dim, 3)
        self.lin2 = nn.Linear(3, feature_dim)

    def forward(self, x):
        return 0.9 * torch.sigmoid(self.lin1(x))

    def reconstruct(self, rgb):
        return self.lin2(rgb)
