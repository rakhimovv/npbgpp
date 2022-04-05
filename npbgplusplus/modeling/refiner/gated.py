import torch
import torch.nn as nn
from kornia import median_blur


# based on "Contextual Residual Aggregation for Ultra High-Resolution Image Inpainting"
# https://arxiv.org/abs/2005.09704

def get_norm_layer(normalization: str, num_channels) -> nn.Module:
    if normalization == 'batch':
        return nn.BatchNorm2d(num_channels)
    elif normalization == 'instance':
        return nn.InstanceNorm2d(num_channels, affine=True)
    else:
        assert normalization == 'identity'
        return nn.Identity()


class GatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1,
                 act_fun=nn.ELU,
                 normalization: str = 'batch',
                 padding_mode: str = 'zeros',
                 median_filter: bool = False):
        super().__init__()

        n_pad_pxl = int(dilation * (kernel_size - 1) / 2)

        self.block = nn.ModuleDict({
            'conv_f': nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation,
                                padding=n_pad_pxl, groups=groups, padding_mode=padding_mode),
            'act_f': act_fun(),
            'conv_m': nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation,
                                padding=n_pad_pxl, groups=groups, padding_mode=padding_mode),
            'act_m': nn.Sigmoid(),
            'norm': get_norm_layer(normalization, out_channels)
        })
        self.median_filter = median_filter

    def forward(self, x: torch.Tensor):
        features = self.block.act_f(self.block.conv_f(x))
        mask = self.block.act_m(self.block.conv_m(x))
        output = features * mask
        output = self.block.norm(output)
        if self.median_filter:
            output = median_blur(output, kernel_size=(3, 3))
        return output


class GatedBlockSC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1,
                 act_fun=nn.ELU,
                 normalization: str = 'batch',
                 padding_mode: str = 'zeros',
                 median_filter: bool = False):
        super().__init__()

        n_pad_pxl = int(dilation * (kernel_size - 1) / 2)

        self.block = nn.ModuleDict({
            'conv_f': nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation,
                                padding=n_pad_pxl, groups=groups, padding_mode=padding_mode),
            'act_f': act_fun(),
            'conv_m': nn.Conv2d(in_channels, 1, kernel_size, stride=stride, dilation=dilation,
                                padding=n_pad_pxl, groups=groups, padding_mode=padding_mode),
            'act_m': nn.Sigmoid(),
            'norm': get_norm_layer(normalization, out_channels)
        })
        self.median_filter = median_filter

    def forward(self, x: torch.Tensor):
        features = self.block.act_f(self.block.conv_f(x))
        mask = self.block.act_m(self.block.conv_m(x))
        output = features * mask
        output = self.block.norm(output)
        if self.median_filter:
            output = median_blur(output, kernel_size=(3, 3))
        return output


class GatedBlockPW(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1,
                 act_fun=nn.ELU,
                 normalization: str = 'batch',
                 padding_mode: str = 'zeros',
                 median_filter: bool = False):
        super().__init__()

        n_pad_pxl = int(dilation * (kernel_size - 1) / 2)

        self.block = nn.ModuleDict({
            'conv_f': nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation,
                                padding=n_pad_pxl, groups=groups, padding_mode=padding_mode),
            'act_f': act_fun(),
            'conv_m': nn.Conv2d(in_channels, out_channels, 1, stride=stride, dilation=dilation, padding=0,
                                groups=groups),
            'act_m': nn.Sigmoid(),
            'norm': get_norm_layer(normalization, out_channels)
        })
        self.median_filter = median_filter

    def forward(self, x: torch.Tensor):
        features = self.block.act_f(self.block.conv_f(x))
        mask = self.block.act_m(self.block.conv_m(x))
        output = features * mask
        output = self.block.norm(output)
        if self.median_filter:
            output = median_blur(output, kernel_size=(3, 3))
        return output


class GatedBlockDS(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1,
                 act_fun=nn.ELU,
                 normalization: str = 'batch',
                 padding_mode: str = 'zeros',
                 median_filter: bool = False):
        super().__init__()

        n_pad_pxl = int(dilation * (kernel_size - 1) / 2)

        self.block = nn.ModuleDict({
            'conv_f': nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation,
                                padding=n_pad_pxl, groups=groups, padding_mode=padding_mode),
            'act_f': act_fun(),
            'conv_m': nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation, padding=n_pad_pxl,
                          groups=in_channels, padding_mode=padding_mode),
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, dilation=dilation, padding=0, groups=groups)
            ),
            'act_m': nn.Sigmoid(),
            'norm': get_norm_layer(normalization, out_channels)
        })
        self.median_filter = median_filter

    def forward(self, x: torch.Tensor):
        features = self.block.act_f(self.block.conv_f(x))
        mask = self.block.act_m(self.block.conv_m(x))
        output = features * mask
        output = self.block.norm(output)
        if self.median_filter:
            output = median_blur(output, kernel_size=(3, 3))
        return output


class GatedBlockSD(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1,
                 act_fun=nn.ELU,
                 normalization: str = 'batch',
                 padding_mode: str = 'zeros',
                 median_filter: bool = False):
        super().__init__()

        n_pad_pxl = int(dilation * (kernel_size - 1) / 2)

        self.block = nn.ModuleDict({
            'conv_f': nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, dilation=dilation, padding=0, groups=groups),
                nn.Conv2d(out_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=n_pad_pxl,
                          groups=out_channels, padding_mode=padding_mode),
            ),
            'act_f': act_fun(),
            'conv_m': nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, dilation=dilation, padding=0, groups=groups),
                nn.Conv2d(out_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=n_pad_pxl,
                          groups=out_channels, padding_mode=padding_mode),
            ),
            'act_m': nn.Sigmoid(),
            'norm': get_norm_layer(normalization, out_channels)
        })
        self.median_filter = median_filter

    def forward(self, x: torch.Tensor):
        features = self.block.act_f(self.block.conv_f(x))
        mask = self.block.act_m(self.block.conv_m(x))
        output = features * mask
        output = self.block.norm(output)
        if self.median_filter:
            output = median_blur(output, kernel_size=(3, 3))
        return output
