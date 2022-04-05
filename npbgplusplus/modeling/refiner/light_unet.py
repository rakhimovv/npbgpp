import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia import median_blur

from .gated import GatedBlock, GatedBlockSC, GatedBlockPW, GatedBlockDS, GatedBlockSD

__all__ = ['RefinerUnetV2']


def get_norm_layer(normalization: str, num_channels) -> nn.Module:
    if normalization == 'batch':
        return nn.BatchNorm2d(num_channels)
    elif normalization == 'instance':
        return nn.InstanceNorm2d(num_channels, affine=True)
    else:
        assert normalization == 'identity'
        return nn.Identity()


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1,
                 normalization: str = 'batch', padding_mode: str = 'zeros', median_filter: bool = False):
        super().__init__()

        n_pad_pxl = int(dilation * (kernel_size - 1) / 2)

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation,
                                             padding=n_pad_pxl, groups=groups, padding_mode=padding_mode),
                                   get_norm_layer(normalization, out_channels),
                                   nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size, stride=stride, dilation=dilation,
                                             padding=n_pad_pxl, groups=groups, padding_mode=padding_mode),
                                   get_norm_layer(normalization, out_channels),
                                   nn.ReLU())
        self.median_filter = median_filter

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        if self.median_filter:
            outputs = median_blur(outputs, kernel_size=(5, 5))
        return outputs


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_block=BasicBlock, padding_mode='zeros', median_filter=False,
                 normalization: str = 'batch'):
        super().__init__()
        self.conv = conv_block(in_channels, out_channels, padding_mode=padding_mode, median_filter=median_filter,
                               normalization=normalization)
        self.down = nn.AvgPool2d(2, 2)

    def forward(self, inputs):
        b, c, h, w = inputs.shape
        if h % 2 != 0 or w % 2 != 0:
            inputs = F.interpolate(inputs, size=(h // 2 * 2, w // 2 * 2), mode='bilinear', align_corners=False)
        outputs = self.down(inputs)
        outputs = self.conv(outputs)
        return outputs


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, upsample_mode, conv_block=BasicBlock,
                 padding_mode='zeros', use_skip=True):
        super().__init__()

        self.use_skip = use_skip
        self.upsample_mode = upsample_mode
        if upsample_mode == 'deconv':
            self.up = nn.ConvTranspose2d(in_channels, skip_channels, 4, stride=2, padding=1, padding_mode=padding_mode)
            conv_in_channels = skip_channels * 2 if use_skip else skip_channels
        elif upsample_mode == 'bilinear' or upsample_mode == 'nearest':
            conv_in_channels = in_channels + skip_channels if use_skip else in_channels
        else:
            assert ValueError

        self.conv = conv_block(conv_in_channels, out_channels, normalization='identity', padding_mode=padding_mode)

    def forward(self, x, skip):
        if self.upsample_mode == 'bilinear':
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        elif self.upsample_mode == 'nearest':
            x = F.interpolate(x, size=skip.shape[2:], mode='nearest')
        elif self.upsample_mode == 'deconv':
            x = self.up(x)

        if self.use_skip:
            output = self.conv(torch.cat([x, skip], 1))
        else:
            output = self.conv(x)

        return output


class RefinerUnetV2(nn.Module):
    r""" Rendering network with UNet architecture and multi-scale input.

    Args:
        in_channels: Number of channels in the input tensor or list of tensors. An integer or a list of integers for each input tensor.
        out_channels: Number of output channels.
        upsample_mode: One of 'deconv', 'bilinear' or 'nearest' for ConvTranspose, Bilinear or Nearest upsampling.c
        last_act: Last layer activation. One of 'sigmoid', 'tanh' or None.
        conv_block: Type of convolutional block, like Convolution-Normalization-Activation. One of 'basic', 'partial' or 'gated'.
    """

    def __init__(
            self,
            in_channels=8,
            out_channels=4,
            filters=(16, 32, 64, 128, 210),
            upsample_mode='bilinear',
            last_act='',
            stem_conv_block_name='gated',
            encoder_conv_block_name='gated',
            decoder_conv_block_name='gated',
            padding_mode='replicate',
            median_filter=False,
            encoder_normalization: str = 'identity',
            use_skip=True
    ):
        super().__init__()
        assert len(filters) == 5

        if isinstance(in_channels, int):
            in_channels = [in_channels]

        if len(in_channels) < 5:
            in_channels += [0] * (5 - len(in_channels))

        self.num_input_channels = in_channels[:5]
        self.out_channels = out_channels

        def get_conv_block(name):
            if name == 'basic':
                return BasicBlock
            elif name == 'gated':
                return GatedBlock
            elif name == 'gated-sc':
                return GatedBlockSC
            elif name == 'gated-pw':
                return GatedBlockPW
            elif name == 'gated-ds':
                return GatedBlockDS
            elif name == 'gated-sd':
                return GatedBlockSD
            else:
                raise ValueError(f'unkwown conv block {name}')

        stem_conv_block = get_conv_block(stem_conv_block_name)
        encoder_conv_block = get_conv_block(encoder_conv_block_name)
        decoder_conv_block = get_conv_block(decoder_conv_block_name)

        self.start = stem_conv_block(self.num_input_channels[0], filters[0], padding_mode=padding_mode,
                                     median_filter=median_filter, normalization=encoder_normalization)

        self.down1 = DownsampleBlock(filters[0], filters[1] - self.num_input_channels[1], conv_block=encoder_conv_block,
                                     padding_mode=padding_mode, median_filter=median_filter,
                                     normalization=encoder_normalization)
        self.down2 = DownsampleBlock(filters[1], filters[2] - self.num_input_channels[2], conv_block=encoder_conv_block,
                                     padding_mode=padding_mode, median_filter=median_filter,
                                     normalization=encoder_normalization)
        self.down3 = DownsampleBlock(filters[2], filters[3] - self.num_input_channels[3], conv_block=encoder_conv_block,
                                     padding_mode=padding_mode, median_filter=median_filter,
                                     normalization=encoder_normalization)
        self.down4 = DownsampleBlock(filters[3], filters[4] - self.num_input_channels[4], conv_block=encoder_conv_block,
                                     padding_mode=padding_mode, median_filter=median_filter,
                                     normalization=encoder_normalization)

        self.center = encoder_conv_block(filters[4], filters[4], padding_mode=padding_mode, median_filter=median_filter,
                                         normalization=encoder_normalization)

        self.up4 = UpsampleBlock(filters[4], filters[3], out_channels * 16, upsample_mode,
                                 conv_block=decoder_conv_block, padding_mode=padding_mode, use_skip=use_skip)
        self.up3 = UpsampleBlock(out_channels * 16, filters[2], out_channels * 8, upsample_mode,
                                 conv_block=decoder_conv_block, padding_mode=padding_mode, use_skip=use_skip)
        self.up2 = UpsampleBlock(out_channels * 8, filters[1], out_channels * 4, upsample_mode,
                                 conv_block=decoder_conv_block, padding_mode=padding_mode, use_skip=use_skip)
        self.up1 = UpsampleBlock(out_channels * 4, filters[0], out_channels * 2, upsample_mode,
                                 conv_block=decoder_conv_block, padding_mode=padding_mode, use_skip=use_skip)

        self.final = nn.Conv2d(out_channels * 2, out_channels, 1)

        if last_act == 'sigmoid':
            self.final = nn.Sequential(self.final, nn.Sigmoid())
        elif last_act == 'tanh':
            self.final = nn.Sequential(self.final, nn.Tanh())
        else:
            assert last_act == ''
            self.final.bias.data.fill_(0.5)
            if out_channels == 4:
                self.final.bias.data[3].fill_(0.0)

    def forward(self, inputs):

        if not isinstance(inputs, list):
            inputs = [inputs]

        n_input = len(inputs)
        n_declared = np.count_nonzero(self.num_input_channels)
        assert n_input == n_declared, f'got {n_input} input scales but declared {n_declared}'

        in64 = self.start(inputs[0])

        down1 = self.down1(in64)

        if self.num_input_channels[1]:
            down1 = torch.cat([down1, inputs[1]], 1)

        down2 = self.down2(down1)

        if self.num_input_channels[2]:
            down2 = torch.cat([down2, inputs[2]], 1)

        down3 = self.down3(down2)

        if self.num_input_channels[3]:
            down3 = torch.cat([down3, inputs[3]], 1)

        down4 = self.down4(down3)

        if self.num_input_channels[4]:
            down4 = torch.cat([down4, inputs[4]], 1)

        down4 = self.center(down4)

        up4 = self.up4(down4, down3)
        up3 = self.up3(up4, down2)
        up2 = self.up2(up3, down1)
        up1 = self.up1(up2, in64)

        result = self.final(up1)

        return result
