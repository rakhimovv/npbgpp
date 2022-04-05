from typing import List

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Unet']


class DecoderBlock(nn.Module):
    def __init__(
            self,
            num_blocks,
            in_channels,
            skip_channels,
            out_channels,
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
            upsample='bilinear',
            kernel_size=3,
    ):
        super().__init__()
        assert num_blocks >= 1

        if upsample == 'bilinear':
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=2.0)
        elif upsample == 'nearest':
            self.upsample = nn.UpsamplingNearest2d(scale_factor=2.0)
        else:
            raise ValueError

        modules = [
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
            norm_layer(out_channels),
            act_layer(inplace=True)
        ]

        for i in range(1, num_blocks):
            modules.append(nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False))
            modules.append(norm_layer(out_channels))
            modules.append(act_layer(inplace=True))
        self.block = nn.Sequential(*modules)

        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if act_layer == nn.LeakyReLU:
                    nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                elif act_layer == nn.ReLU:
                    nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    pass
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.block(x)
        return x


class UnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            decoder_layers,
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
            decoder_kernel_size=3,
            upsample='bilinear',
            skip_rgb=False,
    ):
        super().__init__()

        assert len(decoder_channels) == len(decoder_layers)

        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])

        skip_channels = list(encoder_channels[1:]) + [3 if skip_rgb else 0]
        assert len(in_channels) == len(skip_channels) == len(
            decoder_channels), f"{in_channels}, {skip_channels}, {decoder_channels}"

        # combine decoder keyword arguments
        blocks = [
            DecoderBlock(num_blocks, in_ch, skip_ch, out_ch, act_layer=act_layer, norm_layer=norm_layer,
                         kernel_size=decoder_kernel_size, upsample=upsample) for num_blocks, in_ch, skip_ch, out_ch in
            zip(decoder_layers, in_channels, skip_channels, decoder_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, out):
        out = out[::-1]  # reverse channels to start from head of encoder
        x = out[0]
        skips = out[1:]
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
        return x


class Unet(nn.Module):

    def __init__(
            self,
            encoder_name: str = "spnasnet_100",
            out_channels: int = 8,
            decoder_channels: List[int] = (256, 128, 64, 32, 16),
            decoder_layers: List[int] = (1, 1, 1, 1, 1),
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
            upsample='bilinear',
            decoder_kernel_size=3,
            pretrained_encoder=True,
            skip_rgb=True,
            append_rgb=True,
            last_act='',
            eval_mode=False
    ):
        super().__init__()

        self.skip_rgb = skip_rgb
        self.append_rgb = append_rgb

        if pretrained_encoder:
            assert "dpn" not in encoder_name and "inception" not in encoder_name, "need to use different mean and std"
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
        else:
            mean = (0.0, 0.0, 0.0)
            std = (1.0, 1.0, 1.0)
        self.register_buffer("mean", torch.tensor(mean)[None, :, None, None])
        self.register_buffer("std", torch.tensor(std)[None, :, None, None])

        self.encoder = timm.create_model(encoder_name, features_only=True, pretrained=pretrained_encoder)
        feature_levels_num = len(self.encoder.feature_info.get_dicts())
        assert feature_levels_num == len(decoder_layers), f"{feature_levels_num} != {len(decoder_layers)}"
        self.first_stride = self.encoder.feature_info.reduction()[0]
        self.output_stride = 2 ** feature_levels_num

        for p in self.encoder.parameters():
            p.requires_grad = True

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.feature_info.channels(),
            decoder_channels=decoder_channels,
            decoder_layers=decoder_layers,
            act_layer=act_layer,
            norm_layer=norm_layer,
            decoder_kernel_size=decoder_kernel_size,
            upsample=upsample,
            skip_rgb=skip_rgb,
        )

        if append_rgb:
            self.final = nn.Conv2d(decoder_channels[-1], out_channels - 3, 1, bias=False)
        else:
            self.final = nn.Conv2d(decoder_channels[-1], out_channels, 1, bias=False)

        if last_act == 'sigmoid':
            self.final = nn.Sequential(self.final, nn.Sigmoid())
        elif last_act == 'tanh':
            self.final = nn.Sequential(self.final, nn.Tanh())
        else:
            assert last_act == ''

        self.eval_mode = eval_mode

    def forward(self, inp):
        if self.eval_mode:
            self.eval()

        # pad
        b, c, h, w = inp.size()
        pad_left, pad_right, pad_top, pad_bottom = 0, 0, 0, 0
        if h % self.output_stride != 0:
            pad_height = (h // self.output_stride + 1) * self.output_stride - h
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
        if w % self.output_stride != 0:
            pad_width = (w // self.output_stride + 1) * self.output_stride - w
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
        inp = F.pad(inp, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')
        x = (inp - self.mean) / self.std

        # run unet
        out = self.encoder(x)
        out = self.decoder([inp] + out) if self.skip_rgb else self.decoder(out)
        out = self.final(out)
        if self.append_rgb:
            out = torch.cat([out, inp], dim=1)

        # unpad
        out = out[:, :, pad_top:, pad_left:]
        if pad_bottom:
            out = out[:, :, :-pad_bottom, :]
        if pad_right:
            out = out[:, :, :, :-pad_right]

        return out
