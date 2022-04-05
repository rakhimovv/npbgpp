import os
from collections import OrderedDict
from os.path import join

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

__all__ = ['VGGLoss']

from torchvision.datasets.folder import pil_loader


class View(nn.Module):
    def __init__(self):
        super(View, self).__init__()

    def forward(self, x):
        return x.view(-1)


class VGGLoss(nn.Module):
    def __init__(self, net='caffe', optimized=False, save_dir=os.path.join('~/.cache/torch/models'),
                 style_img_path=None):
        super().__init__()

        save_dir = os.path.expanduser(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        if net == 'pytorch':
            vgg19 = torchvision.models.vgg19(pretrained=True).features

            self.register_buffer('mean_', torch.FloatTensor([0.485, 0.456, 0.406])[None, :, None, None])
            self.register_buffer('std_', torch.FloatTensor([0.229, 0.224, 0.225])[None, :, None, None])

        elif net == 'caffe':
            if not os.path.exists(join(save_dir, 'vgg_caffe_features.pth')):
                vgg_weights = torch.utils.model_zoo.load_url(
                    'https://web.eecs.umich.edu/~justincj/models/vgg19-d01eb7cb.pth', model_dir=save_dir)

                map = {'classifier.6.weight': u'classifier.7.weight', 'classifier.6.bias': u'classifier.7.bias'}
                vgg_weights = OrderedDict([(map[k] if k in map else k, v) for k, v in vgg_weights.items()])

                model = torchvision.models.vgg19()
                model.classifier = nn.Sequential(View(), *model.classifier._modules.values())

                model.load_state_dict(vgg_weights)

                vgg19 = model.features
                torch.save(vgg19, join(save_dir, 'vgg_caffe_features.pth'))

                self.register_buffer('mean_',
                                     torch.FloatTensor([103.939, 116.779, 123.680])[None, :, None, None] / 255.)
                self.register_buffer('std_', torch.FloatTensor([1. / 255, 1. / 255, 1. / 255])[None, :, None, None])

            else:
                vgg19 = torch.load(join(save_dir, 'vgg_caffe_features.pth'))
                self.register_buffer('mean_',
                                     torch.FloatTensor([103.939, 116.779, 123.680])[None, :, None, None] / 255.)
                self.register_buffer('std_', torch.FloatTensor([1. / 255, 1. / 255, 1. / 255])[None, :, None, None])
        else:
            assert False

        vgg19_avg_pooling = []

        for weights in vgg19.parameters():
            weights.requires_grad = False

        for module in vgg19.modules():
            if module.__class__.__name__ == 'Sequential':
                continue
            elif module.__class__.__name__ == 'MaxPool2d':
                vgg19_avg_pooling.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0))
            else:
                vgg19_avg_pooling.append(module)

        if optimized:
            self.layers = [3, 8, 17, 26, 35]
        else:
            self.layers = [1, 3, 6, 8, 11, 13, 15, 17, 20, 22, 24, 26, 29]

        self.vgg19 = nn.Sequential(*vgg19_avg_pooling)

        if style_img_path is not None:
            assert os.path.exists(style_img_path)
            style_img = pil_loader(style_img_path)
            style_img = torch.tensor(np.array(style_img)).permute(2, 0, 1).contiguous()
            style_img = style_img / 255
            features_target = self.normalize_inputs(style_img)
            for i, layer in enumerate(self.vgg19):
                features_target = layer(features_target)
                if i in self.layers:
                    self.register_buffer(f"gt_{i}", self.gram_matrix(features_target))

    def normalize_inputs(self, x):
        return (x - self.mean_) / self.std_

    def gram_matrix(self, input, mask=None):
        b, c, h, w = input.shape
        if mask is None:
            G = torch.einsum('bchw, bdhw -> bcd', input, input)
            return G.div(c * h * w)
        else:
            # one can try smth more sophisticated like in http://cs231n.stanford.edu/reports/2017/pdfs/417.pdf
            cur_mask = mask
            if input.shape[-2:] != mask.shape[-2:]:
                cur_mask = F.interpolate(mask, size=input.shape[-2:], mode='bilinear', align_corners=False)
            cur_mask = cur_mask > 0.5
            out = input.new_zeros(b, c, c)
            for i in range(b):
                t = torch.masked_select(input[i], cur_mask[i]).view(c, -1)
                n = t.shape[1]
                out[i] = t @ t.t()
                if n > 0:
                    out[i] /= (c * n)
            return out

    def forward(self, input, target, mask=None, remain_batch_dim=False, compute_style_loss=False):
        content_loss = 0
        style_loss = 0

        features_input = self.normalize_inputs(input)
        features_target = self.normalize_inputs(target)
        for i, layer in enumerate(self.vgg19):
            features_input = layer(features_input)
            features_target = layer(features_target)

            if i in self.layers:
                if compute_style_loss:
                    if i == 26:
                        if remain_batch_dim:
                            content_loss = content_loss + F.mse_loss(features_input, features_target,
                                                                     reduction='none').mean(
                                dim=(1, 2, 3))
                        else:
                            content_loss = content_loss + F.mse_loss(features_input, features_target)

                    G = self.gram_matrix(features_input, mask)
                    target_G = getattr(self, f"gt_{i}").expand(G.shape[0], -1, -1)
                    if remain_batch_dim:
                        style_loss = style_loss + F.mse_loss(G, target_G).mean(dim=(1, 2))
                    else:
                        style_loss = style_loss + F.mse_loss(G, target_G)
                else:
                    if remain_batch_dim:
                        content_loss = content_loss + F.l1_loss(features_input, features_target, reduction='none').mean(
                            dim=(1, 2, 3))
                    else:
                        content_loss = content_loss + F.l1_loss(features_input, features_target)

        if compute_style_loss:
            return content_loss, style_loss
        return content_loss
