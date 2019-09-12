#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  vgg.py

"""

__author__ = 'Welkin'
__date__ = '2019/6/24 12:11'

import torch.nn as nn

from .util_modules import conv_bn_relu

__all__ = ['vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']


def conv3x3_block(in_channels, out_channels, depth, use_bn):
    block = [conv_bn_relu(in_channels, out_channels, 3, 1, 1, use_bn)]
    for _ in range(depth - 1):
        block += [conv_bn_relu(out_channels, out_channels, 3, 1, 1, use_bn)]
    block += [nn.MaxPool2d(kernel_size = 2)]
    return nn.Sequential(*block)


class VGG(nn.Module):
    """  """

    def __init__(self, num_classes, depth = 16, in_size = 224, use_batch_norm = False):
        """ Constructor for VGG """
        super().__init__()
        assert depth in (16, 19), 'depth should be 16 or 19'
        self._init_model(num_classes, depth, use_batch_norm, in_size)
        self._initialize_weights()

    def _init_model(self, num_classes, depth, use_bn, in_size):
        final_size = in_size // 32
        net = [conv3x3_block(3, 64, 2, use_bn),
               conv3x3_block(64, 128, 2, use_bn)]
        if depth == 16:
            net += [conv3x3_block(128, 256, 3, use_bn),
                    conv3x3_block(256, 512, 3, use_bn),
                    conv3x3_block(512, 512, 3, use_bn)]
        elif depth == 19:
            net += [conv3x3_block(128, 256, 4, use_bn),
                    conv3x3_block(256, 512, 4, use_bn),
                    conv3x3_block(512, 512, 4, use_bn)]
        self.conv = nn.Sequential(*net)
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(nn.Linear(512 * final_size ** 2, 4096),
                                nn.ReLU(inplace = True),
                                nn.Dropout(),
                                nn.Linear(4096, 4096),
                                nn.ReLU(inplace = True),
                                nn.Dropout(),
                                nn.Linear(4096, num_classes))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        output = self.conv(x)
        output = self.flatten(output)
        output = self.fc(output)
        return output


def vgg16(num_classes, in_size = 224):
    return VGG(num_classes, 16, in_size)


def vgg16_bn(num_classes, in_size = 224):
    return VGG(num_classes, 16, in_size, use_batch_norm = True)


def vgg19(num_classes, in_size = 224):
    return VGG(num_classes, 19, in_size)


def vgg19_bn(num_classes, in_size = 224):
    return VGG(num_classes, 19, in_size, use_batch_norm = True)


if __name__ == '__main__':
    # test
    import sys

    fn_list = ['vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']
    for fn in fn_list:
        f = getattr(sys.modules[__name__], fn)
        model = f(10)
        print(' ---', fn, '---')
        for k, v in model.state_dict().items():
            print(k)
        print()
