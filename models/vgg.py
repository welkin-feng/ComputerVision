#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  vgg.py

"""

__author__ = 'Welkin'
__date__ = '2019/6/24 12:11'

import torch.nn as nn

from .util_modules import Flatten, Conv_bn_relu

__all__ = ['vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']


class Conv3x3_block(nn.Module):
    """  """

    def __init__(self, in_channels, out_channels, depth, batch_norm):
        """ Constructor for Conv_ReLU """
        super().__init__()
        conv_list = [Conv_bn_relu(in_channels, out_channels, 3, 1, 1, batch_norm)]
        for _ in range(depth - 1):
            conv_list += [Conv_bn_relu(out_channels, out_channels, 3, 1, 1, batch_norm)]
        self.conv = nn.Sequential(*conv_list)
        self.maxpool = nn.MaxPool2d(kernel_size = 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        return x


class VGG(nn.Module):
    """  """

    def __init__(self, num_classes, depth = 16, in_size = 224, batch_norm = False):
        """ Constructor for VGG """
        super().__init__()
        assert depth in (16, 19), 'depth should be 16 or 19'
        self._init_model(num_classes, depth, batch_norm, in_size)
        self._initialize_weights()

    def _init_model(self, num_classes, depth, batch_norm, in_size):
        final_size = in_size // 32
        net = [Conv3x3_block(3, 64, 2, batch_norm),
               Conv3x3_block(64, 128, 2, batch_norm)]
        if depth == 16:
            net += [Conv3x3_block(128, 256, 3, batch_norm),
                    Conv3x3_block(256, 512, 3, batch_norm),
                    Conv3x3_block(512, 512, 3, batch_norm)]
        elif depth == 19:
            net += [Conv3x3_block(128, 256, 4, batch_norm),
                    Conv3x3_block(256, 512, 4, batch_norm),
                    Conv3x3_block(512, 512, 4, batch_norm)]
        self.conv = nn.Sequential(*net)
        self.flatten = Flatten()
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
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input):
        output = self.conv(input)
        output = self.flatten(output)
        output = self.fc(output)
        return output


def vgg16(num_classes, in_size = 224):
    return VGG(num_classes, 16, in_size)


def vgg16_bn(num_classes, in_size = 224):
    return VGG(num_classes, 16, in_size, batch_norm = True)


def vgg19(num_classes, in_size = 224):
    return VGG(num_classes, 19, in_size)


def vgg19_bn(num_classes, in_size = 224):
    return VGG(num_classes, 19, in_size, batch_norm = True)


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
