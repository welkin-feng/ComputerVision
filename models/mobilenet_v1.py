#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  mobilenet_v1.py

"""

__author__ = 'Welkin'
__date__ = '2019/7/4 00:22'

import torch.nn as nn

from .util_modules import *

__all__ = ['mobilenet_v1']


class DepthwiseSeparableConv(nn.Module):
    """  """

    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, alpha = 1):
        """ Constructor for DepthwiseSeparableConv """
        super().__init__()
        self.dw_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups = in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.dw_conv(x)


class MobileNet(nn.Module):
    """  """

    def __init__(self, num_classes, in_size = 224):
        """ Constructor for MobileNet """
        super().__init__()
        self._init_model(num_classes, in_size)
        self._initialize_weights()

    def _init_model(self, num_classes, in_size):
        final_size = in_size // 32
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride = 2, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            DepthwiseSeparableConv(32, 64, 3),
            DepthwiseSeparableConv(64, 128, 3, stride = 2),
            DepthwiseSeparableConv(128, 128, 3),
            DepthwiseSeparableConv(128, 256, 3, stride = 2),
            DepthwiseSeparableConv(256, 256, 3),
            DepthwiseSeparableConv(256, 512, 3, stride = 2),
            DepthwiseSeparableConv(512, 512, 3),
            DepthwiseSeparableConv(512, 512, 3),
            DepthwiseSeparableConv(512, 512, 3),
            DepthwiseSeparableConv(512, 512, 3),
            DepthwiseSeparableConv(512, 512, 3),
            DepthwiseSeparableConv(512, 1024, 3, stride = 2),
            DepthwiseSeparableConv(1024, 1024, 3),
        )
        self.avgpool = nn.AvgPool2d(kernel_size = final_size)
        self.flatten = Flatten()
        self.fc = nn.Linear(1024, num_classes)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, input):
        output = self.conv(input)
        output = self.avgpool(output)
        output = self.flatten(output)
        output = self.fc(output)
        return output


def mobilenet_v1(num_classes, in_size = 224):
    return MobileNet(num_classes, in_size)


if __name__ == '__main__':
    # test
    import sys

    fn_list = ['mobilenet_v1']
    for fn in fn_list:
        f = getattr(sys.modules[__name__], fn)
        model = f(10)
        print(' ---', fn, '---')
        for k, v in model.state_dict().items():
            print(k)
        print()
