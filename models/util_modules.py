#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  util_modules.py

"""

__author__ = 'Welkin'
__date__ = '2019/6/24 12:23'

import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Flatten', 'Conv_bn_relu']


class Flatten(nn.Module):
    """  """

    def forward(self, x):
        return x.view(x.size(0), -1)


class Conv_bn_relu(nn.Module):
    """  """

    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0,
                 batch_norm = False, dilation = 1, groups = 1, bias = True):
        """ Constructor for Conv_ReLU """
        super().__init__()
        self.batch_norm = batch_norm
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups, bias)
        if self.batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        return F.relu(x, inplace = True)
