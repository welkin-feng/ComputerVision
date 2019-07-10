#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  util_modules.py

"""

__author__ = 'Welkin'
__date__ = '2019/6/24 12:23'

import torch.nn as nn

__all__ = ['Flatten', 'conv_bn_relu']


class Flatten(nn.Module):
    """  """

    def forward(self, x):
        return x.view(x.size(0), -1)


def conv_bn_relu(in_channels, out_channels, kernel_size, stride = 1, padding = 0,
                 batch_norm = False, dilation = 1, groups = 1, bias = True):
    block = [nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                     padding, dilation, groups, bias)]
    if batch_norm:
        block += [nn.BatchNorm2d(out_channels)]
    block += [nn.ReLU(inplace = True)]
    return nn.Sequential(*block)
