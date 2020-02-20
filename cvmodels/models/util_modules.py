#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  util_modules.py

"""

__author__ = 'Welkin'
__date__ = '2019/6/24 12:23'

import torch.nn as nn

__all__ = ['conv_bn_activation', 'conv_bn_relu']


def conv_bn_activation(in_channels, out_channels, kernel_size, stride = 1, padding = 0,
                       use_batch_norm = True, dilation = 1, groups = 1, bias = True, activation = None):
    block = [nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                       padding, dilation, groups, bias)]
    if use_batch_norm:
        block += [nn.BatchNorm2d(out_channels)]
    if activation is not None:
        block += [activation]
    return nn.Sequential(*block)


def conv_bn_relu(in_channels, out_channels, kernel_size, stride = 1, padding = 0,
                 use_batch_norm = True, dilation = 1, groups = 1, bias = True):
    return conv_bn_activation(in_channels, out_channels, kernel_size, stride, padding,
                              use_batch_norm, dilation, groups, bias, nn.ReLU(inplace = True))
