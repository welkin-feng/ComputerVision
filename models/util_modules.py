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


class Flatten(nn.Module):
    """  """

    def forward(self, x):
        return x.view(x.size(0), -1)


class Conv2d_relu(nn.Module):
    """  """

    def __init__(self, in_channels, out_channels, kernel_size, stride = 1,
                 padding = 0, dilation = 1, groups = 1, bias = True):
        """ Constructor for Conv_ReLU """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups, bias)

    def forward(self, x):
        return F.relu(self.conv(x), inplace = True)
