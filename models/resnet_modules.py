#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  resnet_modules.py

"""

__author__ = 'Welkin'
__date__ = '2019/7/10 15:35'

import torch.nn as nn
import torch.nn.functional as F

__all__ = ['BasicBlock', 'Bottleneck']


class BasicBlock(nn.Module):
    """  """
    expansion = 1

    def __init__(self, in_channels, out_channels_reduced, stride = 1, dilation = 1):
        """ Constructor for ResNet_Block """
        super().__init__()
        out_channels = out_channels_reduced * self.expansion
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_reduced, 3, stride, dilation, dilation),
            nn.BatchNorm2d(out_channels_reduced),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels_reduced, out_channels, 3, 1, dilation, dilation),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride),
            nn.BatchNorm2d(out_channels)
        ) if stride != 1 or in_channels != out_channels else None

    def forward(self, x):
        out = self.conv(x)
        if self.downsample:
            x = self.downsample(x)
        out += x
        return F.relu(out, inplace = True)


class Bottleneck(nn.Module):
    """  """
    expansion = 4

    def __init__(self, in_channels, out_channels_reduced, stride = 1, dilation = 1):
        """ Constructor for ResNet_Block """
        super().__init__()
        out_channels = out_channels_reduced * self.expansion
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_reduced, 1, stride),
            nn.BatchNorm2d(out_channels_reduced),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels_reduced, out_channels_reduced, 3, 1, dilation, dilation),
            nn.BatchNorm2d(out_channels_reduced),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels_reduced, out_channels, 1),
            nn.BatchNorm2d(out_channels),
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride),
            nn.BatchNorm2d(out_channels)
        ) if stride != 1 or in_channels != out_channels else None

    def forward(self, x):
        out = self.conv(x)
        if self.downsample:
            x = self.downsample(x)
        out += x
        return F.relu(out, inplace = True)
