#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  test.py

"""

__author__ = 'Welkin'
__date__ = '2019/6/20 02:01'

import torch
import torch.nn as nn

from easydict import EasyDict


class ThreeLayerConvNet(nn.Module):
    def __init__(self, in_channel, channel_1, ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, channel_1, kernel_size = 7,
                               stride = 2, padding = 3)
        self.conv2 = nn.Sequential(
            # (227-11)/4+1=55
            nn.Conv2d(3, 96, kernel_size = 11, stride = 4),
            nn.ReLU(inplace = True),
            nn.LocalResponseNorm(size = 5, k = 2),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            # # (55-3)/2+1=27
            nn.Conv2d(96, 256, kernel_size = 5, stride = 1, padding = 2),
            nn.ReLU(inplace = True),
            nn.LocalResponseNorm(size = 5, k = 2),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            # (27-3)/2+1=13
            nn.Conv2d(256, 384, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(384, 384, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(384, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            # # (13-3)/2+1=6
        )

    def forward(self, x):
        scores = self.conv2(x)
        return scores


def test_ConvNet(model, x_shape):
    x = torch.zeros(x_shape)
    scores = model(x)
    print(scores.size())  # you should see [64, 10]


if __name__ == '__main__':
    x_shape = (4, 3, 227, 227)
    # model = ThreeLayerConvNet(in_channel = 3, channel_1 = 12)
    # test_ConvNet(model, x_shape)
    from models import *
    config = EasyDict({'architecture': 'inception_v1', 'num_classes': 10})
    model = get_model(config)
    # test_ConvNet(model, x_shape)
    x = torch.ones(x_shape)
    scores = model(x)
    print(scores.size())
    print(scores.max(1))

