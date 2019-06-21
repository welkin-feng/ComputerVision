#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  AlexNet.py

"""

__author__ = 'Welkin'
__date__ = '2019/6/18 03:06'

import torch.nn as nn

__all__ = ['alexnet', 'alexnet_cifar10']


class AlexNet(nn.Module):
    """ input size should be 227x227x3  """

    def __init__(self, num_classes, in_size = 227):
        """ Constructor for AlexNet """
        super().__init__()

        mid_size = int((in_size - 35) / 32)
        if mid_size <= 0:
            raise ValueError("`in_size` is too small")

        self.conv = nn.Sequential(
            # (227-11)/4+1=55
            nn.Conv2d(3, 96, kernel_size = 11, stride = 4),
            nn.ReLU(inplace = True),
            nn.LocalResponseNorm(size = 5, k = 2),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            # (55-3)/2+1=27
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
            # (13-3)/2+1=6
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * mid_size ** 2, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


class AlexNet_cifar10(nn.Module):
    """ AlexNet for cifar10  """

    def __init__(self, num_classes = 10, in_size = 32):
        """ Constructor for AlexNet """
        super().__init__()

        mid_size = in_size / 8

        self.conv = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size = 11, stride = 4, padding = 5),
            nn.ReLU(inplace = True),
            nn.LocalResponseNorm(size = 5, k = 2),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(96, 256, kernel_size = 5, stride = 1, padding = 2),
            nn.ReLU(inplace = True),
            nn.LocalResponseNorm(size = 5, k = 2),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.Conv2d(256, 384, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(384, 384, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(384, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * mid_size ** 2, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

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
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def alexnet(num_classes):
    return AlexNet(num_classes = num_classes)


def alexnet_cifar10(num_classes = 10):
    return AlexNet_cifar10(num_classes = num_classes)
