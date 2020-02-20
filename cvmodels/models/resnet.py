#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  resnet.py

"""

__author__ = 'Welkin'
__date__ = '2019/7/10 00:23'

import torch.nn as nn

from .resnet_modules import *

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
           'resnet20_cifar', 'resnet32_cifar', 'resnet44_cifar',
           'resnet56_cifar', 'resnet110_cifar']


class ResNet(nn.Module):
    """  """

    def __init__(self, block, layers, num_classes, in_size = 224):
        """ Constructor for ResNet """
        super().__init__()
        self._init_model(block, layers, num_classes, in_size)
        self._initialize_weights()

    def _init_model(self, block, layers, num_classes, in_size):
        out_size = in_size // 32
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True)
        )
        self.in_channels = 64
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.conv2 = self._make_layer(block, 64, layers[0])
        self.conv3 = self._make_layer(block, 128, layers[1], stride = 2)
        self.conv4 = self._make_layer(block, 256, layers[2], stride = 2)
        self.conv5 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(kernel_size = out_size)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.in_channels, num_classes)

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

    def _make_layer(self, block, out_channels_reduced, block_num, stride = 1):
        layer = []
        layer += [block(self.in_channels, out_channels_reduced, stride)]
        self.in_channels = out_channels_reduced * block.expansion
        layer += [block(self.in_channels, out_channels_reduced) for _ in range(block_num - 1)]
        return nn.Sequential(*layer)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out


class ResNet_Cifar(ResNet):
    """  """

    def __init__(self, block, layers, num_classes, in_size = 32):
        """ Constructor for ResNet_Cifar """
        super().__init__(block, layers, num_classes, in_size)

    def _init_model(self, block, layers, num_classes, in_size):
        out_size = in_size // 4
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace = True)
        )
        self.in_channels = 16
        self.conv2 = self._make_layer(block, 16, layers[0])
        self.conv3 = self._make_layer(block, 32, layers[1], stride = 2)
        self.conv4 = self._make_layer(block, 64, layers[2], stride = 2)
        self.avgpool = nn.AvgPool2d(kernel_size = out_size)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.in_channels, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.fc(out)
        return out


def resnet18(num_class, input_size):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_class, input_size)


def resnet34(num_class, input_size, **kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_class, input_size)


def resnet50(num_class, input_size, **kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_class, input_size)


def resnet101(num_class, input_size, **kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_class, input_size)


def resnet152(num_class, input_size, **kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_class, input_size)


def resnet20_cifar(num_class, input_size, **kwargs):
    return ResNet_Cifar(BasicBlock, [3, 3, 3], num_class, input_size)


def resnet32_cifar(num_class, input_size, **kwargs):
    return ResNet_Cifar(BasicBlock, [5, 5, 5], num_class, input_size)


def resnet44_cifar(num_class, input_size, **kwargs):
    return ResNet_Cifar(BasicBlock, [7, 7, 7], num_class, input_size)


def resnet56_cifar(num_class, input_size, **kwargs):
    return ResNet_Cifar(BasicBlock, [9, 9, 9], num_class, input_size)


def resnet110_cifar(num_class, input_size, **kwargs):
    return ResNet_Cifar(BasicBlock, [18, 18, 18], num_class, input_size)
