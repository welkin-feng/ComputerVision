#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  vgg.py

"""

__author__ = 'Welkin'
__date__ = '2019/6/24 12:11'

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.util_modules import Flatten, Conv_bn_relu

__all__ = ['vgg_16', 'vgg_19']


class Conv3x3_block(nn.Module):
    """  """

    def __init__(self, in_channels, out_channels, depth):
        """ Constructor for Conv_ReLU """
        super().__init__()
        conv_list = [Conv_bn_relu(in_channels, out_channels, 3, 1, 1)]
        for _ in range(depth - 1):
            conv_list.append(Conv_bn_relu(out_channels, out_channels, 3, 1, 1))
        self.conv = nn.Sequential(*conv_list)
        self.maxpool = nn.MaxPool2d(kernel_size = 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        return x


class VGG(nn.Module):
    """  """

    def __init__(self, num_classes, depth = 16, in_size = 224):
        """ Constructor for VGG """
        super().__init__()
        assert depth in (16, 19), 'depth should be 16 or 19'
        self._init_model(num_classes, depth, in_size)
        self._initialize_weights()

    def _init_model(self, num_classes, depth, in_size):
        final_size = int(in_size / 32)

        self.block_1 = Conv3x3_block(3, 64, 2)
        self.block_2 = Conv3x3_block(64, 128, 2)
        if depth == 16:
            self.block_3 = Conv3x3_block(128, 256, 3)
            self.block_4 = Conv3x3_block(256, 512, 3)
            self.block_5 = Conv3x3_block(512, 512, 3)
        elif depth == 19:
            self.block_3 = Conv3x3_block(128, 256, 4)
            self.block_4 = Conv3x3_block(256, 512, 4)
            self.block_5 = Conv3x3_block(512, 512, 4)
        self.flatten = Flatten()
        self.fc = nn.Sequential(
            nn.Linear(512 * final_size ** 2, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

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
        output = self.block_1(input)
        output = self.block_2(output)
        output = self.block_3(output)
        output = self.block_4(output)
        output = self.block_5(output)
        output = self.flatten(output)
        output = self.fc(output)
        return output


def vgg_16(num_classes, in_size = 224):
    return VGG(num_classes, 16, in_size)


def vgg_19(num_classes, in_size = 224):
    return VGG(num_classes, 19, in_size)
