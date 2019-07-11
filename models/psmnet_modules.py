#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  psmnet_modules.py

"""

__author__ = 'Welkin'
__date__ = '2019/7/9 21:59'

import torch
import torch.nn as nn
import torch.nn.functional as F

from .util_modules import conv_bn_relu




class BasicResidualBlock(nn.Module):
    """  """
    expansion = 1

    def __init__(self, in_channels, out_channels_reduced, stride = 1, padding = 1, dilation = 1):
        """ Constructor for ResNet_Block """
        super().__init__()
        out_channels = out_channels_reduced * self.expansion
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_reduced, 3, stride, padding, dilation),
            nn.BatchNorm2d(out_channels_reduced),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels_reduced, out_channels, 3, 1, padding, dilation),
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
        return out


def conv3x3x3_bn_relu(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1,
                      use_batch_norm = True, dilation = 1, groups = 1, bias = True):
    block = [nn.Conv3d(in_channels, out_channels, kernel_size, stride,
                       padding, dilation, groups, bias)]
    if use_batch_norm:
        block += [nn.BatchNorm3d(out_channels)]
    block += [nn.ReLU(inplace = True)]
    return nn.Sequential(*block)


class CNN_Module(nn.Module):
    """  """

    def __init__(self):
        """ Constructor for CNN_Module """
        super().__init__()
        self.conv0 = nn.Sequential(
            conv_bn_relu(3, 32, 3, stride = 2, padding = 1),
            conv_bn_relu(32, 32, 3, padding = 1),
            conv_bn_relu(32, 32, 3, padding = 1)
        )
        layers = [3, 16, 3, 3]
        self.conv1 = self._make_layer(32, 32, layers[0])
        self.conv2 = self._make_layer(32, 64, layers[1], stride = 2)
        self.conv3 = self._make_layer(64, 128, layers[2], padding = 2, dilation = 2)
        self.conv4 = self._make_layer(128, 128, layers[3], padding = 4, dilation = 4)

    def _make_layer(self, in_channels, out_channels, block_num, stride = 1, padding = 1, dilation = 1):
        layer = []
        layer += [BasicResidualBlock(in_channels, out_channels, stride, padding, dilation)]
        layer += [BasicResidualBlock(out_channels, out_channels, 1, padding, dilation) for _ in range(block_num - 1)]
        return nn.Sequential(*layer)

    def forward(self, x):
        out = self.conv0(x)
        out = self.conv1(out)
        out_conv2 = self.conv2(out)
        out = self.conv3(out_conv2)
        out_conv4 = self.conv4(out)
        return out_conv2, out_conv4


class SPP_Module(nn.Module):
    """  """

    def __init__(self, scales = (64, 32, 16, 8)):
        """ Constructor for SPP_Module """
        super().__init__()
        self.scales = scales
        self.branch_1 = nn.Sequential(
            nn.AvgPool2d(self.scales[0]),
            conv_bn_relu(128, 32, 3, padding = 1)
        )
        self.branch_2 = nn.Sequential(
            nn.AvgPool2d(self.scales[1]),
            conv_bn_relu(128, 32, 3, padding = 1)
        )
        self.branch_3 = nn.Sequential(
            nn.AvgPool2d(self.scales[2]),
            conv_bn_relu(128, 32, 3, padding = 1)
        )
        self.branch_4 = nn.Sequential(
            nn.AvgPool2d(self.scales[3]),
            conv_bn_relu(128, 32, 3, padding = 1),
        )
        self.fusion = nn.Sequential(
            conv_bn_relu(320, 128, 3, padding = 1),
            nn.Conv2d(128, 32, kernel_size = 1)
        )

    def forward(self, out_conv2, out_conv4):
        x = out_conv4
        out = [out_conv2, out_conv4]
        out += [F.interpolate(self.branch_1(x), scale_factor = self.scales[0], mode = 'bilinear'),
                F.interpolate(self.branch_2(x), scale_factor = self.scales[1], mode = 'bilinear'),
                F.interpolate(self.branch_3(x), scale_factor = self.scales[2], mode = 'bilinear'),
                F.interpolate(self.branch_4(x), scale_factor = self.scales[3], mode = 'bilinear')]
        out = torch.cat(tuple(out), dim = 1)
        out = self.fusion(out)
        return out


class CostVolume(nn.Module):
    """  """

    def __init__(self, max_disparity):
        """ Constructor for CostVolume """
        super().__init__()
        self.D = max_disparity

    def forward(self, left_feature, right_feature):
        N, C, H, W = left_feature.size()
        cost_volume = torch.zeros(N, C * 2, self.D // 4, H, W).type_as(left_feature)  # [N, 64, 1/4D, 1/4H, 1/4W]

        for i in range(self.D // 4):
            if i > 0:
                cost_volume[:, :C, i, :, i:] = left_feature[:, :, :, i:]
                cost_volume[:, C:, i, :, i:] = right_feature[:, :, :, :-i]
            else:
                cost_volume[:, :C, i, :, :] = left_feature
                cost_volume[:, C:, i, :, :] = right_feature
        return cost_volume


class Basic3dCNN(nn.Module):
    """  """

    def __init__(self, *args, **kw):
        """ Constructor for Basic3dCNN """
        super().__init__()

    def forward(self, *input):
        return


def disparity_regression(x: torch.Tensor):
    out = F.softmax(x, dim = 2)
    out = (x * out).sum(dim = 2)
    return out


class StackedHourglass3dCNN(nn.Module):
    """  """

    def __init__(self):
        """ Constructor for StackedHourglass3dCNN """
        super().__init__()
        self.conv0 = nn.Sequential(
            conv3x3x3_bn_relu(64, 32),
            conv3x3x3_bn_relu(32, 32)
        )
        self.conv1 = nn.Sequential(
            conv3x3x3_bn_relu(32, 32),
            conv3x3x3_bn_relu(32, 32)
        )

        self.stack1_1 = nn.Sequential(
            conv3x3x3_bn_relu(32, 64, stride = 2),
            conv3x3x3_bn_relu(64, 64))
        self.stack1_2 = nn.Sequential(
            conv3x3x3_bn_relu(64, 64, stride = 2),
            conv3x3x3_bn_relu(64, 64))
        self.stack1_3 = nn.ConvTranspose3d(64, 64, 3, stride = 2, padding = 1, output_padding = 1)
        self.stack1_4 = nn.ConvTranspose3d(64, 32, 3, stride = 2, padding = 1, output_padding = 1)

        self.stack2_1 = nn.Sequential(
            conv3x3x3_bn_relu(32, 64, stride = 2),
            conv3x3x3_bn_relu(64, 64))
        self.stack2_2 = nn.Sequential(
            conv3x3x3_bn_relu(64, 64, stride = 2),
            conv3x3x3_bn_relu(64, 64))
        self.stack2_3 = nn.ConvTranspose3d(64, 64, 3, stride = 2, padding = 1, output_padding = 1)
        self.stack2_4 = nn.ConvTranspose3d(64, 32, 3, stride = 2, padding = 1, output_padding = 1)

        self.stack3_1 = nn.Sequential(
            conv3x3x3_bn_relu(32, 64, stride = 2),
            conv3x3x3_bn_relu(64, 64))
        self.stack3_2 = nn.Sequential(
            conv3x3x3_bn_relu(64, 64, stride = 2),
            conv3x3x3_bn_relu(64, 64))
        self.stack3_3 = nn.ConvTranspose3d(64, 64, 3, stride = 2, padding = 1, output_padding = 1)
        self.stack3_4 = nn.ConvTranspose3d(64, 32, 3, stride = 2, padding = 1, output_padding = 1)

        self.output_1 = nn.Sequential(
            conv3x3x3_bn_relu(32, 32),
            conv3x3x3_bn_relu(32, 1)
        )
        self.output_2 = nn.Sequential(
            conv3x3x3_bn_relu(32, 32),
            conv3x3x3_bn_relu(32, 1)
        )
        self.output_3 = nn.Sequential(
            conv3x3x3_bn_relu(32, 32),
            conv3x3x3_bn_relu(32, 1)
        )

    def forward(self, x):
        out = self.conv0(x)
        out_conv1 = self.conv1(out) + out

        out_stack1_1 = self.stack1_1(out_conv1)
        out = self.stack1_2(out_stack1_1)
        out_stack1_3 = self.stack1_3(out) + out_stack1_1
        out = self.stack1_4(out_stack1_3) + out_conv1

        out = self.stack2_1(out) + out_stack1_3
        out = self.stack2_2(out)
        out_stack2_3 = self.stack2_3(out) + out_stack1_1
        out = self.stack2_4(out_stack2_3) + out_conv1

        out = self.stack3_1(out) + out_stack2_3
        out = self.stack3_2(out)
        out = self.stack3_3(out) + out_stack1_1
        out = self.stack3_4(out) + out_conv1

        out1 = self.output_1(out)
        out2 = self.output_2(out) + out1
        out3 = self.output_3(out) + out2

        if self.training:
            out1, out2, out3 = map(lambda x: F.interpolate(x, scale_factor = 4, mode = 'trilinear'), (out1, out2, out3))
            out1, out2, out3 = map(disparity_regression, (out1, out2, out3))
            return out1, out2, out3

        out3 = F.interpolate(out3, scale_factor = 4, mode = 'trilinear')
        out3 = disparity_regression(out3)
        return out3
