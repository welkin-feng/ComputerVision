#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  inception_v1.py

"""

__author__ = 'Welkin'
__date__ = '2019/6/19 02:51'

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['inception_v1', 'inception_v1_cifar10']


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


class Inception_Module(nn.Module):
    """  """

    def __init__(self, in_channels, out_channels1_1, out_channels3_1, out_channels3_3,
                 out_channels5_1, out_channels5_5, out_channelsm_1):
        """ Constructor for Inception_Module """
        super().__init__()
        self.conv1 = Conv2d_relu(in_channels, out_channels1_1, kernel_size = 1, stride = 1)
        self.conv3 = nn.Sequential(
            Conv2d_relu(in_channels, out_channels3_1, kernel_size = 1, stride = 1),
            Conv2d_relu(out_channels3_1, out_channels3_3, kernel_size = 3, stride = 1, padding = 1),
        )
        self.conv5 = nn.Sequential(
            Conv2d_relu(in_channels, out_channels5_1, kernel_size = 1, stride = 1),
            Conv2d_relu(out_channels5_1, out_channels5_5, kernel_size = 5, stride = 1, padding = 2),
        )
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1),
            Conv2d_relu(in_channels, out_channelsm_1, kernel_size = 1, stride = 1),
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv3(x)
        out3 = self.conv5(x)
        out4 = self.maxpool(x)
        return torch.cat((out1, out2, out3, out4), dim = 1)


class Inception_v1(nn.Module):
    """  """

    def __init__(self, num_classes, in_size = 224):
        """ Constructor for Inception """
        super().__init__()
        self._init_model(num_classes, in_size)
        self._initialize_weights()

    def _init_model(self, num_classes, in_size):
        if in_size >= 65:
            mid_size = int((int((in_size + 15) / 16) - 2) / 3)
            final_size = int((in_size + 31) / 32)
            self.conv_1 = Conv2d_relu(3, 64, kernel_size = 7, stride = 2, padding = 3)
        else:
            mid_size = int(int((in_size + 7) / 8) - 2)
            final_size = int((in_size + 15) / 16)
            self.conv_1 = Conv2d_relu(3, 64, kernel_size = 7, stride = 1, padding = 3)
        self.maxpool_1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.conv_2_1 = Conv2d_relu(64, 64, kernel_size = 1)
        self.conv_2_2 = Conv2d_relu(64, 192, kernel_size = 3, stride = 1, padding = 1)
        self.maxpool_2 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.inception_3a = Inception_Module(192, 64, 96, 128, 16, 32, 32)
        self.inception_3b = Inception_Module(256, 128, 128, 192, 32, 96, 64)
        self.maxpool_3 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.inception_4a = Inception_Module(480, 192, 96, 208, 16, 48, 64)
        # auxiliary classifier 1
        if in_size >= 65:
            self.avgpool_4a = nn.AvgPool2d(kernel_size = 5, stride = 3)
        else:
            self.avgpool_4a = nn.AvgPool2d(kernel_size = 3, stride = 1)
        self.auxiliary_classifier_1 = nn.Sequential(
            Conv2d_relu(512, 128, kernel_size = 1, stride = 1),
            Flatten(),
            nn.Linear(128 * mid_size ** 2, 1024), nn.ReLU(inplace = True),
            nn.Dropout(0.7),
            nn.Linear(1024, num_classes)
        )
        self.inception_4b = Inception_Module(512, 160, 112, 224, 24, 64, 64)
        self.inception_4c = Inception_Module(512, 128, 128, 256, 24, 64, 64)
        self.inception_4d = Inception_Module(512, 112, 144, 288, 32, 64, 64)
        # auxiliary classifier 2
        if in_size >= 65:
            self.avgpool_4d = nn.AvgPool2d(kernel_size = 5, stride = 3)
        else:
            self.avgpool_4d = nn.AvgPool2d(kernel_size = 3, stride = 1)
        self.auxiliary_classifier_2 = nn.Sequential(
            Conv2d_relu(528, 128, kernel_size = 1, stride = 1),
            Flatten(),
            nn.Linear(128 * mid_size ** 2, 1024), nn.ReLU(inplace = True),
            nn.Dropout(0.7),
            nn.Linear(1024, num_classes)
        )
        self.inception_4e = Inception_Module(528, 256, 160, 320, 32, 128, 128)
        self.maxpool_4 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.inception_5a = Inception_Module(832, 256, 160, 320, 32, 128, 128)
        self.inception_5b = Inception_Module(832, 384, 192, 384, 48, 128, 128)
        # classifier
        self.classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size = final_size, stride = 1),
            Flatten(),
            nn.Dropout(0.4),
            nn.Linear(1024, num_classes)
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
        output = self.conv_1(input)
        output = self.maxpool_1(output)
        output = F.local_response_norm(output, size = 5)
        output = self.conv_2_1(output)
        output = self.conv_2_2(output)
        output = F.local_response_norm(output, size = 5)
        output = self.maxpool_2(output)
        output = self.inception_3a(output)
        output = self.inception_3b(output)
        output = self.maxpool_3(output)
        output = self.inception_4a(output)
        if self.training:
            output1 = self.avgpool_4a(output)
            output1 = self.auxiliary_classifier_1(output1)
        output = self.inception_4b(output)
        output = self.inception_4c(output)
        output = self.inception_4d(output)
        if self.training:
            output2 = self.avgpool_4d(output)
            output2 = self.auxiliary_classifier_2(output2)
        output = self.inception_4e(output)
        output = self.maxpool_4(output)
        output = self.inception_5a(output)
        output = self.inception_5b(output)
        output = self.classifier(output)

        if self.training:
            return output, output1, output2
        return output


def inception_v1(num_classes, in_size):
    return Inception_v1(num_classes, in_size)


def inception_v1_cifar10(num_classes, in_size):
    return Inception_v1(num_classes, in_size)
