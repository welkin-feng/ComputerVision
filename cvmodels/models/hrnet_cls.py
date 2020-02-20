#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  hrnet_cls.py

"""

__author__ = 'Welkin'
__date__ = '2020/2/3 18:17'

import logging

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F

from cvmodels.models import BasicBlock, Bottleneck
from cvmodels.models import HighResolutionNet

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HRNetClassification(HighResolutionNet):

    def __init__(self, cfg, **kwargs):
        super(HighResolutionNet, self).__init__()
        extra = cfg['EXTRA']
        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 2, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(64, momentum = BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size = 3, stride = 2, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(64, momentum = BN_MOMENTUM)
        self.relu = nn.ReLU(inplace = True)

        self.stage1_cfg = extra['STAGE1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(self.stage4_cfg, num_channels, multi_scale_output = True)

        # Classification Head
        self.incre_modules, self.downsamp_modules, self.final_layer = self._make_head(pre_stage_channels)

        self.classifier = nn.Linear(2048, 1000)

    def _make_head(self, pre_stage_channels):
        head_block = Bottleneck
        head_channels = [32, 64, 128, 256]

        # Increasing the #channels on each resolution
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            incre_module = self._make_layer(head_block, channels, head_channels[i], 1, stride = 1)
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)

        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i + 1] * head_block.expansion

            downsamp_module = nn.Sequential(
                nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
                          kernel_size = 3, stride = 2, padding = 1),
                nn.BatchNorm2d(out_channels, momentum = BN_MOMENTUM),
                nn.ReLU(inplace = True)
            )

            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.Conv2d(in_channels = head_channels[3] * head_block.expansion, out_channels = 2048,
                      kernel_size = 1, stride = 1, padding = 0),
            nn.BatchNorm2d(2048, momentum = BN_MOMENTUM),
            nn.ReLU(inplace = True)
        )

        return incre_modules, downsamp_modules, final_layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        # Classification Head
        y = self.incre_modules[0](y_list[0])
        for i in range(len(self.downsamp_modules)):
            y = self.incre_modules[i + 1](y_list[i + 1]) + self.downsamp_modules[i](y)

        y = self.final_layer(y)

        if torch._C._get_tracing_state():
            y = y.flatten(start_dim = 2).mean(dim = 2)
        else:
            y = F.avg_pool2d(y, kernel_size = y.size()[2:]).view(y.size(0), -1)

        y = self.classifier(y)

        return y


def get_cls_net(config, **kwargs):
    model = HighResolutionNet(config, **kwargs)
    model.init_weights()
    return model
