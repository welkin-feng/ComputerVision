#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  hrnet.py

"""
import os
import logging

import torch
import torch.nn as nn
import torch._utils

from .hrnet_module import BasicBlock, Bottleneck, HighResolutionModule, BN_MOMENTUM

__all__ = ['hrnet']

logger = logging.getLogger(__name__)

hrnet_w48_cfg = {
    'EXTRA': {
        'STAGE1': {'NUM_MODULES': 1, 'NUM_BRANCHES': 1, 'BLOCK': 'BOTTLENECK', 'NUM_BLOCKS': [4, ],
                   'NUM_CHANNELS': [64, ], 'FUSE_METHOD': 'SUM'},
        'STAGE2': {'NUM_MODULES': 1, 'NUM_BRANCHES': 2, 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4],
                   'NUM_CHANNELS': [48, 96], 'FUSE_METHOD': 'SUM'},
        'STAGE3': {'NUM_MODULES': 4, 'NUM_BRANCHES': 3, 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4, 4],
                   'NUM_CHANNELS': [48, 96, 192], 'FUSE_METHOD': 'SUM'},
        'STAGE4': {'NUM_MODULES': 3, 'NUM_BRANCHES': 4, 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4, 4, 4],
                   'NUM_CHANNELS': [48, 96, 192, 384], 'FUSE_METHOD': 'SUM'},
    },
}

hrnet_w64_cfg = {
    'EXTRA': {
        'STAGE1': {'NUM_MODULES': 1, 'NUM_BRANCHES': 1, 'BLOCK': 'BOTTLENECK', 'NUM_BLOCKS': [4, ],
                   'NUM_CHANNELS': [64, ], 'FUSE_METHOD': 'SUM'},
        'STAGE2': {'NUM_MODULES': 1, 'NUM_BRANCHES': 2, 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4],
                   'NUM_CHANNELS': [64, 128], 'FUSE_METHOD': 'SUM'},
        'STAGE3': {'NUM_MODULES': 4, 'NUM_BRANCHES': 3, 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4, 4],
                   'NUM_CHANNELS': [64, 128, 256], 'FUSE_METHOD': 'SUM'},
        'STAGE4': {'NUM_MODULES': 3, 'NUM_BRANCHES': 4, 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4, 4, 4],
                   'NUM_CHANNELS': [64, 128, 256, 512], 'FUSE_METHOD': 'SUM'},
    },
}

blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

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

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i],
                                  3, 1, 1, bias = False),
                        nn.BatchNorm2d(num_channels_cur_layer[i], momentum = BN_MOMENTUM),
                        nn.ReLU(inplace = True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(inchannels, outchannels,
                                  3, 2, 1, bias = False),
                        nn.BatchNorm2d(outchannels, momentum = BN_MOMENTUM),
                        nn.ReLU(inplace = True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride = 1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(planes * block.expansion, momentum = BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output = True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(num_branches, block, num_blocks, num_inchannels, num_channels,
                                     fuse_method, reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

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

        return y_list

    def init_weights(self, pretrained = '', ):
        print('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            print('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            print('=> loading pretrained model weight length {}'.format(len(pretrained_dict)))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict, strict = False)


def hrnet(config, pretrained_path = '', **kwargs):
    model = HighResolutionNet(config, **kwargs)
    model.init_weights(pretrained_path)
    return model
