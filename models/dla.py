#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

import torch
from torch import nn
from torch.utils import model_zoo

from .res2net_module import Bottle2neck, Bottle2neckX, SEBottle2neck, SEBottle2neckX

BatchNorm = nn.BatchNorm2d

__all__ = ['res2net_dla60', 'res2next_dla60', 'se_res2net_dla60', 'se_res2next_dla60']

model_urls = {
    'res2net_dla60': 'http://data.kaizhao.net/projects/res2net/pretrained/res2net_dla60_4s-d88db7f9.pth',
    'res2next_dla60': 'http://data.kaizhao.net/projects/res2net/pretrained/res2next_dla60_4s-d327927b.pth',
}


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride = 1, bias = False, padding = (kernel_size - 1) // 2)
        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace = True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride = 1,
                 level_root = False, root_dim = 0, root_kernel_size = 1,
                 dilation = 1, root_residual = False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation = dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation = dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim = 0,
                              root_kernel_size = root_kernel_size,
                              dilation = dilation, root_residual = root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim = root_dim + out_channels,
                              root_kernel_size = root_kernel_size,
                              dilation = dilation, root_residual = root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride = stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size = 1, stride = 1, bias = False),
                BatchNorm(out_channels)
            )

    def forward(self, x, residual = None, children = None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children = children)
        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes = 1000,
                 block = Bottle2neck, residual_root = False, return_levels = False,
                 pool_size = 7, linear_root = False):
        super(DLA, self).__init__()
        self.channels = channels
        self.return_levels = return_levels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size = 7, stride = 1,
                      padding = 3, bias = False),
            BatchNorm(channels[0]),
            nn.ReLU(inplace = True))
        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride = 2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root = False,
                           root_residual = residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root = True, root_residual = residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root = True, root_residual = residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root = True, root_residual = residual_root)

        self.avgpool = nn.AvgPool2d(pool_size)
        self.fc = nn.Conv2d(channels[-1], num_classes, kernel_size = 1,
                            stride = 1, padding = 0, bias = True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride = 1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride = stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size = 1, stride = 1, bias = False),
                BatchNorm(planes),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample = downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride = 1, dilation = 1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size = 3,
                          stride = stride if i == 0 else 1,
                          padding = dilation, bias = False, dilation = dilation),
                BatchNorm(planes),
                nn.ReLU(inplace = True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        if self.return_levels:
            return y
        else:
            x = self.avgpool(x)
            x = self.fc(x)
            x = x.view(x.size(0), -1)

            return x

    def load_pretrained_model(self, model_name):
        try:
            model_url = model_urls[model_name]
        except KeyError:
            raise ValueError(
                'trained {} does not exist.'.format(model_name))
        self.load_state_dict(model_zoo.load_url(model_url), strict = False)


def res2net_dla60(pretrained = None, **kwargs):
    Bottle2neck.expansion = 2
    model = DLA([1, 1, 1, 2, 3, 1],
                [16, 32, 128, 256, 512, 1024],
                block = Bottle2neck, **kwargs)
    if pretrained:
        model.load_pretrained_model('res2net_dla60')
    return model


def res2next_dla60(pretrained = None, **kwargs):
    Bottle2neckX.expansion = 2
    model = DLA([1, 1, 1, 2, 3, 1],
                [16, 32, 128, 256, 512, 1024],
                block = Bottle2neckX, **kwargs)
    if pretrained:
        model.load_pretrained_model('res2next_dla60')
    return model


def se_res2net_dla60(pretrained = None, **kwargs):
    SEBottle2neck.expansion = 2
    model = DLA([1, 1, 1, 2, 3, 1],
                [16, 32, 128, 256, 512, 1024],
                block = SEBottle2neck, **kwargs)
    if pretrained:
        model.load_pretrained_model('res2net_dla60')
    return model


def se_res2next_dla60(pretrained = None, **kwargs):
    SEBottle2neckX.expansion = 2
    model = DLA([1, 1, 1, 2, 3, 1],
                [16, 32, 128, 256, 512, 1024],
                block = SEBottle2neckX, **kwargs)
    if pretrained:
        model.load_pretrained_model('res2next_dla60')
    return model


if __name__ == '__main__':
    images = torch.rand(1, 3, 224, 224).cuda(0)
    model = res2next_dla60(pretrained = True)
    model = model.cuda(0)
    print(model(images).size())
