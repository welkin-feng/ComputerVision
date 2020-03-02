#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  res2net_module.py

"""
import math

import torch
from torch import nn
from cvmodels.models.layers.se_module import SELayer

BatchNorm = nn.BatchNorm2d

__author__ = 'Welkin'
__date__ = '2020/1/15 01:09'

__all__ = ['Bottle2neck', 'Bottle2neckX', 'SEBottle2neck', 'SEBottle2neckX']


class Bottle2neck(nn.Module):
    """
    RexNeXt bottleneck type C
    """
    expansion = 2

    def __init__(self, inplanes, planes, stride = 1, dilation = 1, baseWidth = 28, scale = 4):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': frist blokc of a new stage.
        """
        super(Bottle2neck, self).__init__()
        if stride != 1:
            stype = 'stage'
        else:
            stype = 'normal'
        width = int(math.floor(planes * (baseWidth / 128.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size = 1, bias = False)
        self.bn1 = BatchNorm(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size = 3, stride = stride, padding = 1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size = 3, stride = stride,
                                   padding = dilation, dilation = dilation, bias = False))
            bns.append(BatchNorm(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes, kernel_size = 1, bias = False)
        self.bn3 = BatchNorm(planes)

        self.relu = nn.ReLU(inplace = True)
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x, residual = None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Bottle2neckX(nn.Module):
    """
    RexNeXt bottleneck type C
    """
    expansion = 2
    cardinality = 8

    def __init__(self, inplanes, planes, stride = 1, dilation = 1, scale = 4):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': frist blokc of a new stage.
        """
        super(Bottle2neckX, self).__init__()
        if stride != 1:
            stype = 'stage'
        else:
            stype = 'normal'
        cardinality = Bottle2neckX.cardinality
        width = bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size = 1, bias = False)
        self.bn1 = BatchNorm(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size = 3, stride = stride, padding = 1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size = 3, stride = stride,
                                   padding = dilation, dilation = dilation, groups = cardinality, bias = False))
            bns.append(BatchNorm(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes, kernel_size = 1, bias = False)
        self.bn3 = BatchNorm(planes)

        self.relu = nn.ReLU(inplace = True)
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x, residual = None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class SEBottle2neck(nn.Module):
    """
    RexNeXt bottleneck type C
    """
    expansion = 2

    def __init__(self, inplanes, planes, stride = 1, dilation = 1, baseWidth = 28, scale = 4, reduction = 16):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': frist blokc of a new stage.
        """
        super(SEBottle2neck, self).__init__()
        if stride != 1:
            stype = 'stage'
        else:
            stype = 'normal'
        width = int(math.floor(planes * (baseWidth / 128.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size = 1, bias = False)
        self.bn1 = BatchNorm(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size = 3, stride = stride, padding = 1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size = 3, stride = stride,
                                   padding = dilation, dilation = dilation, bias = False))
            bns.append(BatchNorm(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes, kernel_size = 1, bias = False)
        self.bn3 = BatchNorm(planes)

        self.se = SELayer(planes, reduction)
        self.relu = nn.ReLU(inplace = True)
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x, residual = None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        out += residual
        out = self.relu(out)

        return out


class SEBottle2neckX(nn.Module):
    """
    RexNeXt bottleneck type C
    """
    expansion = 2
    cardinality = 8

    def __init__(self, inplanes, planes, stride = 1, dilation = 1, scale = 4, reduction = 16):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': frist blokc of a new stage.
        """
        super(SEBottle2neckX, self).__init__()
        if stride != 1:
            stype = 'stage'
        else:
            stype = 'normal'
        cardinality = SEBottle2neckX.cardinality
        width = bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size = 1, bias = False)
        self.bn1 = BatchNorm(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size = 3, stride = stride, padding = 1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size = 3, stride = stride,
                                   padding = dilation, dilation = dilation, groups = cardinality, bias = False))
            bns.append(BatchNorm(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes, kernel_size = 1, bias = False)
        self.bn3 = BatchNorm(planes)

        self.se = SELayer(planes, reduction)
        self.relu = nn.ReLU(inplace = True)
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x, residual = None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        out += residual
        out = self.relu(out)

        return out
