#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  spp.py

"""

__author__ = 'Welkin'
__date__ = '2019/7/2 03:23'

import math
import torch
import torch.nn as nn

from .util_modules import *

__all__ = ['SPP', 'SPP_multi_level']


class SPP(nn.Module):
    """   """

    def __init__(self, in_size, out_size):
        """ Constructor for SPP """
        super().__init__()
        assert (in_size >= out_size)
        size_x = math.ceil(in_size / out_size)
        stride = int(in_size / out_size)
        self.pool = nn.MaxPool2d(kernel_size = size_x, stride = stride)

    def forward(self, x):
        return self.pool(x)


class SPP_multi_level(nn.Module):
    """  """

    def __init__(self, in_size, out_size_list):
        """ Constructor for SPP_multi_level """
        super().__init__()
        self.flatten = Flatten()
        self.spp_list = [SPP(in_size, out_size) for out_size in out_size_list]

    def forward(self, x):
        out_list = tuple(self.flatten(spp(x)) for spp in self.spp_list)
        out = torch.cat(out_list)
        return out