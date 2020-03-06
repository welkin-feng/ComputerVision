#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  avg_pool_same.py

"""

import math
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# Calculate asymmetric TensorFlow-like 'SAME' padding for a convolution
def get_same_padding(x: int, k: int, s: int, d: int):
    return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)


def pad_same(x, k: List[int], s: List[int], d: List[int] = (1, 1)):
    ih, iw = x.size()[-2:]
    pad_h, pad_w = get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(iw, k[1], s[1], d[1])
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
    return x


def avg_pool2d_same(x, kernel_size: List[int], stride: List[int], padding: List[int] = (0, 0),
                    ceil_mode: bool = False, count_include_pad: bool = True):
    x = pad_same(x, kernel_size, stride)
    return F.avg_pool2d(x, kernel_size, stride, (0, 0), ceil_mode, count_include_pad)


class AvgPool2dSame(nn.AvgPool2d):
    """ Tensorflow like 'SAME' wrapper for 2D average pooling
    """

    def __init__(self, kernel_size: int, stride = None, padding = 0, ceil_mode = False, count_include_pad = True):
        kernel_size = (kernel_size, kernel_size)
        stride = (stride, stride)
        super(AvgPool2dSame, self).__init__(kernel_size, stride, (0, 0), ceil_mode, count_include_pad)

    def forward(self, x):
        return avg_pool2d_same(
            x, self.kernel_size, self.stride, self.padding, self.ceil_mode, self.count_include_pad)
