"""Split-Attention"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Conv2d, Module, Linear, BatchNorm2d, ReLU
from torch.nn.modules.utils import _pair
from .dropblock import DropBlock2D

__all__ = ['SplAtConv2d']


class SplAtConv2d(Module):
    """Split-Attention Conv2d
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, bias=True, radix=2, reduction_factor=4,
                 rectify=False, rectify_avg=False, act_layer=nn.ReLU, norm_layer=None, drop_block=None, **kwargs):
        super(SplAtConv2d, self).__init__()
        padding = _pair(padding)
        rectify = rectify and (padding[0] > 0 or padding[1] > 0)
        inter_channels = max(in_channels * radix // reduction_factor, 32)
        self.radix = radix
        self.drop_block = drop_block
        if rectify:
            from rfconv import RFConv2d
            self.conv = RFConv2d(in_channels, out_channels * radix, kernel_size, stride, padding, dilation,
                                 groups=groups * radix, bias=bias, average_mode=rectify_avg, **kwargs)
        else:
            self.conv = Conv2d(in_channels, out_channels * radix, kernel_size, stride, padding, dilation,
                               groups=groups * radix, bias=bias, **kwargs)
        self.bn0 = norm_layer(out_channels * radix) if norm_layer is not None else None
        self.act0 = act_layer(inplace=True)
        self.fc1 = Conv2d(out_channels, inter_channels, 1, groups=groups)
        self.bn1 = norm_layer(inter_channels) if norm_layer is not None else None
        self.act1 = act_layer(inplace=True)
        self.fc2 = Conv2d(inter_channels, out_channels * radix, 1, groups=groups)
        self.rsoftmax = RadixSoftmax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        if self.bn0 is not None:
            x = self.bn0(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act0(x)

        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            if torch.__version__ < '1.5':
                splited = torch.split(x, int(rchannel // self.radix), dim=1)
            else:
                splited = torch.split(x, rchannel // self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)

        if self.bn1 is not None:
            gap = self.bn1(gap)
        gap = self.act1(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        if self.radix > 1:
            if torch.__version__ < '1.5':
                attens = torch.split(atten, int(rchannel // self.radix), dim=1)
            else:
                attens = torch.split(atten, rchannel // self.radix, dim=1)
            out = sum([att * split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()


class RadixSoftmax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x
