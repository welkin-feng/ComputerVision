#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  psmnet.py

"""

__author__ = 'Welkin'
__date__ = '2019/7/11 16:57'

import torch.nn as nn

from .psmnet_modules import CNN_Module, SPP_Module, CostVolume, StackedHourglass3dCNN


class PSM_Net(nn.Module):
    """  """

    def __init__(self, max_disparity = 128, spp_scales = (64, 32, 16, 8)):
        """ Constructor for PSM_Net """
        super().__init__()
        self._init_model(max_disparity, spp_scales)

    def _init_model(self, max_disparity, spp_scales):
        self.cnn = CNN_Module()
        self.spp = SPP_Module(spp_scales)
        self.cost_volume = CostVolume(max_disparity)
        self.cnn_3d = StackedHourglass3dCNN()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, left_img, right_img):
        l_conv2, l_conv4 = self.cnn(left_img)
        r_conv2, r_conv4 = self.cnn(right_img)

        l_spp = self.spp(l_conv2, l_conv4)
        r_spp = self.spp(r_conv2, r_conv4)

        cost = self.cost_volume(l_spp, r_spp)

        if self.training:
            out1, out2, out3 = self.cnn_3d(cost)
            return out1, out2, out3

        out3 = self.cnn_3d(cost)
        return out3
