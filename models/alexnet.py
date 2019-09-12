#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  alexnet.py

"""

__author__ = 'Welkin'
__date__ = '2019/6/18 03:06'

import torch.nn as nn

__all__ = ['alexnet', 'alexnet_cifar10']


class AlexNet(nn.Module):
    """ input size should be 227x227x3  """

    def __init__(self, num_classes, in_size = 227):
        """ Constructor for AlexNet """
        super().__init__()
        self._init_model(num_classes, in_size)
        self._initialize_weights()

    def _init_model(self, num_classes, in_size):
        final_size = (in_size - 35) // 32
        if final_size <= 0:
            raise ValueError("`in_size` is too small")

        self.conv = nn.Sequential(
            # (227-11)/4+1=55
            nn.Conv2d(3, 96, kernel_size = 11, stride = 4),
            nn.ReLU(inplace = True),
            nn.LocalResponseNorm(size = 5, k = 2),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            # (55-3)/2+1=27
            nn.Conv2d(96, 256, kernel_size = 5, stride = 1, padding = 2),
            nn.ReLU(inplace = True),
            nn.LocalResponseNorm(size = 5, k = 2),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            # (27-3)/2+1=13
            nn.Conv2d(256, 384, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(384, 384, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(384, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2),
            # (13-3)/2+1=6
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * final_size ** 2, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


class AlexNet_cifar10(AlexNet):
    """ AlexNet for cifar10  """

    def __init__(self, num_classes = 10, in_size = 32):
        """ Constructor for AlexNet """
        super().__init__(num_classes, in_size)

    def _init_model(self, num_classes, in_size):
        final_size = (in_size + 23) // 24

        self.conv = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size = 11, stride = 3, padding = 5),
            nn.ReLU(inplace = True),
            nn.LocalResponseNorm(size = 5, k = 2),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),
            nn.Conv2d(96, 256, kernel_size = 5, stride = 1, padding = 2),
            nn.ReLU(inplace = True),
            nn.LocalResponseNorm(size = 5, k = 2),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),
            nn.Conv2d(256, 384, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(384, 384, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(384, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * final_size ** 2, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )


def alexnet(num_classes, in_size = 227):
    return AlexNet(num_classes, in_size)


def alexnet_cifar10(num_classes, in_size = 32):
    return AlexNet_cifar10(num_classes, in_size)


if __name__ == '__main__':
    # test
    import sys

    fn_list = ['alexnet']
    for fn in fn_list:
        f = getattr(sys.modules[__name__], fn)
        model = f(10)
        print(' ---', fn, '---')
        for k, v in model.state_dict().items():
            print(k)
        print()
