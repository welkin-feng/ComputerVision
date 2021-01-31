#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision

File Name:  fast_autoaugment.py

"""
import random
from .fast_autoaugment_policies import apply_augment
from .fast_autoaugment_params import fa_resnet50_rimagenet_policies, fa_reduced_svhn_policies


class Augmentation(object):
    def __init__(self, policies, n_layers=1, fillcolor=None):
        """
        Args:
            policies (list): policy list
            n_layers (int): number of augs
            fillcolor (tuple): color to fill
        """
        self.policies = policies
        self.n_layers = n_layers
        self.fillcolor = fillcolor

    def __call__(self, img):
        for _ in range(self.n_layers):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                if random.random() <= pr:
                    img = apply_augment(img, name, level, self.fillcolor)
        return img


class FastAutoAugmentImageNet(Augmentation):
    def __init__(self, n_layers=1, fillcolor=None, filter=None):
        policies = fa_resnet50_rimagenet_policies(filter)
        super(FastAutoAugmentImageNet, self).__init__(policies, n_layers, fillcolor)


class FastAutoAugmentSVHN(Augmentation):
    def __init__(self, n_layers=1, fillcolor=None):
        policies = fa_reduced_svhn_policies(filter)
        super(FastAutoAugmentSVHN, self).__init__(policies, n_layers, fillcolor)
