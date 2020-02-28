#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  fast_autoaugment.py

"""
from .fast_autoaugment_policies import *
from .fast_autoaugment_params import fa_resnet50_rimagenet, fa_reduced_svhn


class Augmentation(object):
    def __init__(self, policies, n_layers = 1, fillcolor = None):
        self.policies = policies
        self.n_layers = n_layers
        self.fillcolor = fillcolor

    def __call__(self, img):
        for _ in range(self.n_layers):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                if random.random() > pr:
                    continue
                img = apply_augment(img, name, level, self.fillcolor)
        return img


class FastAutoAugmentImageNet(Augmentation):
    def __init__(self, n_layers = 1, fillcolor = None, filter = None):
        policies = fa_resnet50_rimagenet(filter)
        super(FastAutoAugmentImageNet, self).__init__(policies, n_layers, fillcolor)


class FastAutoAugmentSVHN(Augmentation):
    def __init__(self, n_layers = 1, fillcolor = None):
        policies = fa_reduced_svhn(filter)
        super(FastAutoAugmentSVHN, self).__init__(policies, n_layers, fillcolor)
