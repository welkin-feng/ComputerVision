#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  fast_autoaugment.py

"""
from .fast_autoaugment_policies import *
from .fast_autoaugment_params import fa_resnet50_rimagenet, fa_reduced_svhn


class Augmentation(object):
    def __init__(self, policies):
        self.policies = policies

    def __call__(self, img):
        for _ in range(1):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                if random.random() > pr:
                    continue
                img = apply_augment(img, name, level)
        return img


class FastAutoAugmentImageNet(Augmentation):
    def __init__(self):
        policies = fa_resnet50_rimagenet()
        super(FastAutoAugmentImageNet, self).__init__(policies)


class FastAutoAugmentSVHN(Augmentation):
    def __init__(self):
        policies = fa_reduced_svhn()
        super(FastAutoAugmentSVHN, self).__init__(policies)
