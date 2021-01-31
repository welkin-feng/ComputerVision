#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision

File Name:  __init__.py.py

"""

__author__ = 'Welkin'
__date__ = '2020/2/11 12:33'


from .autoaugment import ImageNetPolicy, CIFAR10Policy, SVHNPolicy
from .fast_autoaugment import FastAutoAugmentImageNet, FastAutoAugmentSVHN
