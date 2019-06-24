#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  __init__.py.py

"""

__author__ = 'Welkin'
__date__ = '2019/6/21 08:34'

from .alexnet import *
from .inception_v1 import *
from .vgg import *


def get_model(config):
    return globals()[config.architecture](config.num_classes, config.input_size)
