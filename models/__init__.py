#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  __init__.py.py

"""

__author__ = 'Welkin'
__date__ = '2019/6/21 08:34'

from .AlexNet import *
from .Inception import *


def get_model(config):
    return globals()[config.architecture](config.num_classes, config.input_size)
