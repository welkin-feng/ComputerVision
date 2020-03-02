#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  __init__.py.py

"""

__author__ = 'Welkin'
__date__ = '2019/6/21 08:34'

from .util_modules import *
from .alexnet import *
from .inception_v1 import *
from .vgg import *
from .mobilenet_v1 import *
from .resnet import *
from .yolo_v1 import *
from .yolo_v2 import *
from .res2net_module import *
from .dla import *
from .hrnet import *
from .resnet_modified import *


def get_model(config):
    return globals()[config.architecture](**config)
