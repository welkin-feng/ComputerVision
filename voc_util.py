#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  voc_util.py

"""

__author__ = 'Welkin'
__date__ = '2019/9/16 15:36'

import torch
import os

from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader


class VOCTargetTransform():

    def __init__(self):
        self.cls_to_idx = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4,
                           'bus': 5, 'car': 6, 'cat': 7, 'chair': 8, 'cow': 9,
                           'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13, 'person': 14,
                           'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}
        self.idx_to_cls = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

    def __call__(self, target):
        coords = []
        labels = []
        diff = []
        obj = target['annotation']['object']
        if not isinstance(obj, list):
            obj = [obj, ]
        for t in obj:
            coords.append([int(t['bndbox']['xmin']), int(t['bndbox']['ymin']),
                           int(t['bndbox']['xmax']), int(t['bndbox']['ymax']), ])
            labels.append(self.cls_to_idx[t['name']])
            diff.append(int(t['difficult']))
        target = dict(boxes = torch.tensor(coords),
                      labels = torch.tensor(labels).long(),
                      difficult = torch.tensor(diff).long())
        return target


def get_data_loader(transform_train, transform_test, config):
    assert config.dataset in ['voc2007', 'voc2012']
    if config.dataset == "voc2007":
        train_root = os.path.join(config.data_path, 'voctrainval_06-nov-2007')
        test_root = os.path.join(config.data_path, 'voctest_06-nov-2007')
        year = '2007'
    elif config.dataset == "voc2012":
        # train_root = os.path.join(config.data_path, 'voctrainval_06-nov-2007')
        # test_root = os.path.join(config.data_path, 'voctest_06-nov-2007')
        year = '2012'

    trainset = VOCDetection(root = train_root, year = year, image_set = 'trainval', transform = transform_train)
    testset = VOCDetection(root = test_root, year = year, image_set = 'test', transform = transform_test)

    train_loader = DataLoader(trainset, batch_size = config.batch_size, shuffle = True, num_workers = config.workers)
    test_loader = DataLoader(testset, batch_size = config.test_batch, shuffle = False, num_workers = config.workers)

    return train_loader, test_loader
