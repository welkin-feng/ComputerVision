#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  kitti_util.py

"""

__author__ = 'Welkin'
__date__ = '2019/7/13 16:25'

from torch.utils.data import DataLoader
from dataloader.kitti_loader import *


def data_augmentation(config):
    aug = []

    # random crop
    if config.augmentation.random_crop:
        aug.append(RandomCrop(config.input_size))

    aug.append(ToTensor())
    # normalize  [- mean / std]
    if config.augmentation.normalize:
        if config.dataset == 'kitti2015':
            aug.append(Normalize([0.406, 0.456, 0.485], [0.225, 0.224, 0.229]))
        elif config.dataset == 'kitti2012':
            aug.append(Normalize([0.406, 0.456, 0.485], [0.225, 0.224, 0.229]))

    return aug


def get_data_loader(transform_train, transform_test, config):
    assert config.dataset.lower() in ['kitti2015', 'kitti2012', 'sceneflow']
    if config.dataset.lower() == "kitti2015":
        trainset = KITTI2015(
            root = config.data_path, mode = 'train', transform = transform_train, download = False)

        testset = KITTI2015(
            root = config.data_path, mode = 'validate', transform = transform_test, download = False)
    elif config.dataset.lower() == "kitti2012":
        pass

    train_loader = DataLoader(trainset, batch_size = config.batch_size,
                              shuffle = True, num_workers = config.workers)

    test_loader = DataLoader(testset, batch_size = config.test_batch,
                             shuffle = False, num_workers = config.workers)
    return train_loader, test_loader


def compute_npx_error(prediction, gt, n, use_mask = False):
    # computing n-px error
    if use_mask:
        mask = gt > 0
        prediction = prediction[mask]
        gt = gt[mask]
    dif = (gt - prediction).abs()

    correct = (dif < n) | (dif < gt * 0.05)  # Tensor, size [N, H, W]

    correct = float(correct.sum())
    total = float(gt.size(0))

    return correct, total


def calculate_acc(outputs, targets, config, correct = 0, total = 0, mask = None, is_train = True, **kwargs):
    if config.dataset in ['kitti2015', 'kitti2012', 'sceneflow']:
        if mask is not None:
            correct, total = compute_npx_error(outputs[mask], targets[mask], n = 3)
        else:
            correct, total = compute_npx_error(outputs, targets, n = 3)
        train_acc = correct / total
    return train_acc, correct, total
