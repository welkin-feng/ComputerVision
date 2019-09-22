#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  cifar_util.py

"""

__author__ = 'Welkin'
__date__ = '2019/7/13 16:19'

import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

__all__ = ['Cutout', 'data_augmentation', 'get_data_loader',
           'mixup_data', 'mixup_criterion', 'calculate_acc']


class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


def data_augmentation(config, train_mode = True):
    aug = []
    if train_mode:
        # random crop
        if config.augmentation.random_crop:
            aug.append(transforms.RandomCrop(config.input_size, padding = 4))
        # horizontal filp
        if config.augmentation.random_horizontal_filp:
            aug.append(transforms.RandomHorizontalFlip())

    aug.append(transforms.ToTensor())
    # normalize  [- mean / std]
    if config.augmentation.normalize:
        if config.dataset == 'cifar10':
            aug.append(transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        elif config.dataset == 'cifar100':
            aug.append(transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)))

    if train_mode and config.augmentation.cutout:
        # cutout
        aug.append(Cutout(n_holes = config.augmentation.holes,
                          length = config.augmentation.length))
    return aug


def get_data_loader(transform, config, train_mode):
    assert config.dataset in ['cifar10', 'cifar100']
    if config.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(root = config.data_path, train = train_mode, download = True,
                                               transform = transform)
    else:
        dataset = torchvision.datasets.CIFAR100(
            root = config.data_path, train = train_mode, download = True, transform = transform)

    data_loader = DataLoader(dataset, batch_size = config.batch_size,
                             shuffle = train_mode, num_workers = config.workers)

    return data_loader


def mixup_data(x, y, alpha, device):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def calculate_acc(outputs, targets, config, correct = 0, total = 0, train_mode = True, **kwargs):
    if not config.dataset in ['cifar10', 'cifar100']:
        raise ValueError("`dataset` in `config` should be 'cifar10' or 'cifar100'")
    _, predicted = outputs.max(1)
    total += targets.size(0)
    if train_mode and config.mixup:
        lam, targets_a, targets_b = kwargs['lam'], kwargs['targets_a'], kwargs['targets_b']
        correct += (lam * predicted.eq(targets_a).sum().item()
                    + (1 - lam) * predicted.eq(targets_b).sum().item())
    else:
        correct += predicted.eq(targets).sum().item()
    acc = correct / total
    return acc, correct, total
