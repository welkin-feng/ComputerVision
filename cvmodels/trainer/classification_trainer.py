#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  classification_trainer.py

"""

__author__ = 'Welkin'
__date__ = '2019/10/11 15:09'

import torch.nn as nn
from cvmodels import cifar_util

from torchvision.transforms import transforms
from cvmodels.trainer import Trainer


class ClassificationTrainer(Trainer):
    """  """

    def __init__(self, work_path, resume = False, config_dict = None):
        super().__init__(work_path, resume, config_dict)

        # 设置loss计算函数
        # define loss
        self.criterion = nn.CrossEntropyLoss()

    def train_step(self, train_loader):
        self.correct = 0
        self.total = 0
        self._acc = 0
        return super().train_step(train_loader)

    def test(self, test_loader):
        self.correct = 0
        self.total = 0
        self._acc = 0
        return super().test(test_loader)

    def _get_transforms(self, train_mode = True):
        return transforms.Compose(cifar_util.data_augmentation(self.config, train_mode))

    def _get_dataloader(self, transforms, train_mode = True):
        return cifar_util.get_data_loader(transforms, self.config, train_mode)

    def _get_model_outputs(self, inputs, targets, train_mode = True):
        if train_mode:
            if self.config.mixup:
                inputs, self.targets_a, self.targets_b, self.lam = cifar_util.mixup_data(inputs, targets,
                                                                                         self.config.mixup_alpha,
                                                                                         self.device)
                outputs = self.net(inputs)
                loss = cifar_util.mixup_criterion(self.criterion, outputs, self.targets_a, self.targets_b, self.lam)
            else:
                outputs = self.net(inputs)
                if isinstance(outputs, tuple):
                    # losses for multi classifier
                    losses = list(map(self.criterion, outputs, [targets] * len(outputs)))
                    losses = list(map(lambda x, y: x * y, self.config.classifier_weight, losses))
                    loss = sum(losses[:self.config.num_classifier])
                    outputs = outputs[0]
                else:
                    loss = self.criterion(outputs, targets)
        else:
            outputs = self.net(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = self.criterion(outputs, targets)

        return outputs, loss

    def _calculate_acc(self, outputs, targets, train_mode = True):
        if self.config.mixup:
            self._acc, self.correct, self.total, = cifar_util.calculate_acc(outputs, targets, self.config, self.correct,
                                                                            self.total, train_mode = train_mode,
                                                                            lam = self.lam, targets_a = self.targets_a,
                                                                            targets_b = self.targets_b)
        else:
            self._acc, self.correct, self.total, = cifar_util.calculate_acc(outputs, targets, self.config, self.correct,
                                                                            self.total, train_mode = train_mode)

    def _get_acc(self):
        return self._acc
