#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  yolo_trainer.py

"""

__author__ = 'Welkin'
__date__ = '2019/10/11 15:14'

from cvmodels import voc_util

from cvmodels.trainer import DetectionTrainer, ClassificationTrainer
from cvmodels.util import get_learning_rate_scheduler


class YOLOTrainer(DetectionTrainer):
    def __init__(self, work_path, resume = False, config_dict = None):
        super().__init__(work_path, resume, config_dict)

    def test(self, test_loader):
        if self.epoch < 10:
            return
        return super().test(test_loader)

    def _get_model_outputs(self, inputs, targets, train_mode = True):
        if train_mode:
            kwargs = {'class_scale': 1, 'coord_scale': None, 'object_scale': 5,
                      'noobject_scale': 1, 'prior_scale': 0.1}
            get_prior_anchor_loss = False
            if self.epoch < 10:
                get_prior_anchor_loss = True

            if self.epoch in [1, ]:
                for i, param_group in enumerate(self.optimizer.param_groups):
                    param_group['lr'] = self.config.lr_scheduler.base_lr
                self.lr_scheduler = get_learning_rate_scheduler(self.optimizer, self.epoch, self.config)

            kwargs['get_prior_anchor_loss'] = get_prior_anchor_loss
            outputs, loss = self.net(inputs, targets, **kwargs)
        else:
            outputs, loss = self.net(inputs)

        return outputs, loss


class YOLOBackboneTrainer(ClassificationTrainer):
    def train_step(self, train_loader):
        if self.epoch % self.config.size_change_freq == 0:
            train_loader = self._get_dataloader(self._get_transforms(train_mode = True), train_mode = True)
        return super().train_step(train_loader)

    def _get_transforms(self, train_mode = True):
        from torchvision import transforms
        size_list = self.config.size_list
        size = size_list[0]
        if self.epoch % self.config.size_change_freq == 0:
            size = size_list[self.epoch // self.config.size_change_freq % len(size_list)]
        return transforms.Compose([transforms.ColorJitter(),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomVerticalFlip(),
                                   transforms.RandomRotation(180),
                                   voc_util.RandomScale(scale = (0.5, 1.2)),
                                   transforms.RandomCrop(size, pad_if_needed = True),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), ])

    def _get_dataloader(self, transforms, train_mode = True):
        import os
        from torch.utils.data import DataLoader
        if train_mode:
            root = os.path.join(self.config.data_path, 'voctrainval_06-nov-2007')
            image_set = 'trainval'
        else:
            root = os.path.join(self.config.data_path, 'voctest_06-nov-2007')
            image_set = 'test'
        dataset = voc_util.VOCClassification(root = root, year = '2007', image_set = image_set, transform = transforms)
        return DataLoader(dataset, batch_size = self.config.batch_size, shuffle = train_mode,
                          num_workers = self.config.workers)

    def _get_model_outputs(self, inputs, targets, train_mode = True):
        outputs = self.net(inputs)
        loss = self.criterion(outputs, targets)
        return outputs, loss

    def _calculate_acc(self, outputs, targets, train_mode = True):
        self.total += targets.size(0)
        # _, predicted = outputs.max(1)
        # self.correct += predicted.eq(targets).sum().item()
        _, predicted = outputs.sort(dim = -1)
        predicted = predicted[:, -5:]
        for i in range(5):
            self.correct += predicted[:, i].eq(targets).sum().item()
        self._acc = self.correct / self.total
