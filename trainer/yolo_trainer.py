#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  yolo_trainer.py

"""

__author__ = 'Welkin'
__date__ = '2019/10/11 15:14'

import voc_util, cifar_util

from trainer import DetectionTrainer, ClassificationTrainer


class YOLOTrainer(DetectionTrainer):
    def __init__(self, work_path, resume = False, config_dict = None):
        super().__init__(work_path, resume, config_dict)

    def test(self, test_loader):
        if self.epoch < 10:
            return
        return super().test(test_loader)

    def _calculate_acc(self, outputs, targets, train_mode = True):
        if train_mode and self.epoch < 10:
            return
        super()._calculate_acc(outputs, targets, train_mode)

    def _get_model_outputs(self, inputs, targets, train_mode = True):
        if train_mode:
            if self.epoch < 6:
                kwargs = {'class_scale': 1, 'coord_scale': 0, 'object_scale': 0,
                          'noobject_scale': 0, 'prior_scale': 0}
                outputs, loss = self.net(inputs, targets, get_prior_anchor_loss = False, **kwargs)
            # elif self.epoch < 10:
            #     kwargs = {'class_scale': 1, 'coord_scale': 0, 'object_scale': 0,
            #               'noobject_scale': 0, 'prior_scale': 0.01}
            #     outputs, loss = self.net(inputs, targets, get_prior_anchor_loss = True, **kwargs)
            elif self.epoch < 15:
                kwargs = {'class_scale': 1, 'coord_scale': 0, 'object_scale': 5,
                          'noobject_scale': 1, 'prior_scale': 0.01}
                outputs, loss = self.net(inputs, targets, get_prior_anchor_loss = True, **kwargs)
            else:
                # kwargs = {'class_scale': 1, 'coord_scale': None, 'object_scale': 5,
                #           'noobject_scale': 1, 'prior_scale': 0.01}
                outputs, loss = self.net(inputs, targets)
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
            size = size_list[self.epoch // 5 % len(size_list)]
        return transforms.Compose([transforms.RandomHorizontalFlip(),
                                   transforms.RandomVerticalFlip(),
                                   voc_util.RandomScale(),
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
        self._acc, self.correct, self.total, = cifar_util.calculate_acc(outputs, targets, self.config, self.correct,
                                                                        self.total, train_mode = train_mode)
