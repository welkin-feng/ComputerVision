#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  yolo_trainer.py

"""

__author__ = 'Welkin'
__date__ = '2019/10/11 15:14'

from trainer import DetectionTrainer


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
