#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  detection_trainer.py

"""

__author__ = 'Welkin'
__date__ = '2019/10/11 15:12'

import torch
from cvmodels import voc_util

from cvmodels.trainer import Trainer


class DetectionTrainer(Trainer):

    def __init__(self, work_path, resume = False, config_dict = None):
        super().__init__(work_path, resume, config_dict)

    def train_step(self, train_loader):
        self.all_cls_record = [dict(gt_num = 0, confidence_score = [], tp_list = []) for _ in
                               range(self.config.num_classes)]
        if self.epoch % self.config.size_change_freq == 0:
            train_loader = self._get_dataloader(self._get_transforms(train_mode = True), train_mode = True)
        return super().train_step(train_loader)

    def test(self, test_loader):
        self.all_cls_record = [dict(gt_num = 0, confidence_score = [], tp_list = []) for _ in
                               range(self.config.num_classes)]
        return super().test(test_loader)

    def _get_transforms(self, train_mode = True):
        # size_list = (320, 352, 384, 416, 448, 480, 512, 544, 576, 608)
        size_list = self.config.size_list
        size = size_list[0]
        if self.epoch % self.config.size_change_freq == 0:
            size = size_list[self.epoch // self.config.size_change_freq % len(size_list)]

        return voc_util.data_augmentation(self.config, size, train_mode)

    def _get_dataloader(self, transforms, train_mode = True):
        return voc_util.get_data_loader(transforms, self.config, train_mode)

    def _get_model_outputs(self, inputs, targets, train_mode = True):
        if train_mode:
            outputs, loss = self.net(inputs, targets)
        else:
            outputs, loss = self.net(inputs)

        return outputs, loss

    def _calculate_acc(self, outputs, targets, train_mode = True):
        """
        calculate mAP

        Args:
            outputs:
            targets:
            train_mode:

        Returns:
            mAP
        """
        for idx, (pred, gt) in enumerate(zip(outputs, targets)):
            pred_boxes = pred['boxes']
            pred_labels = pred['labels']
            pred_scores = pred['scores']
            gt_difficult = gt['difficult']
            # 去除当前图片中 diffcult 的目标
            gt_boxes = gt['boxes']
            gt_labels = gt['labels']

            cls_set = torch.cat((pred_labels, gt_labels)).unique().tolist()

            for i in cls_set:
                pred_mask = pred_labels == i
                gt_mask = gt_labels == i
                gt_num, tp_list, confidence_score = voc_util.calculate_tp(pred_boxes[pred_mask], pred_scores[pred_mask],
                                                                          gt_boxes[gt_mask], gt_difficult[gt_mask])
                self.all_cls_record[i]['gt_num'] += gt_num
                self.all_cls_record[i]['tp_list'].extend(tp_list)
                self.all_cls_record[i]['confidence_score'].extend(confidence_score)

    def _get_acc(self):
        all_cls_AP = [voc_util.voc_ap(*voc_util.calculate_pr(**r), use_07_metric = True) for r in self.all_cls_record]
        mAP = sum(all_cls_AP) / len(self.all_cls_record)
        return mAP
