#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  detection_trainer.py

"""

__author__ = 'Welkin'
__date__ = '2019/10/11 15:12'

import torch
import voc_util

from torchvision.transforms import transforms
from torchvision.datasets import vision
from trainer import Trainer


class DetectionTrainer(Trainer):

    def __init__(self, work_path, resume = False, config_dict = None):
        super().__init__(work_path, resume, config_dict)

    def train_step(self, train_loader):
        self.all_cls_pr = [dict(recall = [], precision = []) for _ in range(self.config.num_classes)]
        if self.epoch % self.config.size_change_freq == 0:
            train_loader = self._get_dataloader(self._get_transforms(train_mode = True), train_mode = True)
        return super().train_step(train_loader)

    def test(self, test_loader):
        self.all_cls_pr = [dict(recall = [], precision = []) for _ in range(self.config.num_classes)]
        return super().test(test_loader)

    def _get_transforms(self, train_mode = True):
        # size_list = (320, 352, 384, 416, 448, 480, 512, 544, 576, 608)
        size_list = self.config.size_list
        size = size_list[0]
        if self.epoch % self.config.size_change_freq == 0:
            size = size_list[self.epoch // 5 % len(size_list)]

        if train_mode:
            img_trans = transforms.Compose([transforms.ColorJitter(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            target_trans = voc_util.VOCTargetTransform()
            trans = voc_util.VOCTransformCompose([vision.StandardTransform(None, target_trans),
                                                  voc_util.VOCTransformFlip(0.5, 0.5),
                                                  voc_util.VOCTransformResize(size = size),
                                                  voc_util.VOCTransformRandomScale(scale = (0.8, 1.2)),
                                                  voc_util.VOCTransformRandomExpand(ratio = (0.8, 1.2)),
                                                  voc_util.VOCTransformRandomCrop(size = size),
                                                  vision.StandardTransform(img_trans, None)])
            return trans
        else:
            img_trans = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            target_trans = voc_util.VOCTargetTransform()
            trans = voc_util.VOCTransformCompose([vision.StandardTransform(None, target_trans),
                                                  voc_util.VOCTransformResize(size = (size, size),
                                                                              scale_with_padding = True),
                                                  vision.StandardTransform(img_trans, None)])
            return trans

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
                recall, precision = voc_util.calculate_pr(pred_boxes[pred_mask], pred_scores[pred_mask],
                                                          gt_boxes[gt_mask], gt_difficult[gt_mask])
                self.all_cls_pr[i]['recall'].extend(recall)
                self.all_cls_pr[i]['precision'].extend(precision)

    def _get_acc(self):
        all_cls_AP = [voc_util.voc_ap(pr['recall'], pr['precision']) for pr in self.all_cls_pr]
        mAP = sum(all_cls_AP) / len(self.all_cls_pr)
        return mAP
