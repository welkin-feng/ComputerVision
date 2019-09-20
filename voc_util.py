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
import numpy as np

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


def calculate_pr(pred_boxes, pred_scores, gt_boxes, gt_difficult, score_range = torch.arange(0, 1, 0.1),
                 iou_thresh = 0.5):
    """
    calculate all p-r pairs among different score_thresh for one class of one image.

    Args:
        pred_boxes:
        pred_scores:
        gt_boxes:
        score_range:
        iou_thresh:

    Returns:
        recall
        precision

    """
    if gt_boxes.numel() == 0:
        return [0], [0]

    from collections import Iterable
    assert isinstance(score_range, Iterable), "`score_range` should be iterable"

    recall = []
    precision = []
    for s in score_range:
        pb = pred_boxes[pred_scores > s]
        if pb.numel() == 0:
            recall.append(0)
            precision.append(0)
            continue
        ious = torch.zeros((len(gt_boxes), len(pb))).to(pb)
        for i in range(len(gt_boxes)):
            gb = gt_boxes[i]
            area_pb = (pb[:, 2] - pb[:, 0]) * (pb[:, 3] - pb[:, 1])
            area_gb = (gb[2] - gb[0]) * (gb[3] - gb[1])
            xx1 = pb[:, 0].clamp(min = gb[0].item())  # [N-1,]
            yy1 = pb[:, 1].clamp(min = gb[1].item())
            xx2 = pb[:, 2].clamp(max = gb[2].item())
            yy2 = pb[:, 3].clamp(max = gb[3].item())
            inter = (xx2 - xx1).clamp(min = 0) * (yy2 - yy1).clamp(min = 0)  # [N-1,]
            ious[i] = inter / (area_pb + area_gb - inter)

        max_ious, max_ious_idx = ious.max(dim = 0)

        # 去掉 max_iou 属于 difficult 目标的预测框
        not_difficult_idx = torch.where(gt_difficult != 0)[0]
        if not_difficult_idx.numel() == 0:
            continue
        pb_mask = (max_ious == ious[not_difficult_idx].max(dim = 0)[0])
        max_ious, max_ious_idx = max_ious[pb_mask], max_ious_idx[pb_mask]
        if max_ious_idx.numel() == 0:
            recall.append(0)
            precision.append(0)
            continue

        tp = max_ious_idx[max_ious > iou_thresh].unique().numel()
        recall.append(tp / not_difficult_idx.numel())
        precision.append(tp / max_ious_idx.numel())

    return recall, precision


def voc_ap(rec, prec, use_07_metric = False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if isinstance(rec, (tuple, list)):
        rec = np.array(rec)
    if isinstance(prec, (tuple, list)):
        prec = np.array(prec)
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
