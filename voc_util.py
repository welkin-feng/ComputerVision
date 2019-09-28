#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  voc_util.py

"""

__author__ = 'Welkin'
__date__ = '2019/9/16 15:36'

import warnings
import os
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F

from torch.utils.data import DataLoader


class VOCTransformCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class VOCTransformFlip(object):
    def __init__(self, horizontal_flip_prob = 0.5, vertical_flip_prob = 0.5):
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob

    def __call__(self, img, target):
        w, h = img.size
        if random.random() < self.vertical_flip_prob:
            img = F.vflip(img)
            target['boxes'][:, (0, 2)] = w - target['boxes'][:, (2, 0)]
        if random.random() < self.horizontal_flip_prob:
            img = F.hflip(img)
            target['boxes'][:, (1, 3)] = h - target['boxes'][:, (3, 1)]
        return img, target


class VOCTransformResize(object):
    def __init__(self, size, scale_with_padding = False):
        """
        Args:
            size (sequence or int): Desired output size. If size is a sequence like (h, w), output size will be matched
                to this. If size is an int, smaller edge of the image will be matched to this number.
                i.e, if height > width, then image will be rescaled to (size * height / width, size)
            scale_with_padding
        """
        assert isinstance(size, int) or (isinstance(size, (list, tuple)) and len(size) == 2)
        self.size = size
        self.scale_with_padding = scale_with_padding

    def __call__(self, img, target):
        w, h = img.size
        if isinstance(self.size, int):
            w_ratio, h_ratio = self.size / min(w, h), self.size / min(w, h)
        else:
            if w / h != self.size[1] / self.size[0] and self.scale_with_padding:
                if w / h < self.size[1] / self.size[0]:
                    pad = (int((h * self.size[1] / self.size[0] - w) / 2), 0)
                else:
                    pad = (0, int((w * self.size[0] / self.size[1] - h) / 2))
                img = F.pad(img, pad)
                target['boxes'][:, (0, 2)] = target['boxes'][:, (0, 2)] + pad[0]
                target['boxes'][:, (1, 3)] = target['boxes'][:, (1, 3)] + pad[1]

            w_ratio, h_ratio = self.size[1] / img.size[0], self.size[0] / img.size[1]

        img = F.resize(img, self.size)
        target['boxes'][:, (0, 2)] = (target['boxes'][:, (0, 2)] * w_ratio).long().float()
        target['boxes'][:, (1, 3)] = (target['boxes'][:, (1, 3)] * h_ratio).long().float()
        return img, target


class VOCTransformRandomScale(object):
    def __init__(self, scale = (0.8, 1.2)):
        if isinstance(scale, (int, float)):
            scale = (scale, scale)
        assert (isinstance(scale, (list, tuple)) and len(scale) == 2)
        if scale[0] > scale[1]:
            warnings.warn("range should be of kind (min, max)")
        self.scale = scale

    def __call__(self, img, target):
        r_scale = random.uniform(*self.scale)
        img = F.resize(img, (int(img.size[1] * r_scale), int(img.size[0] * r_scale)))
        target['boxes'] = (target['boxes'] * r_scale).long().float()
        return img, target


class VOCTransformExpand(object):
    def __init__(self, ratio, prob = 0.5):
        self.ratio = ratio
        self.p = prob

    @staticmethod
    def get_params(img_size, output_size):
        """Get parameters for ``expand`` for a random expand.

        Args:
            img_size (tuple): Image size (h, w).
            output_size (tuple): Expected output size of the expend.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        h, w = img_size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0
        i = random.randint(0, th - h)
        j = random.randint(0, tw - w)
        return i, j

    def __call__(self, img, target):
        w, h = img.size
        if random.random() < self.p:
            if self.ratio < 1:
                img_h, img_w = int(h * self.ratio), int(w * self.ratio)
                expand_h, expand_w = h, w
                img = F.resize(img, (img_h, img_w))
                target['boxes'] = (target['boxes'] * self.ratio).long().float()
            else:
                img_h, img_w = h, w
                expand_h, expand_w = int(h * self.ratio), int(w * self.ratio)
            i, j = self.get_params((img_h, img_w), (expand_h, expand_w))
            img = F.pad(img, (j, i, expand_w - img_w - j, expand_h - img_h - i))
            target['boxes'][:, (0, 2)] = target['boxes'][:, (0, 2)] + j
            target['boxes'][:, (1, 3)] = target['boxes'][:, (1, 3)] + i

        return img, target


class VOCTransformRandomExpand(VOCTransformExpand):
    def __init__(self, ratio = (0.8, 1.2)):
        assert isinstance(ratio, (float, int)) or (isinstance(ratio, (list, tuple)) and len(ratio) == 2)
        if isinstance(ratio, (list, tuple)):
            if ratio[0] > ratio[1]:
                warnings.warn("range should be of kind (min, max)")
            ratio = random.uniform(*ratio)
        super().__init__(ratio, prob = 1)


class VOCTransformRandomCrop(object):
    def __init__(self, size, padding = None, pad_if_needed = True, fill = 0, padding_mode = 'constant'):
        """Crop the given PIL Image at a random location.

        Args:
            size (sequence or int): Desired output size of the crop. If size is an int instead of sequence like
                (h, w), a square crop (size, size) is made.
            padding (int or sequence, optional): Optional padding on each border of the image. Default is None,
                i.e no padding. If a sequence of length 4 is provided, it is used to pad left, top, right, bottom
                borders respectively. If a sequence of length 2 is provided, it is used to pad left/right, top/bottom
                borders, respectively.
            pad_if_needed (boolean): It will pad the image if smaller than the desired size to avoid raising an
                exception. Since cropping is done after padding, the padding seems to be done at a random offset.
            fill: Pixel fill value for constant fill. Default is 0. If a tuple of length 3, it is used to fill R, G, B
                channels respectively. This value is only used when the padding_mode is constant
            padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.

                 - constant: pads with a constant value, this value is specified with fill
                 - edge: pads with the last value on the edge of the image
                 - reflect: pads with reflection of image (without repeating the last value on the edge)
                    padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                    will result in [3, 2, 1, 2, 3, 4, 3, 2]
                 - symmetric: pads with reflection of image (repeating the last value on the edge)
                    padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                    will result in [2, 1, 1, 2, 3, 4, 4, 3]
        """
        if isinstance(size, (int, float)):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, target):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)
            if isinstance(self.padding, (int, float)):
                target['boxes'] = target['boxes'] + int(self.padding)
            elif isinstance(self.padding, (list, tuple)) and len(self.padding) >= 2:
                target['boxes'][:, (0, 2)] = target['boxes'][:, (0, 2)] + self.padding[0]
                target['boxes'][:, (1, 3)] = target['boxes'][:, (1, 3)] + self.padding[1]

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            target['boxes'][:, (0, 2)] = target['boxes'][:, (0, 2)] + (self.size[1] - img.size[0])
            img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            target['boxes'][:, (1, 3)] = target['boxes'][:, (1, 3)] + (self.size[0] - img.size[1])
            img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)

        center_x = target['boxes'][:, (0, 2)].mean(dim = -1)
        center_y = target['boxes'][:, (1, 3)].mean(dim = -1)

        for it in range(20):
            if it < 10:
                i, j, h, w = self.get_params(img, self.size)
            else:
                sel_idx = random.randint(0, center_x.shape[0] - 1)
                x, y = center_x[sel_idx].item(), center_y[sel_idx].item()
                i = random.randint(max(0, int(y) - self.size[0]), min(int(y), img.size[1] - self.size[0]))
                j = random.randint(max(0, int(x) - self.size[1]), min(int(x), img.size[0] - self.size[1]))
                h, w = self.size

            remain_obj_idx = (center_x > j) * (center_x < j + w) * (center_y > i) * (center_y < i + h)
            if remain_obj_idx.sum() > 0:
                img = F.crop(img, i, j, h, w)
                target['boxes'][:, (0, 2)] = (target['boxes'][:, (0, 2)] - j).clamp(min = 0, max = w)
                target['boxes'][:, (1, 3)] = (target['boxes'][:, (1, 3)] - i).clamp(min = 0, max = h)
                center_x = target['boxes'][:, (0, 2)].mean(dim = -1)
                center_y = target['boxes'][:, (1, 3)].mean(dim = -1)
                obj_idx = (center_x > 0) * (center_x < w) * (center_y > 0) * (center_y < h)
                target['difficult'][~remain_obj_idx] = 1
                target['boxes'] = target['boxes'][obj_idx]
                target['labels'] = target['labels'][obj_idx]
                target['difficult'] = target['difficult'][obj_idx]

                return img, target

        w_ratio, h_ratio = self.size[1] / img.size[0], self.size[0] / img.size[1]
        img = F.resize(img, self.size)
        target['boxes'][:, (0, 2)] = (target['boxes'][:, (0, 2)] * w_ratio).long().float()
        target['boxes'][:, (1, 3)] = (target['boxes'][:, (1, 3)] * h_ratio).long().float()

        return img, target


class VOCTargetTransform(object):

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
        target = dict(boxes = torch.tensor(coords).float(),
                      labels = torch.tensor(labels).long(),
                      difficult = torch.tensor(diff).long())
        return target


class VOCTarget(tuple):
    def to(self, *args, **kwargs):
        return tuple({'boxes': t['boxes'].to(*args, **kwargs),
                      'labels': t['labels'].to(*args, **kwargs),
                      'difficult': t['difficult'].to(*args, **kwargs)} for t in self)


class VOCDetection(torchvision.datasets.VOCDetection):
    def __init__(self,
                 root,
                 year = '2012',
                 image_set = 'train',
                 download = False,
                 transform = None,
                 target_transform = None,
                 transforms = None):
        super(torchvision.datasets.VOCDetection, self).__init__(root, transforms, transform, target_transform)
        DATASET_YEAR_DICT = {'2012': {'base_dir': 'VOCdevkit/VOC2012', 'filename': 'VOCtrainval_11-May-2012.tar'},
                             '2007': {'base_dir': 'VOCdevkit/VOC2007', 'filename': 'VOCtrainval_06-Nov-2007.tar'}}
        self.year = year
        # self.url = DATASET_YEAR_DICT[year]['url']
        # self.filename = DATASET_YEAR_DICT[year]['filename']
        # self.md5 = DATASET_YEAR_DICT[year]['md5']
        # self.image_set = verify_str_arg(image_set, "image_set", ("train", "trainval", "val", "test"))
        valid_values = ("train", "trainval", "val", "test")
        if image_set not in valid_values:
            raise ValueError(f"Unknown value '{image_set}' for argument 'image_set'. Valid values are {valid_values}.")
        self.image_set = image_set

        base_dir = DATASET_YEAR_DICT[year]['base_dir']
        voc_root = os.path.join(self.root, base_dir)
        image_dir = os.path.join(voc_root, 'JPEGImages')
        annotation_dir = os.path.join(voc_root, 'Annotations')

        # if download:
        #     download_extract(self.url, self.root, self.filename, self.md5)

        if not os.path.isdir(voc_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        splits_dir = os.path.join(voc_root, 'ImageSets/Main')

        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.annotations = [os.path.join(annotation_dir, x + ".xml") for x in file_names]
        assert (len(self.images) == len(self.annotations))


def voc_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        return torch.stack(batch, 0, out = out)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype = torch.float)
    elif isinstance(elem, dict):
        return VOCTarget(batch)
    elif isinstance(elem, (tuple, list)):  # namedtuple
        return elem_type((voc_collate(samples) for samples in zip(*batch)))

    default_collate_err_msg_format = ("default_collate: batch must contain tensors, numpy arrays, numbers, "
                                      "dicts or lists; found {}")

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def get_data_loader(transforms, config, train_mode = True):
    assert config.dataset in ['voc2007', 'voc2012']
    if config.dataset == "voc2007":
        year = '2007'
        if train_mode:
            root = os.path.join(config.data_path, 'voctrainval_06-nov-2007')
            image_set = 'trainval'
        else:
            root = os.path.join(config.data_path, 'voctest_06-nov-2007')
            image_set = 'test'

    elif config.dataset == "voc2012":
        year = '2012'
        if train_mode:
            root = os.path.join(config.data_path, 'voctrainval_11-may-2012')
            image_set = 'trainval'
        else:
            root = os.path.join(config.data_path, 'voctest_11-may-2012')
            image_set = 'test'

    dataset = VOCDetection(root = root, year = year, image_set = image_set, transforms = transforms)
    data_loader = DataLoader(dataset, batch_size = config.batch_size, shuffle = train_mode,
                             num_workers = config.workers, collate_fn = voc_collate)

    return data_loader


def calculate_pr(pred_boxes, pred_scores, gt_boxes, gt_difficult, score_range = tuple(i / 10 for i in range(10)),
                 iou_thresh = 0.5):
    """
    calculate all p-r pairs among different score_thresh for one class of one image.

    Args:
        pred_boxes:
        pred_scores:
        gt_boxes:
        gt_difficult:
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
    # 对于不同score阈值，计算相应的 p-r 值
    for s in score_range:
        pb = pred_boxes[pred_scores > s]
        # 若在该score阈值下无对应的boxes，则 p-r 都为0
        if pb.numel() == 0:
            recall.append(0)
            precision.append(0)
            continue
        # 否则计算所有预测框与gt之间的iou
        ious = pb.new_zeros((len(gt_boxes), len(pb)))
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
        # 每个预测框的最大iou所对应的gt记为其匹配的gt
        max_ious, max_ious_idx = ious.max(dim = 0)

        not_difficult = gt_difficult == 0
        if not_difficult.sum() == 0:
            continue
        # 保留 max_iou 中属于 非difficult 目标的预测框，即应该去掉与 difficult gt 相匹配的预测框，不参与p-r计算
        # 如果去掉与 difficult gt 对应的iou分数后，候选框的最大iou依然没有发生改变，则可认为此候选框不与difficult gt相匹配，应该保留
        pb_mask = (ious[not_difficult].max(dim = 0)[0] == max_ious)
        max_ious, max_ious_idx = max_ious[pb_mask], max_ious_idx[pb_mask]
        if max_ious_idx.numel() == 0:
            recall.append(0)
            precision.append(0)
            continue

        tp = max_ious_idx[max_ious > iou_thresh].unique().numel()
        recall.append(tp / not_difficult.sum())
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
