#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  detection_utils.py

"""

__author__ = 'Welkin'
__date__ = '2019/8/28 14:16'

import torch


class ImageList(object):
    """
    Structure that holds a list of images (of possibly varying sizes) as a single tensor.
    This works by padding the images to the same size, and storing in a field the original sizes of each image
    """

    def __init__(self, tensors, image_sizes):
        """
        Arguments:
            tensors (tensor)
            image_sizes (list[tuple[height, width]])
        """
        self.tensors = tensors
        self.image_sizes = image_sizes

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        return ImageList(cast_tensor, self.image_sizes)


def clip_boxes_to_image(boxes, size):
    """
    Clip boxes so that they lie inside an image of size `size`.

    Arguments:
        boxes (Tensor[N, 4]): boxes in [x0, y0, x1, y1] format
        size (Tuple[height, width]): size of the image

    Returns:
        clipped_boxes (Tensor[N, 4])
    """
    height, width = size
    boxes[..., 0::2] = boxes[..., 0::2].clamp(min = 0, max = width)
    boxes[..., 1::2] = boxes[..., 1::2].clamp(min = 0, max = height)
    return boxes


def nms(bboxes, scores, threshold = 0.5):
    """
        Performs non-maximum suppression (NMS) on the boxes according
        to their intersection-over-union (IoU).

        NMS iteratively removes lower scoring boxes which have an
        IoU greater than iou_threshold with another (higher scoring)
        box.

        Arguments:
            bboxes (Tensor[N, 4]): boxes to perform NMS on
            scores (Tensor[N]): scores for each one of the boxes
            threshold (float): discards all overlapping boxes with IoU < iou_threshold

        Returns:
            keep (Tensor): int64 tensor with the indices of the elements that have been kept
                by NMS, sorted in decreasing order of scores
    """
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)  # [N,] 每个bbox的面积
    _, order = scores.sort(0, descending = True)  # 降序排列

    keep = []
    while order.numel() > 0:  # torch.numel()返回张量元素个数
        if order.numel() == 1:  # 保留框只剩一个
            i = order.item()
            keep.append(i)
            break
        else:
            i = order[0].item()  # 保留scores最大的那个框box[i]
            keep.append(i)

        # 计算box[i]与其余各框的IOU(思路很好)
        xx1 = x1[order[1:]].clamp(min = x1[i].item())  # [N-1,]
        yy1 = y1[order[1:]].clamp(min = y1[i].item())
        xx2 = x2[order[1:]].clamp(max = x2[i].item())
        yy2 = y2[order[1:]].clamp(max = y2[i].item())
        inter = (xx2 - xx1).clamp(min = 0) * (yy2 - yy1).clamp(min = 0)  # [N-1,]

        iou = inter / (areas[i] + areas[order[1:]] - inter)  # [N-1,]
        idx = (iou <= threshold).nonzero().squeeze()  # 注意此时idx为[N-1,] 而order为[N,]
        if idx.numel() == 0:
            break
        order = order[idx + 1]  # 修补索引之间的差值

    return torch.tensor(keep).long()  # Pytorch的索引值为LongTensor


def batched_nms(boxes, scores, labels, nms_threshold):
    """
    Performs non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Arguments:
        boxes (Tensor[N, 4]): boxes where NMS will be performed
        scores (Tensor[N]): scores for each one of the boxes
        labels (Tensor[N]): indices of the categories for each one of the boxes.
        nms_threshold (float): discards all overlapping boxes with IoU < nms_threshold

    Returns:
        keep (Tensor): int64 tensor with the indices of
            the elements that have been kept by NMS, sorted
            in decreasing order of scores
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype = torch.int64, device = boxes.device)
    # strategy: in order to perform NMS independently per class.
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    scores = scores.view(-1)
    labels = labels.view(-1)
    max_coordinate = boxes.max()
    offsets = labels.to(boxes) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, nms_threshold)
    return keep
