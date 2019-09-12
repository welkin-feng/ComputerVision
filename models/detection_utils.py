#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  detection_utils.py

"""

__author__ = 'Welkin'
__date__ = '2019/8/28 14:16'

import torch
import random
import math


class GeneralizedYOLOTransform(torch.nn.Module):
    """
    Performs input / target transformation before feeding the data to a GeneralizedYOLO
    model.

    The transformations it perform are:
        - input normalization (mean subtraction and std division)
        - input / target resizing to match min_size / max_size

    It returns a ImageList for the inputs, and a List[Dict[Tensor]] for the targets
    """

    def __init__(self, min_size, max_size, image_mean, image_std, downsample_factor = 32):
        super().__init__()
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.size_divisible = downsample_factor

    def forward(self, images, targets = None):
        if isinstance(images, torch.Tensor):
            if images.dim() == 4:
                images = list(images.unbind(0))
            else:
                raise ValueError("images is expected to be a 4d Tensor of shape [N, C, H, W] or "
                                 "a list of 3d tensors of shape [C, H, W], "
                                 "got {}".format(images.shape))

        for i in range(len(images)):
            image = images[i]
            target = targets[i] if targets is not None else targets
            if image.dim() != 3:
                raise ValueError("images is expected to be a list of 3d tensors "
                                 "of shape [C, H, W], got {}".format(image.shape))
            image = self.normalize(image)
            image, target = self.resize(image, target)
            images[i] = image
            if targets is not None:
                targets[i] = target
        image_sizes = [img.shape[-2:] for img in images]
        images = self.batch_images(images)
        image_list = ImageList(images, image_sizes)
        return image_list, targets

    def normalize(self, image):
        dtype, device = image.dtype, image.device
        mean = torch.as_tensor(self.image_mean, dtype = dtype, device = device)
        std = torch.as_tensor(self.image_std, dtype = dtype, device = device)
        return (image - mean[:, None, None]) / std[:, None, None]

    def resize(self, image, target):
        h, w = image.shape[-2:]
        min_size = float(min(image.shape[-2:]))
        max_size = float(max(image.shape[-2:]))
        if self.training:
            size = random.choice(self.min_size)
        else:
            # FIXME assume for now that testing uses the largest scale
            size = self.min_size[-1]

        if min_size > size and max_size < self.max_size:
            return image, target

        scale_factor = size / min_size
        if max_size * scale_factor > self.max_size:
            scale_factor = self.max_size / max_size
        image = torch.nn.functional.interpolate(
            image[None], scale_factor = scale_factor, mode = 'bilinear', align_corners = False)[0]

        if target is None:
            return image, target

        bbox = target["boxes"]
        bbox = self.resize_boxes(bbox, (h, w), image.shape[-2:])
        target["boxes"] = bbox

        return image, target

    def batch_images(self, images):
        # 将所有img添加0原始变成相同size，以便于使用Tensor表示
        # concatenate
        max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))

        stride = self.size_divisible
        max_size = list(max_size)
        max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
        max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
        max_size = tuple(max_size)

        batch_shape = (len(images),) + max_size
        batched_imgs = images[0].new(*batch_shape).zero_()
        for img, pad_img in zip(images, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        return batched_imgs

    def postprocess(self, result, image_shapes, original_image_sizes):
        if self.training:
            return result
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_shapes, original_image_sizes)):
            boxes = pred["boxes"]
            boxes = self.resize_boxes(boxes, im_s, o_im_s)
            result[i]["boxes"] = boxes

        return result

    @staticmethod
    def resize_boxes(boxes, original_size, new_size):
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(new_size, original_size))
        ratio_height, ratio_width = ratios
        xmin, ymin, xmax, ymax = boxes.unbind(1)
        xmin = xmin * ratio_width
        xmax = xmax * ratio_width
        ymin = ymin * ratio_height
        ymax = ymax * ratio_height
        return torch.stack((xmin, ymin, xmax, ymax), dim = 1)


class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """

    def __init__(self, tensors, image_sizes):
        """
        Arguments:
            tensors (tensor)
            image_sizes (list[tuple[int, int]])
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
        size (Tuple[width, height]): size of the image

    Returns:
        clipped_boxes (Tensor[N, 4])
    """
    width, height = size
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
