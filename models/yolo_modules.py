#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  yolo_modules.py

"""

__author__ = 'Welkin'
__date__ = '2019/9/5 13:48'

import torch
import random
import math
from .detection_utils import ImageList


class GeneralizedYOLO(torch.nn.Module):

    def __init__(self, backbone, reg_net, postprocess, transform):
        """
        Args:
            backbone (torch.nn.Module): should have attribute `out_channels`
            reg_net (torch.nn.Module):
            postprocess:
        """

        super().__init__()
        self.transform = transform
        self.backbone = backbone
        self.reg_net = reg_net
        self.postprocess = postprocess

    def forward(self, images, targets = None, **kwargs):
        """
        Args:
            images (Tensor[N, C, H, W] or list[Tensor[C, H, W]]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
            kwargs:
                get_prior_anchor_loss: if use `prior_anchor_loss`, input `get_prior_anchor_loss = True`

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels`.
        """
        if self.training:
            if targets is None:
                raise ValueError("In training mode, targets should be passed")
            else:
                assert len(targets) == len(images), "the length of `images` do not match the length of `targets`"

        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        # boxes (Tensor): original boxes without filtering, [N, 7*7*30] in YOLOv1 and [N, 125, 13, 13] in YOLOv2
        proposed_boxes_bias = self.reg_net(features)
        #
        detections, detector_losses = self.postprocess(proposed_boxes_bias, images.image_sizes, targets, **kwargs)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        for detection in detections:
            detection['boxes'] = detection['boxes'].long()
            detection['scores'] = detection['scores'].view(-1)

        return detections, detector_losses


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
        if isinstance(images, torch.Tensor) and images.dim() != 4:
            raise ValueError("images is expected to be a 4d Tensor of shape [N, C, H, W] or "
                             "a list of 3d tensors of shape [C, H, W], got {}".format(images.shape))
        images = [img for img in images]
        targets = [t for t in targets] if targets is not None else targets
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
        target = dict(boxes = bbox, labels = target['labels'])

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

    def postprocess(self, result, image_sizes, original_image_sizes):
        if self.training:
            return result
        for i, (pred, im_s, o_im_s) in enumerate(zip(result, image_sizes, original_image_sizes)):
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
