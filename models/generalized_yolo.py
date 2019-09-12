#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  generalized_yolo.py

"""

__author__ = 'Welkin'
__date__ = '2019/9/5 13:48'

import torch


class GeneralizedYOLO(torch.nn.Module):
    """  """

    def __init__(self, backbone, reg_net, postprocess, transform):
        """
        Constructor for GeneralizedYOLO

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

    def forward(self, images, targets = None):
        """

        Args:
            images (Tensor[N, C, H, W] or list[Tensor[C, H, W]]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

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
        detections, detector_losses = self.postprocess(proposed_boxes_bias, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        if self.training:
            return detector_losses

        for detection in detections:
            detection['boxes'] = detection['boxes'].long()
            detection['scores'] = detection['scores'].view(-1)

        return detections
