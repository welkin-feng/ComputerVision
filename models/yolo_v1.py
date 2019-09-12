#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  yolo_v1.py

"""

__author__ = 'Welkin'
__date__ = '2019/8/26 16:46'

import torch
import torch.nn as nn
import torch.nn.functional as F
from .util_modules import conv_bn_activation
from . import detection_utils

__all__ = ['yolo_v1']


class YOLOv1Backbone24(nn.Module):
    """  """

    def __init__(self, num_classes, num_grid, num_bbox, in_size):
        """ Constructor for YOLO_backbone """
        super().__init__()
        self.num_classes = num_classes
        self.num_grid = num_grid
        self.num_bbox = num_bbox
        self.in_size = in_size
        self._init_model()

    def _init_model(self):
        LReLU = nn.LeakyReLU
        out_features = self.num_grid ** 2 * (self.num_classes + 5 * self.num_bbox)
        self.conv1 = nn.Sequential(conv_bn_activation(3, 64, 7, 2, 3, activation = LReLU(0.1, inplace = True)),
                                   nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(conv_bn_activation(64, 192, 3, 1, 1, activation = LReLU(0.1, inplace = True)),
                                   nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(conv_bn_activation(192, 128, 1, activation = LReLU(0.1, inplace = True)),
                                   conv_bn_activation(128, 256, 3, 1, 1, activation = LReLU(0.1, inplace = True)),
                                   conv_bn_activation(256, 256, 1, activation = LReLU(0.1, inplace = True)),
                                   conv_bn_activation(256, 512, 3, 1, 1, activation = LReLU(0.1, inplace = True)),
                                   nn.MaxPool2d(2))
        conv4_list = []
        for i in range(4):
            conv4_list += [conv_bn_activation(512, 256, 1, activation = LReLU(0.1, inplace = True)),
                           conv_bn_activation(256, 512, 3, 1, 1, activation = LReLU(0.1, inplace = True))]
        self.conv4 = nn.Sequential(*conv4_list,
                                   conv_bn_activation(512, 1024, 3, 1, 1, activation = LReLU(0.1, inplace = True)),
                                   nn.MaxPool2d(2))
        del conv4_list
        conv5_list = []
        for i in range(2):
            conv5_list += [conv_bn_activation(1024, 512, 1, activation = LReLU(0.1, inplace = True)),
                           conv_bn_activation(512, 1024, 3, 1, 1, activation = LReLU(0.1, inplace = True))]
        self.conv5 = nn.Sequential(*conv5_list,
                                   conv_bn_activation(1024, 1024, 3, 1, 1, activation = LReLU(0.1, inplace = True)),
                                   conv_bn_activation(1024, 1024, 3, 2, 1, activation = LReLU(0.1, inplace = True)))
        del conv5_list
        self.conv6 = nn.Sequential(conv_bn_activation(1024, 1024, 3, 1, 1, activation = LReLU(0.1, inplace = True)),
                                   conv_bn_activation(1024, 1024, 3, 1, 1, activation = LReLU(0.1, inplace = True)))

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)
        return output


class YOLOv1(nn.Module):
    """  """

    def __init__(self, num_classes, num_grid, num_bbox, in_size = 448, threshold = 0.7, backbone = YOLOv1Backbone24):
        """ Constructor for YOLOv1 """
        super().__init__()
        self.num_classes = num_classes
        self.num_grid = num_grid
        self.num_bbox = num_bbox
        self.in_size = in_size
        self.threshold = threshold
        self._init_model(backbone)
        self._initialize_weights()

    def _init_model(self, backbone):
        final_size = self.in_size // 64
        out_features = self.num_grid ** 2 * (self.num_classes + 5 * self.num_bbox)
        self.backbone_net = backbone(self.num_classes, self.num_grid, self.num_bbox, self.in_size)
        self.fc = nn.Sequential(nn.Flatten(),
                                nn.Linear(1024 * final_size ** 2, 4096),
                                nn.Dropout(),
                                nn.ReLU(inplace = True),
                                nn.Linear(4096, out_features))
        self.postprocess = YOLOv1Postprocess(self.num_classes, self.num_grid, self.num_bbox,
                                             (self.in_size, self.in_size), self.threshold)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        output = self.backbone_net(x)
        output = self.fc(output)
        all_boxes, all_scores, all_masks = self.postprocess(output)
        return all_boxes, all_scores, all_masks


class YOLOv1Postprocess(nn.Module):
    """  """

    def __init__(self, num_classes, num_grid, num_bbox, image_shapes, num_threshold):
        """ Constructor for YOLOv1Postprocess """
        super().__init__()
        self.num_classes = num_classes
        self.num_grid = num_grid
        self.num_bbox = num_bbox
        self.image_shapes = image_shapes
        self.num_threshold = num_threshold

    def boxes_decode(self, yolo_bbox, image_shape):
        """

        Args:
            yolo_bbox:  (Tensor[num_grid**2, 5])
            image_shape: Tuple(height, weight)

        Returns:
            transformed_boxes: (Tensor[N, num_grid**2, 5])

        """
        dtype = yolo_bbox.dtype
        device = yolo_bbox.device
        H, W = image_shape
        grid_idx = torch.arange(self.num_grid, dtype = dtype, device = device).unsqueeze(1)
        x_grid = grid_idx.repeat(self.num_grid, 1).flatten()
        y_grid = grid_idx.repeat(1, self.num_grid).flatten()
        x0 = ((yolo_bbox[:, 0] + x_grid) / self.num_grid - yolo_bbox[:, 2] / 2).unsqueeze(1) * W
        y0 = ((yolo_bbox[:, 1] + y_grid) / self.num_grid - yolo_bbox[:, 3] / 2).unsqueeze(1) * H
        x1 = ((yolo_bbox[:, 0] + x_grid) / self.num_grid + yolo_bbox[:, 2] / 2).unsqueeze(1) * W
        y1 = ((yolo_bbox[:, 1] + y_grid) / self.num_grid + yolo_bbox[:, 3] / 2).unsqueeze(1) * H

        return torch.cat((x0, y0, x1, y1, yolo_bbox[:, 4].unsqueeze(1)), dim = 1)

    def postprocess_detections(self, class_logits, proposals, image_shapes):
        """

        Args:
            class_logits: (Tensor[N, num_grid**2, classes]),
            proposals: (Tensor[N, num_grid**2, 5])
            image_shapes: (Tuple[height, width]): size of the image

        Returns:

        """
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        pred_probs = F.softmax(class_logits, -1)  # Tensor[N, num_grid**2, classes]

        # boxes_per_image = [boxes_in_image.sum() for boxes_in_image in proposals_mask]

        all_masks = []
        all_boxes = []

        image_shape = image_shapes
        for pred_boxes, probs in zip(proposals, pred_probs):
            # shape Tensor[num_grid**2, (x0, y0, x1, y1, score)], (x,y): range (0, W) and (0, H)
            boxes = self.boxes_decode(pred_boxes, image_shape)
            # remove boxes which do not have object (remove low scoring boxes)
            reserve_mask = boxes[:, -1] > 0.1
            boxes = boxes[reserve_mask]
            probs = probs[reserve_mask]
            boxes, scores = boxes[:, :4], boxes[:, -1]
            boxes = detection_utils.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.argmax(probs, dim = -1)

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.flatten()
            labels = labels.flatten()

            # non-maximum suppression, independently done per class
            keep_mask = detection_utils.batched_nms(boxes, scores, labels, self.num_threshold)

            boxes, scores = boxes[keep_mask], scores[keep_mask]

            mask_dict = {
                'reserve_mask': reserve_mask,
                'keep_mask': keep_mask,
            }
            all_boxes.append(boxes)
            all_masks.append(mask_dict)

        return all_boxes, all_masks

    def forward(self, x):
        """

        Args:
            x: Tensor(N, out_features)
                out_features = num_grid**2 * (num_classes + num_bbox * 5)

        Returns:
            all_boxes: List(Tensor)
            all_scores: List(Tensor)
            all_masks: List(Dict(Tensor))

        """
        N = x.shape[0]
        x = x.view(N, -1, self.num_classes + 5 * self.num_bbox)  # Tensor(N, num_grid**2, num_classes + num_bbox * 5)
        # classes shape: [N, num_grid**2, num_classes]
        class_logits = x[..., :self.num_classes]
        # bbox shape: [N * num_grid**2, num_bbox, (x, y, w, h, score)]
        yolo_bboxes = x[..., self.num_classes:].view(-1, self.num_bbox, 5)
        # bbox shape: [N, num_grid**2, 5]
        yolo_bboxes = yolo_bboxes[range(N * self.num_grid ** 2), yolo_bboxes[..., -1].max(-1)[1]].reshape(N, -1, 5)
        pred_boxes, pred_masks = self.postprocess_detections(class_logits, yolo_bboxes, self.image_shapes)

        return class_logits, yolo_bboxes, pred_boxes, pred_masks


def yolo_v1(num_classes, in_size = 448):
    return YOLOv1(num_classes, 7, 2, in_size)


if __name__ == '__main__':
    # test
    import sys

    print(sys.modules[__name__])
    fn_list = ['yolo_v1']
    for fn in fn_list:
        f = getattr(sys.modules[__name__], fn)
        model = f(10)
        print(' ---', fn, '---')
        for k, v in model.state_dict().items():
            print(k)
        print()
