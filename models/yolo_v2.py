#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Project Name:   ComputerVision 

File Name:  yolo_v2.py

"""

__author__ = 'Welkin'
__date__ = '2019/9/3 13:02'

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from . import detection_utils
from .util_modules import conv_bn_activation, conv_bn_relu
from .yolo_modules import GeneralizedYOLO
from .yolo_modules import GeneralizedYOLOTransform

__all__ = ['Darknet19', 'YOLOv2', 'yolo_v2_darknet19', 'yolo_v2_resnet50', 'yolo_v2_resnet50_backbone']


class Darknet19(nn.Module):
    out_channels = 3072
    downsample_factor = 32

    def __init__(self):
        super().__init__()
        LReLU = nn.LeakyReLU

        self.conv1 = conv_bn_activation(3, 32, 3, 1, 1, activation = LReLU(0.1, inplace = True))
        self.layer1 = nn.Sequential(
            nn.MaxPool2d(2),
            conv_bn_activation(32, 64, 3, 1, 1, activation = LReLU(0.1, inplace = True)),
        )
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(2),
            conv_bn_activation(64, 128, 3, 1, 1, activation = LReLU(0.1, inplace = True)),
            conv_bn_activation(128, 64, 1, activation = LReLU(0.1, inplace = True)),
            conv_bn_activation(64, 128, 3, 1, 1, activation = LReLU(0.1, inplace = True)),
        )
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(2),
            conv_bn_activation(128, 256, 3, 1, 1, activation = LReLU(0.1, inplace = True)),
            conv_bn_activation(256, 128, 1, activation = LReLU(0.1, inplace = True)),
            conv_bn_activation(128, 256, 3, 1, 1, activation = LReLU(0.1, inplace = True)),
        )
        self.layer4 = nn.Sequential(
            nn.MaxPool2d(2),
            conv_bn_activation(256, 512, 3, 1, 1, activation = LReLU(0.1, inplace = True)),
            conv_bn_activation(512, 256, 1, activation = LReLU(0.1, inplace = True)),
            conv_bn_activation(256, 512, 3, 1, 1, activation = LReLU(0.1, inplace = True)),
            conv_bn_activation(512, 256, 1, activation = LReLU(0.1, inplace = True)),
            conv_bn_activation(256, 512, 3, 1, 1, activation = LReLU(0.1, inplace = True)),
        )
        self.layer5 = nn.Sequential(
            nn.MaxPool2d(2),
            conv_bn_activation(512, 1024, 3, 1, 1, activation = LReLU(0.1, inplace = True)),
            conv_bn_activation(1024, 512, 1, activation = LReLU(0.1, inplace = True)),
            conv_bn_activation(512, 1024, 3, 1, 1, activation = LReLU(0.1, inplace = True)),
            conv_bn_activation(1024, 512, 1, activation = LReLU(0.1, inplace = True)),
            conv_bn_activation(512, 1024, 3, 1, 1, activation = LReLU(0.1, inplace = True)),
        )
        self.passthrough_layer = PassthroughLayer()

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        feature4 = self.layer4(out)  # [N, 512, 26, 26]
        feature5 = self.layer5(feature4)  # [N, 1024, 13, 13]
        out = self.passthrough_layer(high_res_feature = feature4, low_res_feature = feature5)
        return out


class PassthroughLayer(nn.Module):
    def __init__(self, backbone = None):
        super().__init__()
        self.backbone = backbone

    def forward(self, images = None, high_res_feature = None, low_res_feature = None):
        if high_res_feature is not None and low_res_feature is not None:
            return torch.cat((low_res_feature, high_res_feature[..., 0::2, 0::2], high_res_feature[..., 0::2, 1::2],
                              high_res_feature[..., 1::2, 0::2], high_res_feature[..., 1::2, 1::2]), dim = 1)

        if self.backbone is None:
            raise ValueError("`high_res_feature` and `low_res_feature` should not be None")
        if images is None:
            raise ValueError("`images` should not be None")

        high_res_feature, low_res_feature = self.backbone(images).values()
        return torch.cat((low_res_feature, high_res_feature[..., 0::2, 0::2], high_res_feature[..., 0::2, 1::2],
                          high_res_feature[..., 1::2, 0::2], high_res_feature[..., 1::2, 1::2]), dim = 1)


class YOLOv2(GeneralizedYOLO):
    """
        Backbone: Darknet-19
        Classification: cls layer with 1x1x1000 conv and avgpool
        Detection: Generate 5 Anchor Boxes at each point of feature map for predicting bounding boxes.
    """

    def __init__(self, backbone,
                 # Box parameters
                 num_classes, anchor_boxes = None,
                 # for testing
                 box_iou_thresh = 0.6, nms_threshold = 0.5,
                 # transform parameters
                 min_size = (288,), max_size = 608,
                 image_mean = None, image_std = None,
                 use_transform = True, **kwargs):
        """
        Args:
            backbone:
            num_classes:
            anchor_boxes:
            box_iou_thresh: minimum IoU between the anchor and the GT box so that they can be
                considered as positive.
            nms_threshold:
            min_size
            max_size
            image_mean
            image_std
        """
        if not hasattr(backbone, "out_channels"):
            raise ValueError(
                "backbone should contain an attribute out_channels "
                "specifying the number of output channels (assumed to be the "
                "same for all the levels)")
        if not hasattr(backbone, "downsample_factor"):
            raise ValueError(
                "backbone should contain an attribute downsample_factor "
                "the conv layers downsample the image by the factor "
                "(usually be 32).")
        if not isinstance(anchor_boxes, (list, tuple, type(None))):
            raise ValueError('`anchor_boxes` should be a list of (width_pix, height_pix) or None')

        if anchor_boxes is None:
            # anchor_boxes = [(1.19, 1.99), (2.79, 4.60), (4.54, 8.93), (8.06, 5.29), (10.33, 10.65)]
            anchor_boxes = [[38.08, 63.68],  # width, height for anchor 1
                            [89.28, 147.20],  # width, height for anchor 2
                            [145.28, 285.76],  # etc.
                            [257.92, 169.28],
                            [330.56, 340.80]]

        out_channels = (num_classes + 5) * len(anchor_boxes)
        reg_net = nn.Sequential(
            conv_bn_relu(backbone.out_channels, 1024, 3, 1, 1),
            conv_bn_relu(1024, 1024, 3, 1, 1),
            conv_bn_relu(1024, 1024, 3, 1, 1),
            nn.Conv2d(1024, out_channels, 1)
        )

        postprocess = YOLOv2Postprocess(backbone.downsample_factor, num_classes, anchor_boxes, box_iou_thresh,
                                        nms_threshold)

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = GeneralizedYOLOTransform(min_size, max_size, image_mean, image_std)

        super().__init__(backbone, reg_net, postprocess, transform, use_transform)


class YOLOv2Postprocess(nn.Module):

    def __init__(self, downsample_factor, num_classes, anchor_boxes, box_iou_thresh, nms_thresh):
        """
        Args:
            downsample_factor
            num_classes
            anchor_boxes (List[Tuple[float, float]]): the (width, height) size of each anchor box
            box_iou_thresh
            nms_thresh
        """
        super().__init__()
        if not isinstance(anchor_boxes, (list, tuple)):
            raise ValueError("anchor_boxes should be a list")
        self.size_divisible = float(downsample_factor)
        self.num_classes = num_classes
        if not isinstance(anchor_boxes[0], (list, tuple)):
            anchor_boxes = tuple((s, s) for s in anchor_boxes)
        self.anchors = anchor_boxes
        self.box_iou_thresh = box_iou_thresh
        self.nms_thresh = nms_thresh

    def forward(self, boxes_offset, image_sizes, targets = None, get_prior_anchor_loss = False, **kwargs):
        """
        Args:
            boxes_offset (Tensor): shape [N, 7*7*30] in YOLOv1, [N, 13, 13, 125] in YOLOv2
            image_sizes (List[Tuple[height, width]]):
            targets (List[Dict[Tensor]): ground-truth boxes present in the image (optional).
                If provided, each element in the dict should contain a field `boxes`,
                with the locations of the ground-truth boxes, and a field `labels`
                with the classifications of the ground-truth boxes.
            get_prior_anchor_loss

        Returns:
            result (List[Dict[Tensor]]): the predicted boxes from the RPN, one Tensor per image.
            losses (Dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        _start_time = time.time()
        proposed_boxes_classes, proposed_boxes_xywh, proposed_boxes_score, prior_anchor_losses = self.get_proposed_boxes(
            boxes_offset, get_prior_anchor_loss)
        _get_proposed_boxes_time = time.time() - _start_time

        result, losses = [], torch.scalar_tensor(0.)
        if self.training:
            self.check_targets(targets)
            assert len(targets) == len(boxes_offset), "the length of `boxes_offset` don't match the length of `targets`"

            noobject_scale = kwargs.get('noobject_scale', 1)
            class_scale = kwargs.get('class_scale', 1)
            coord_scale = kwargs.get('coord_scale', None)
            object_scale = kwargs.get('object_scale', 5)
            prior_scale = kwargs.get('prior_scale', 0.01)

            losses = self.comupute_loss(proposed_boxes_classes, proposed_boxes_xywh, proposed_boxes_score, targets,
                                        image_sizes, noobject_scale, class_scale, coord_scale, object_scale)
            losses = losses['class_losses'] + losses['coord_losses'] + losses['obj_score_losses'] + \
                     losses['noobj_score_losses'] + prior_scale * prior_anchor_losses

        _comupute_loss_time = time.time() - _start_time - _get_proposed_boxes_time

        # transform xywh to xyxy
        proposed_boxes_xyxy = self.bboxes_transform(proposed_boxes_xywh, image_sizes)
        _bboxes_transform_time = time.time() - _start_time - _get_proposed_boxes_time - _comupute_loss_time
        # filter
        cls_list, loc_list, score_list = self.filter_proposals(proposed_boxes_classes, proposed_boxes_xyxy,
                                                               proposed_boxes_score)
        _filter_proposals_time = time.time() - _start_time - _get_proposed_boxes_time - _comupute_loss_time - _bboxes_transform_time
        # nms
        pred_boxes_label, pred_boxes_loc, pred_boxes_score = self.nms(cls_list, loc_list, score_list)
        num_images = len(pred_boxes_loc)
        for i in range(num_images):
            result.append(
                dict(boxes = pred_boxes_loc[i],
                     scores = pred_boxes_score[i],
                     labels = pred_boxes_label[i])
            )
        _nms_time = time.time() - _start_time - _get_proposed_boxes_time - _comupute_loss_time - _bboxes_transform_time - _filter_proposals_time
        # print(f"  - get_proposed_boxes_time: {_get_proposed_boxes_time:.2f}, "
        #       f"comupute_loss_time: {_comupute_loss_time:.2f}, "
        #       f"bboxes_transform_time: {_bboxes_transform_time:.2f}, "
        #       f"filter_proposals_time: {_filter_proposals_time:.2f}, "
        #       f"nms_time: {_nms_time:.2f}")

        return result, losses

    def get_proposed_boxes(self, boxes_offset, get_prior_anchor_loss):
        N, C, H, W = boxes_offset.shape
        num_anchors = len(self.anchors)
        stride = self.num_classes + 5
        assert C == num_anchors * stride, "the channels of boxes_offset can not match the number of anchors"

        boxes_offset = boxes_offset.permute(0, 2, 3, 1)  # [N, H, W, C]
        boxes_offset = boxes_offset.reshape(N, H, W, num_anchors, stride)  # [N, H, W, 5, 25]

        proposed_boxes_classes = boxes_offset[..., :self.num_classes]  # [N, H, W, 5, 20]
        proposed_boxes_score = torch.sigmoid(boxes_offset[..., -1:])  # [N, H, W, 5, 1]

        boxes_offset_xy = torch.sigmoid(boxes_offset[..., self.num_classes:self.num_classes + 2])  # [N, H, W, 5, 2]
        boxes_offset_wh = boxes_offset[..., self.num_classes + 2:self.num_classes + 4]  # [N, H, W, 5, 2]

        prior_anchor_losses = 0
        if get_prior_anchor_loss:
            prior_anchor_losses = F.mse_loss(boxes_offset_xy, torch.zeros_like(boxes_offset_xy) + 0.5) + \
                                  F.mse_loss(boxes_offset_wh, torch.zeros_like(boxes_offset_wh))

        x_grid = torch.arange(W).repeat(H, 1).to(boxes_offset)
        y_grid = torch.arange(H).unsqueeze(1).repeat(1, W).to(boxes_offset)
        anchors_xy = torch.stack((x_grid, y_grid), dim = -1).unsqueeze(-2).to(boxes_offset)  # [H, W, 1, 2]
        anchors_wh = boxes_offset.new(self.anchors)  # [5, 2]

        # 从特征图中的坐标转换成原图中的坐标
        # convert coordinates from feature map to coordinates in the original image
        proposed_boxes_xy = (anchors_xy + boxes_offset_xy) * self.size_divisible
        proposed_boxes_wh = anchors_wh * boxes_offset_wh.exp()
        proposed_boxes_xywh = torch.cat((proposed_boxes_xy, proposed_boxes_wh), dim = -1)

        return proposed_boxes_classes, proposed_boxes_xywh, proposed_boxes_score, prior_anchor_losses

    def filter_proposals(self, boxes_classes, boxes_location, boxes_score):
        cls, loc, score = [], [], []
        N, n_classes = boxes_classes.shape[0], boxes_classes.shape[-1]
        boxes_classes = boxes_classes.reshape(N, -1, n_classes)
        boxes_location = boxes_location.reshape(N, -1, 4)
        boxes_score = boxes_score.reshape(N, -1, 1)
        for i in range(N):
            if self.training:
                mask = (boxes_score[i] > 0.01).view(-1)
            else:
                mask = (boxes_score[i] > self.box_iou_thresh).view(-1)
            cls.append(boxes_classes[i, mask])
            loc.append(boxes_location[i, mask])
            score.append(boxes_score[i, mask])

        return cls, loc, score

    def nms(self, boxes_cls_list, boxes_loc_list, boxes_score_list):
        label_list, loc, score = [], [], []
        N = len(boxes_cls_list)
        for i in range(N):
            if boxes_cls_list[i].numel() > 0:
                # cls_probs = torch.softmax(boxes_cls_list[i], -1)
                # labels = torch.argmax(cls_probs, dim = -1)
                labels = torch.argmax(boxes_cls_list[i], dim = -1)
                if self.training:
                    label_list.append(labels)
                    loc.append(boxes_loc_list[i])
                    score.append(boxes_score_list[i])
                else:
                    # non-maximum suppression, independently done per class
                    keep_mask = detection_utils.batched_nms(boxes_loc_list[i], boxes_score_list[i], labels,
                                                            self.nms_thresh)
                    label_list.append(labels[keep_mask])
                    loc.append(boxes_loc_list[i][keep_mask])
                    score.append(boxes_score_list[i][keep_mask])
            else:
                label_list.append(boxes_cls_list[i].new([]))
                loc.append(boxes_loc_list[i])
                score.append(boxes_score_list[i])

        return label_list, loc, score

    def comupute_loss(self, proposed_boxes_classes, proposed_boxes_xywh, proposed_boxes_score,
                      targets, image_sizes, noobject_scale = 1, class_scale = 1, coord_scale = None, object_scale = 5):
        """
        for pred_box in all prediction box:
            if (max iou pred_box has with all truth box < threshold):
                costs[pred_box][obj] = (sigmoid(obj)-0)^2 * 1
            else:
                costs[pred_box][obj] = 0
            costs[pred_box][x, y] = (sigmoid(x, y)-0.5)^2 * 0.01
            costs[pred_box][w, h] = ((w-0)^2 + (h-0)^2) * 0.01
        for truth_box all ground truth box:
            pred_box = the one prediction box that is supposed to predict for truth_box
            costs[pred_box][obj] = (1-sigmoid(obj))^2 * 5
            costs[pred_box][x, y] = (sigmoid(x, y)-true(x, y))^2 * (2- truew*trueh/imagew*imageh)
            costs[pred_box][w, h] = ((w-log(truew))^2 + (h-log(trueh))^2) * (2- truew*trueh/imagew*imageh)
            costs[pred_box][classes] = softmax_euclidean
        total_loss = sum(costs)

        Args:
            proposed_boxes_classes
            proposed_boxes_xywh
            proposed_boxes_score
            targets
            image_sizes

        Returns:
            losses (Dict[Tensor]): include `class_losses`, `coord_losses`, `obj_score_losses`,
                `noobj_score_losses` and `prior_anchor_loss`.
        """
        class_losses, coord_losses, obj_score_losses, noobj_score_losses = 0, 0, 0, 0

        # transform xywh to xyxy
        proposed_boxes_xyxy = self.bboxes_transform(proposed_boxes_xywh, image_sizes)

        proposed_boxes_xywh = proposed_boxes_xywh / self.size_divisible

        cal_coord_scale = coord_scale is None

        for i in range(len(targets)):
            t_boxes = targets[i]['boxes']
            t_labels = targets[i]['labels']
            image_size = image_sizes[i]
            p_boxes_cls = proposed_boxes_classes[i]
            p_boxes_xywh = proposed_boxes_xywh[i]
            p_boxes_score = proposed_boxes_score[i]

            # 计算 背景 损失
            noobj_score_loss = self._compute_noobj_loss(proposed_boxes_xyxy[i], p_boxes_score, t_boxes, noobject_scale)

            # target transform to xywh
            t_boxes = self.bboxes_transform_to_xywh(t_boxes)
            # [t_boxes.shape[0], 1]
            if cal_coord_scale:
                coord_scale = 2 - t_boxes[:, 2:].prod(dim = -1, keepdim = True) / (image_size[0] * image_size[1])
            t_boxes = t_boxes / self.size_divisible

            # 得到每个gt落在哪个cell
            t_box_cell_idx = t_boxes[:, :2].long()  # [t_boxes.shape[0], 2]
            # 得到与每个gt匹配的anchor的idx
            t_box_correspond_anchor_idx = self._get_matched_anchor_idx(t_boxes)
            # 取与gt匹配的原始anchor对应的预测框
            correspond_box_idx = torch.cat((t_box_cell_idx, t_box_correspond_anchor_idx), dim = -1).t().tolist()
            # 计算 目标 损失
            class_loss, coord_loss, obj_score_loss = self._compute_obj_loss(t_boxes, t_labels, correspond_box_idx,
                                                                            p_boxes_cls, p_boxes_xywh, p_boxes_score,
                                                                            class_scale, coord_scale, object_scale)
            noobj_score_losses += noobj_score_loss
            class_losses += class_loss
            coord_losses += coord_loss
            obj_score_losses += obj_score_loss

        losses = dict(class_losses = class_losses / len(targets),
                      coord_losses = coord_losses / len(targets),
                      obj_score_losses = obj_score_losses / len(targets),
                      noobj_score_losses = noobj_score_losses / len(targets))
        return losses

    @staticmethod
    def _compute_noobj_loss(p_boxes, p_boxes_score, t_boxes, noobject_scale):
        # 计算各个 预测框 与所有gt的iou, 取最大值
        all_pred_box_iou = t_boxes.new_zeros(list(p_boxes.shape[:-1]) + [t_boxes.shape[0]])  # [H, W, 5, T]
        for t in range(len(t_boxes)):
            t_box = t_boxes[t]  # (x0, y0, x1, y1)

            xx0 = p_boxes[..., 0].clamp(min = t_box[0].item())  # [H, W, 5,]
            yy0 = p_boxes[..., 1].clamp(min = t_box[1].item())
            xx1 = p_boxes[..., 2].clamp(max = t_box[2].item())
            yy1 = p_boxes[..., 3].clamp(max = t_box[3].item())
            overlap = (xx1 - xx0).clamp(min = 0) * (yy1 - yy0).clamp(min = 0)  # [H, W, 5,]
            pred_area = (p_boxes[..., 2] - p_boxes[..., 0]) * (p_boxes[..., 3] - p_boxes[..., 1])
            t_box_area = (t_box[2] - t_box[0]) * (t_box[3] - t_box[1])

            all_pred_box_iou[..., t] = overlap / (pred_area + t_box_area - overlap)  # [H, W, 5,]
        # 每个 预测框 的max_iou小于0.6的，记为背景bg，计算noobj loss
        noobj_mask = all_pred_box_iou.max(dim = -1)[0] < 0.6
        bg_boxes_score = p_boxes_score[noobj_mask]
        noobj_score_loss = noobject_scale * F.mse_loss(bg_boxes_score, torch.zeros_like(bg_boxes_score))
        return noobj_score_loss

    @staticmethod
    def _compute_obj_loss(t_boxes, t_labels, correspond_box_idx, p_boxes_cls, p_boxes_xywh, p_boxes_score,
                          class_scale, coord_scale, object_scale):
        cor_box_cls = p_boxes_cls[correspond_box_idx]  # [t_boxes.shape[0], 20]
        cor_box_xywh = p_boxes_xywh[correspond_box_idx]  # [t_boxes.shape[0],4]
        cor_box_score = p_boxes_score[correspond_box_idx]  # [t_boxes.shape[0],1]
        # 计算其coord loss, class loss 和 iou_score loss
        class_loss = class_scale * F.cross_entropy(cor_box_cls, t_labels)
        coord_loss = torch.mean(coord_scale * (F.mse_loss(cor_box_xywh[:, :2], t_boxes[:, :2], reduction = 'none') +
                                               F.mse_loss(cor_box_xywh[:, 2:].log(), t_boxes[:, 2:].log(),
                                                          reduction = 'none')))
        obj_score_loss = object_scale * F.mse_loss(cor_box_score, torch.ones_like(cor_box_score))

        return class_loss, coord_loss, obj_score_loss

    def _get_matched_anchor_idx(self, t_boxes):
        # 计算这个gt与哪个原始(先验)anchor的iou最大
        prior_anchor_iou = t_boxes.new_zeros((t_boxes.shape[0], len(self.anchors)))
        for a in range(len(self.anchors)):
            anchor = t_boxes.new(self.anchors[a]) / self.size_divisible  # shape of (2,)
            overlap = torch.min(t_boxes[:, 2], anchor[0]) * torch.min(t_boxes[:, 3], anchor[1])
            prior_anchor_iou[:, a] = overlap / (t_boxes[:, 2:].prod(dim = -1) + anchor.prod(dim = -1) - overlap)
        # 返回与每个gt匹配的anchor的idx
        return prior_anchor_iou.argmax(dim = -1).reshape(-1, 1).long()  # [t_boxes.shape[0],]

    @staticmethod
    def bboxes_transform(boxes_xywh, image_sizes):
        """
        Args:
            boxes_xywh (Tensor): shape of [N, H, W, 5, 4]
            image_sizes (List[Tuple[height, width]])

        Returns:
            boxes_xyxy (Tensor): shape of [N, H, W, 5, 4]
        """
        boxes_xyxy = torch.zeros_like(boxes_xywh)
        boxes_xyxy[..., 0] = boxes_xywh[..., 0] - boxes_xywh[..., 2] * 0.5
        boxes_xyxy[..., 1] = boxes_xywh[..., 1] - boxes_xywh[..., 3] * 0.5
        boxes_xyxy[..., 2] = boxes_xywh[..., 0] + boxes_xywh[..., 2] * 0.5
        boxes_xyxy[..., 3] = boxes_xywh[..., 1] + boxes_xywh[..., 3] * 0.5

        for i in range(len(image_sizes)):
            boxes_xyxy[i] = detection_utils.clip_boxes_to_image(boxes_xyxy[i], image_sizes[i])

        return boxes_xyxy

    @staticmethod
    def bboxes_transform_to_xywh(boxes_xyxy):
        """
        Args:
            boxes_xyxy (Tensor): shape of [M, 4]

        Returns:
            boxes_xywh (Tensor): shape of [M, 4]
        """
        boxes_xywh = torch.zeros_like(boxes_xyxy)
        boxes_xywh[..., 0] = (boxes_xyxy[..., 2] + boxes_xyxy[..., 0]) * 0.5
        boxes_xywh[..., 1] = (boxes_xyxy[..., 3] + boxes_xyxy[..., 1]) * 0.5
        boxes_xywh[..., 2] = boxes_xyxy[..., 2] - boxes_xyxy[..., 0]
        boxes_xywh[..., 3] = boxes_xyxy[..., 3] - boxes_xyxy[..., 1]

        return boxes_xywh

    @staticmethod
    def check_targets(targets):
        assert targets is not None
        assert all("boxes" in t for t in targets)
        assert all("labels" in t for t in targets)
        for t in targets:
            assert t["boxes"].dtype.is_floating_point, 'target boxes must of float type'
            assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
            assert len(t["boxes"]) == len(t["labels"]), "the length of boxes do not match the length of labels"


def yolo_v2_darknet19(num_classes, **kwargs):
    backbone = Darknet19()
    return YOLOv2(backbone, num_classes, **kwargs)


def yolo_v2_resnet50(num_classes, pretrained_backbone = False, **kwargs):
    from torchvision.models.resnet import resnet50
    from torchvision.models._utils import IntermediateLayerGetter

    backbone = resnet50(pretrained = pretrained_backbone)
    return_layers = {'layer3': 'feat3', 'layer4': 'feat4'}
    body = IntermediateLayerGetter(backbone, return_layers = return_layers)
    backbone = nn.Sequential(body)
    backbone = PassthroughLayer(backbone)
    backbone.out_channels = 6144
    backbone.downsample_factor = 32

    anchor_boxes = (0.8 * torch.tensor([[38.08, 63.68],  # width, height for anchor 1
                                        [89.28, 147.20],  # width, height for anchor 2
                                        [145.28, 285.76],  # etc.
                                        [257.92, 169.28],
                                        [330.56, 340.80]])).tolist()

    return YOLOv2(backbone, num_classes, anchor_boxes, **kwargs)


def yolo_v2_resnet50_backbone(num_classes, pretrained_backbone = False, **kwargs):
    from torchvision.models.resnet import resnet50
    backbone = resnet50(pretrained = pretrained_backbone)
    backbone.fc = nn.Linear(512 * 4, num_classes)
    return backbone
