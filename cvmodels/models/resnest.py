##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## Email: zhanghang0704@gmail.com
## Copyright (c) 2020
##
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""ResNet variants"""
import math
import torch
import torch.nn as nn

from .layers.splat import SplAtConv2d
from .resnet_modified import ResNet, load_pretrained_model

__all__ = ['resnest50', 'resnest101', 'resnest200', 'resnest269', 'resnest50_fast_1s1x64d', 'resnest50_fast_2s1x64d',
           'resnest50_fast_4s1x64d', 'resnest50_fast_1s2x40d', 'resnest50_fast_2s2x40d', 'resnest50_fast_4s2x40d',
           'resnest50_fast_1s4x24d']

_url_format = 'https://s3.us-west-1.wasabisys.com/resnest/torch/{}-{}.pth'

_model_sha256 = {name: checksum for checksum, name in [
    ('528c19ca', 'resnest50'),
    ('22405ba7', 'resnest101'),
    ('75117900', 'resnest200'),
    ('0cc87c48', 'resnest269'),
    ('d8fbf808', 'resnest50_fast_1s1x64d'),
    ('44938639', 'resnest50_fast_2s1x64d'),
    ('f74f3fc3', 'resnest50_fast_4s1x64d'),
    ('32830b84', 'resnest50_fast_1s2x40d'),
    ('9d126481', 'resnest50_fast_2s2x40d'),
    ('41d14ed0', 'resnest50_fast_4s2x40d'),
    ('d4a4f76f', 'resnest50_fast_1s4x24d'),
]}


def short_hash(name):
    if name not in _model_sha256:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=name))
    return _model_sha256[name][:8]


resnest_model_urls = {name: _url_format.format(name, short_hash(name)) for name in _model_sha256.keys()}


class DropBlock2D(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return nn.functional.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)


class ResNestBottleneck(nn.Module):
    """ResNet Bottleneck
    Copy from `https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/resnest.py`
    """
    # pylint: disable=unused-argument
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 radix=1, cardinality=1, base_width=64,
                 reduce_first=1, dilation=1, first_dilation=None,
                 act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, drop_block=None,
                 use_pooling=False, pool_type='avg', pool_first=False,
                 rectified_conv=False, rectify_avg=False, ):
        super(ResNestBottleneck, self).__init__()
        assert reduce_first == 1  # not supported
        group_width = int(planes * (base_width / 64.)) * cardinality
        first_planes = group_width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        self.pool_layer = None
        if stride > 1 and use_pooling:
            if pool_type == 'max':
                self.pool_layer = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
                stride = 1
            elif pool_type == 'avg':
                self.pool_layer = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
                stride = 1
        if radix >= 1:
            self.conv2 = SplAtConv2d(first_planes, group_width, kernel_size=3, stride=stride, padding=first_dilation,
                                     dilation=first_dilation, groups=cardinality, radix=radix, norm_layer=norm_layer,
                                     drop_block=drop_block)
            self.bn2 = None  # FIXME revisit, here to satisfy current torchscript fussyness
            self.act2 = None
        elif rectified_conv:
            from rfconv import RFConv2d
            self.conv2 = RFConv2d(
                first_planes, group_width, kernel_size=3, stride=stride, padding=first_dilation,
                dilation=first_dilation,
                groups=cardinality, bias=False, average_mode=rectify_avg)
            self.bn2 = norm_layer(group_width)
        else:
            self.conv2 = nn.Conv2d(
                first_planes, group_width, kernel_size=3, stride=stride, padding=first_dilation,
                dilation=first_dilation, groups=cardinality, bias=False)
            self.bn2 = norm_layer(group_width)
            self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(group_width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer(inplace=True)

        self.drop_block = drop_block
        self.radix = radix
        self.pool_first = pool_first
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.drop_block(out) if self.drop_block is not None else out
        out = self.act1(out)

        if self.pool_layer is not None and self.pool_first:
            out = self.pool_layer(out)

        out = self.conv2(out)
        if self.bn2 is not None:
            out = self.bn2(out)
            out = self.drop_block(out) if self.drop_block is not None else out
            out = self.act2(out)

        if self.pool_layer is not None and not self.pool_first:
            out = self.pool_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.drop_block(out) if self.drop_block is not None else out

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual
        out = self.act3(out)
        return out


# class ResNet(nn.Module):
#     """ResNet Variants
#
#     Parameters
#     ----------
#     block : Block
#         Class for the residual block. Options are BasicBlockV1, BottleneckV1.
#     layers : list of int
#         Numbers of layers in each block
#     classes : int, default 1000
#         Number of classification classes.
#     dilated : bool, default False
#         Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
#         typically used in Semantic Segmentation.
#     norm_layer : object
#         Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
#         for Synchronized Cross-GPU BachNormalization).
#
#     Reference:
#
#         - He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
#
#         - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
#     """
#
#     # pylint: disable=unused-variable
#     def __init__(self, block, layers, radix = 1, groups = 1, bottleneck_width = 64,
#                  num_classes = 1000, dilated = False, dilation = 1,
#                  deep_stem = False, stem_width = 64, avg_down = False,
#                  rectified_conv = False, rectify_avg = False,
#                  avd = False, avd_first = False,
#                  final_drop = 0.0, dropblock_prob = 0,
#                  last_gamma = False, norm_layer = nn.BatchNorm2d):
#         self.cardinality = groups
#         self.bottleneck_width = bottleneck_width
#         # ResNet-D params
#         self.inplanes = stem_width * 2 if deep_stem else 64
#         self.avg_down = avg_down
#         self.last_gamma = last_gamma
#         # ResNeSt params
#         self.radix = radix
#         self.avd = avd
#         self.avd_first = avd_first
#
#         super(ResNet, self).__init__()
#         self.rectified_conv = rectified_conv
#         self.rectify_avg = rectify_avg
#         if rectified_conv:
#             from rfconv import RFConv2d
#             conv_layer = RFConv2d
#         else:
#             conv_layer = nn.Conv2d
#         conv_kwargs = {'average_mode': rectify_avg} if rectified_conv else {}
#         if deep_stem:
#             self.conv1 = nn.Sequential(
#                 conv_layer(3, stem_width, kernel_size = 3, stride = 2, padding = 1, bias = False, **conv_kwargs),
#                 norm_layer(stem_width),
#                 nn.ReLU(inplace = True),
#                 conv_layer(stem_width, stem_width, kernel_size = 3, stride = 1, padding = 1, bias = False, **conv_kwargs),
#                 norm_layer(stem_width),
#                 nn.ReLU(inplace = True),
#                 conv_layer(stem_width, stem_width * 2, kernel_size = 3, stride = 1, padding = 1, bias = False, **conv_kwargs),
#             )
#         else:
#             self.conv1 = conv_layer(3, 64, kernel_size = 7, stride = 2, padding = 3,
#                                     bias = False, **conv_kwargs)
#         self.bn1 = norm_layer(self.inplanes)
#         self.relu = nn.ReLU(inplace = True)
#         self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
#         self.layer1 = self._make_layer(block, 64, layers[0], norm_layer = norm_layer, is_first = False)
#         self.layer2 = self._make_layer(block, 128, layers[1], stride = 2, norm_layer = norm_layer)
#         if dilated or dilation == 4:
#             self.layer3 = self._make_layer(block, 256, layers[2], stride = 1,
#                                            dilation = 2, norm_layer = norm_layer,
#                                            dropblock_prob = dropblock_prob)
#             self.layer4 = self._make_layer(block, 512, layers[3], stride = 1,
#                                            dilation = 4, norm_layer = norm_layer,
#                                            dropblock_prob = dropblock_prob)
#         elif dilation == 2:
#             self.layer3 = self._make_layer(block, 256, layers[2], stride = 2,
#                                            dilation = 1, norm_layer = norm_layer,
#                                            dropblock_prob = dropblock_prob)
#             self.layer4 = self._make_layer(block, 512, layers[3], stride = 1,
#                                            dilation = 2, norm_layer = norm_layer,
#                                            dropblock_prob = dropblock_prob)
#         else:
#             self.layer3 = self._make_layer(block, 256, layers[2], stride = 2,
#                                            norm_layer = norm_layer,
#                                            dropblock_prob = dropblock_prob)
#             self.layer4 = self._make_layer(block, 512, layers[3], stride = 2,
#                                            norm_layer = norm_layer,
#                                            dropblock_prob = dropblock_prob)
#         self.num_features = 512 * block.expansion
#         self.avgpool = GlobalAvgPool2d()
#         self.drop = nn.Dropout(final_drop) if final_drop > 0.0 else None
#         self.fc = nn.Linear(self.num_features, num_classes)

def resnest50(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    model_kwargs = dict(stem_width=32, stem_type='deep', act_layer=nn.ReLU, avg_down=True,
                        block_args=dict(radix=2, use_pooling=True, pool_type='avg', pool_first=False))
    model = ResNet(ResNestBottleneck, [3, 4, 6, 3], num_classes=num_classes, in_chans=in_chans, **model_kwargs)

    if pretrained:
        pretrained_url = resnest_model_urls['resnest50']
        load_pretrained_model(model, 'conv1', in_chans, url=pretrained_url)
    return model


def resnest101(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    model_kwargs = dict(stem_width=64, stem_type='deep', act_layer=nn.ReLU, avg_down=True,
                        block_args=dict(radix=2, use_pooling=True, pool_type='avg', pool_first=False))
    model = ResNet(ResNestBottleneck, [3, 4, 23, 3], num_classes=num_classes, in_chans=in_chans, **model_kwargs)

    if pretrained:
        pretrained_url = resnest_model_urls['resnest101']
        load_pretrained_model(model, 'conv1', in_chans, url=pretrained_url)
    return model


def resnest200(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    model_kwargs = dict(stem_width=64, stem_type='deep', act_layer=nn.ReLU, avg_down=True,
                        block_args=dict(radix=2, use_pooling=True, pool_type='avg', pool_first=False))
    model = ResNet(ResNestBottleneck, [3, 24, 36, 3], num_classes=num_classes, in_chans=in_chans, **model_kwargs)

    if pretrained:
        pretrained_url = resnest_model_urls['resnest200']
        load_pretrained_model(model, 'conv1', in_chans, url=pretrained_url)
    return model


def resnest269(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    model_kwargs = dict(stem_width=64, stem_type='deep', act_layer=nn.ReLU, avg_down=True,
                        block_args=dict(radix=2, use_pooling=True, pool_type='avg', pool_first=False))
    model = ResNet(ResNestBottleneck, [3, 30, 48, 8], num_classes=num_classes, in_chans=in_chans, **model_kwargs)

    if pretrained:
        pretrained_url = resnest_model_urls['resnest200']
        load_pretrained_model(model, 'conv1', in_chans, url=pretrained_url)
    return model


def resnest50_fast_1s1x64d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    model_kwargs = dict(stem_width=32, stem_type='deep', act_layer=nn.ReLU, avg_down=True,
                        block_args=dict(radix=1, cardinality=1, use_pooling=True, pool_type='avg', pool_first=True))
    model = ResNet(ResNestBottleneck, [3, 4, 6, 3], num_classes=num_classes, in_chans=in_chans, **model_kwargs)

    if pretrained:
        pretrained_url = resnest_model_urls['resnest50_fast_1s1x64d']
        load_pretrained_model(model, 'conv1', in_chans, url=pretrained_url)
    return model


def resnest50_fast_2s1x64d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    model_kwargs = dict(stem_width=32, stem_type='deep', act_layer=nn.ReLU, avg_down=True,
                        block_args=dict(radix=2, cardinality=1, use_pooling=True, pool_type='avg', pool_first=True))
    model = ResNet(ResNestBottleneck, [3, 4, 6, 3], num_classes=num_classes, in_chans=in_chans, **model_kwargs)

    if pretrained:
        pretrained_url = resnest_model_urls['resnest50_fast_2s1x64d']
        load_pretrained_model(model, 'conv1', in_chans, url=pretrained_url)
    return model


def resnest50_fast_4s1x64d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    model_kwargs = dict(stem_width=32, stem_type='deep', act_layer=nn.ReLU, avg_down=True,
                        block_args=dict(radix=4, cardinality=1, use_pooling=True, pool_type='avg', pool_first=True))
    model = ResNet(ResNestBottleneck, [3, 4, 6, 3], num_classes=num_classes, in_chans=in_chans, **model_kwargs)

    if pretrained:
        pretrained_url = resnest_model_urls['resnest50_fast_4s1x64d']
        load_pretrained_model(model, 'conv1', in_chans, url=pretrained_url)
    return model


def resnest50_fast_1s2x40d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    model_kwargs = dict(stem_width=32, stem_type='deep', act_layer=nn.ReLU, avg_down=True,
                        block_args=dict(radix=1, cardinality=2, base_width=40, use_pooling=True, pool_type='avg',
                                        pool_first=True))
    model = ResNet(ResNestBottleneck, [3, 4, 6, 3], num_classes=num_classes, in_chans=in_chans, **model_kwargs)

    if pretrained:
        pretrained_url = resnest_model_urls['resnest50_fast_1s2x40d']
        load_pretrained_model(model, 'conv1', in_chans, url=pretrained_url)
    return model


def resnest50_fast_2s2x40d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    model_kwargs = dict(stem_width=32, stem_type='deep', act_layer=nn.ReLU, avg_down=True,
                        block_args=dict(radix=2, cardinality=2, base_width=40, use_pooling=True, pool_type='avg',
                                        pool_first=True))
    model = ResNet(ResNestBottleneck, [3, 4, 6, 3], num_classes=num_classes, in_chans=in_chans, **model_kwargs)

    if pretrained:
        pretrained_url = resnest_model_urls['resnest50_fast_2s2x40d']
        load_pretrained_model(model, 'conv1', in_chans, url=pretrained_url)
    return model


def resnest50_fast_4s2x40d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    model_kwargs = dict(stem_width=32, stem_type='deep', act_layer=nn.ReLU, avg_down=True,
                        block_args=dict(radix=4, cardinality=2, base_width=40, use_pooling=True, pool_type='avg',
                                        pool_first=True))
    model = ResNet(ResNestBottleneck, [3, 4, 6, 3], num_classes=num_classes, in_chans=in_chans, **model_kwargs)

    if pretrained:
        pretrained_url = resnest_model_urls['resnest50_fast_4s2x40d']
        load_pretrained_model(model, 'conv1', in_chans, url=pretrained_url)
    return model


def resnest50_fast_1s4x24d(pretrained=False, num_classes=1000, in_chans=3, **kwargs):
    model_kwargs = dict(stem_width=32, stem_type='deep', act_layer=nn.ReLU, avg_down=True,
                        block_args=dict(radix=1, cardinality=4, base_width=24, use_pooling=True, pool_type='avg',
                                        pool_first=True))
    model = ResNet(ResNestBottleneck, [3, 4, 6, 3], num_classes=num_classes, in_chans=in_chans, **model_kwargs)

    if pretrained:
        pretrained_url = resnest_model_urls['resnest50_fast_1s4x24d']
        load_pretrained_model(model, 'conv1', in_chans, url=pretrained_url)
    return model
