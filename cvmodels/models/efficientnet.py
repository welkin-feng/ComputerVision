import torch
from torch import nn
from torch.nn import functional as F

from .efficientnet_utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    url_map,
    url_map_advprop,
    Swish,
    MemoryEfficientSwish,
)

__all__ = ['EfficientNet', 'effcientnet_b0']


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size = global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels = inp, out_channels = oup, kernel_size = 1, bias = False)
            self._bn0 = nn.BatchNorm2d(num_features = oup, momentum = self._bn_mom, eps = self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels = oup, out_channels = oup, groups = oup,  # groups makes it depthwise
            kernel_size = k, stride = s, bias = False)
        self._bn1 = nn.BatchNorm2d(num_features = oup, momentum = self._bn_mom, eps = self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels = oup, out_channels = num_squeezed_channels, kernel_size = 1)
            self._se_expand = Conv2d(in_channels = num_squeezed_channels, out_channels = oup, kernel_size = 1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels = oup, out_channels = final_oup, kernel_size = 1, bias = False)
        self._bn2 = nn.BatchNorm2d(num_features = final_oup, momentum = self._bn_mom, eps = self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate = None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p = drop_connect_rate, training = self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient = True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks

    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')

    """

    def __init__(self, blocks_args = None, global_params = None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size = global_params.image_size)

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size = 3, stride = 2, bias = False)
        self._bn0 = nn.BatchNorm2d(num_features = out_channels, momentum = bn_mom, eps = bn_eps)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters = round_filters(block_args.input_filters, self._global_params),
                output_filters = round_filters(block_args.output_filters, self._global_params),
                num_repeat = round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters = block_args.output_filters, stride = 1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        # Head
        in_channels = block_args.output_filters  # output of final block
        self.out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(in_channels, self.out_channels, kernel_size = 1, bias = False)
        self._bn1 = nn.BatchNorm2d(num_features = self.out_channels, momentum = bn_mom, eps = bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(self.out_channels, self._global_params.num_classes)
        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient = True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def forward_features(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate = drop_connect_rate)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        bs = inputs.size(0)
        # Convolution layers
        x = self.forward_features(inputs)

        # Pooling and final linear layer
        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        x = self._dropout(x)
        x = self._fc(x)
        return x

    @classmethod
    def from_name(cls, model_name, override_params = None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, advprop = False, num_classes = 1000, in_channels = 3):
        model = cls.from_name(model_name, override_params = {'num_classes': num_classes})
        load_pretrained_weights(model, model_name, load_fc = (num_classes == 1000), advprop = advprop)
        if in_channels != 3:
            Conv2d = get_same_padding_conv2d(image_size = model._global_params.image_size)
            out_channels = round_filters(32, model._global_params)
            model._conv_stem = Conv2d(in_channels, out_channels, kernel_size = 3, stride = 2, bias = False)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        """ Validates model name. """
        valid_models = ['efficientnet-b' + str(i) for i in range(9)]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))


def load_pretrained_weights(model, model_name, load_fc = True, advprop = False):
    """ Loads pretrained weights, and downloads if loading for the first time. """
    # AutoAugment or Advprop (different preprocessing)
    url_map_ = url_map_advprop if advprop else url_map

    if load_fc:
        load_pretrained_model(model, '_conv_stem', 3, url = url_map_[model_name])
    else:
        load_pretrained_model(model, '_conv_stem', 3, url = url_map_[model_name], skip = ('_fc.weight', '_fc.bias',))


def load_pretrained_model(model, conv1_name, in_chans = 3, model_path = '', url = '', skip = (), conversion = ()):
    import os, numpy
    if os.path.isfile(model_path):
        state_dict = torch.load(model_path, map_location = 'cpu')
    elif url != '':
        import torch.utils.model_zoo as model_zoo
        state_dict = model_zoo.load_url(url, progress = False, map_location = 'cpu')
    else:
        return
    if in_chans == 1:
        print('Converting first conv (%s) from 3 to 1 channel' % conv1_name)
        conv1_weight = state_dict[conv1_name + '.weight']
        state_dict[conv1_name + '.weight'] = conv1_weight.sum(dim = 1, keepdim = True)
    elif in_chans != 3:
        assert False, "Invalid in_chans for pretrained weights"
    print('=> loading pretrained model {}'.format(model_path))
    conversion = numpy.array(conversion).reshape(-1, 2) if len(conversion) else []
    model_dict = model.state_dict()
    pretrained_state_dict = {}
    for ks in state_dict.keys():
        if ks in model_dict.keys() and all(s not in ks for s in skip):
            km = ks
            for _km, _ks in conversion:
                if ks == _ks:
                    km = _km
                    break
            pretrained_state_dict[km] = state_dict[ks]
    print(
        f"=> loading pretrained model weight length {len(pretrained_state_dict)} / total_state_dict {len(state_dict)} / total_model_dict {len(model_dict)}")
    model_dict.update(pretrained_state_dict)
    model.load_state_dict(model_dict, strict = False)


def effcientnet_b0(pretrained = False, in_chans = 3, **kwargs):
    model = EfficientNet.from_name('efficientnet-b0')
    if in_chans != 3:
        Conv2d = get_same_padding_conv2d(image_size = model._global_params.image_size)
        out_channels = round_filters(32, model._global_params)
        model._conv_stem = Conv2d(in_chans, out_channels, kernel_size = 3, stride = 2, bias = False)
    if pretrained:
        load_pretrained_model(model, '_conv_stem', in_chans, url = url_map['efficientnet-b0'])
    return model