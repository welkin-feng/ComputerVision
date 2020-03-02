from torch import nn

__all__ = ['SELayer', 'SEModule']


class SELayer(nn.Module):
    def __init__(self, channel, reduction = 16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias = False),
            nn.ReLU(inplace = True),
            nn.Linear(channel // reduction, channel, bias = False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEModule(nn.Module):

    def __init__(self, channels, reduction = 16, act_layer = nn.ReLU):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduction_channels = max(channels // reduction, 8)
        self.fc1 = nn.Conv2d(
            channels, reduction_channels, kernel_size = 1, padding = 0, bias = True)
        self.act = act_layer(inplace = True)
        self.fc2 = nn.Conv2d(
            reduction_channels, channels, kernel_size = 1, padding = 0, bias = True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.fc1(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        return x * x_se.sigmoid()
