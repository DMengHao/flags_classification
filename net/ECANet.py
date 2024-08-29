import math

import torch
from torch import nn


class ECABlock(nn.Module):
    def __init__(self, in_channels, gamma = 2, b = 1):
        super(ECABlock, self).__init__()
        kernel_size = int(abs((math.log(in_channels,2)+b)/gamma))
        kernel_size = kernel_size if kernel_size %2 else kernel_size+1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size = kernel_size, bias = False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, h, w = x.size()
        avg = self.avg_pool(x).view(b, 1, c)
        out = self.conv(avg)
        out = self.sigmoid(out).view(b, c, 1, 1)
        return x * out
if __name__ == '__main__':
    x = torch.randn(64, 64, 50, 40)
    ecl = ECABlock(64, gamma = 2)(x)
    print(ecl)
