import torch
from torch import nn

'''
Channel Attention Mechanisms module
'''
class senet(nn.Module):
    def __init__(self, input_channels=64,ratio = 16):
        super(senet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.s1 = nn.Sequential(
            nn.Linear(input_channels, input_channels // ratio, False),
            nn.ReLU(),
            nn.Linear(input_channels // ratio, input_channels, False)
        )
        self.Sigmoid = nn.Sigmoid()
    def forward(self, x):
        temp = x
        b, c, h, w = x.size()
        avg = self.avg_pool(x).view(b, -1)
        max = self.max_pool(x).view(b, -1)
        avg_MLP = self.s1(avg)
        max_MLP = self.s1(max)
        add_avg_max = avg_MLP + max_MLP
        x = self.Sigmoid(add_avg_max).view(b, c, 1, 1)
        out =x*temp
        return out

if __name__ == '__main__':
    x = torch.randn(1, 3, 40, 50)
    conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    x = conv(x)
    x = senet(input_channels=64)(x)
    print(x.size())