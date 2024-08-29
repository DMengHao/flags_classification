import torch
from torch import nn

class SAM(nn.Module):
    def __init__(self,kernel_size = 7):
        super(SAM, self).__init__()
        padding = 7 // 2 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, stride=1, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg, _ = torch.max(x, dim=1, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        max_avg = torch.cat([avg, mean], dim=1)
        SA = self.conv(max_avg)
        out = self.sigmoid(SA)
        return x * out
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1conv=False, stride=1):
        super().__init__()
        self.ReLu = nn.ReLU()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        if use_1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1)
        else:
            self.conv3 = None
    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.ReLu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        if self.conv3:
            x= self.conv3(x)
        out = self.ReLu(x+y)
        return out

class ResNet18_SAM(nn.Module):
    def __init__(self):
        super(ResNet18_SAM, self).__init__()
        self.s1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2,padding=1),
        )
        self.SAM = SAM()
        self.s2 = nn.Sequential(
            Residual(64, 64, use_1conv=False, stride=1),
            Residual(64, 64, use_1conv=False, stride=2),
        )
        self.s3 = nn.Sequential(
            Residual(64, 128, use_1conv=True, stride=2),
            Residual(128, 128, use_1conv=False, stride=2),
        )
        self.s4 = nn.Sequential(
            Residual(128, 256, use_1conv=True, stride=2),
            Residual(256, 256, use_1conv=False, stride=2),
        )
        self.s5 = nn.Sequential(
            Residual(256, 512, use_1conv=True, stride=2),
            Residual(512, 512, use_1conv=True, stride=2),
        )
        self.s6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 4),
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.s1(x)
        x = self.SAM(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        x = self.s5(x)
        x = self.s6(x)
        return x
if __name__ == '__main__':
    x = torch.rand(64, 50, 40, 3)
    y = ResNet18_SAM()(x)
    print(y.shape)

