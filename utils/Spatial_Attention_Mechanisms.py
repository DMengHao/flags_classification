import torch
from torch import nn

'''
    Spatial Attention Mechanisms model
'''
class Spatial_Attention_Mechanisms(object):
    def __init__(self, kernel_size = 7):
        super(Spatial_Attention_Mechanisms, self).__init__()
        padding = 7//2 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2,1,kernel_size,stride=1, padding=padding,bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        avg,_ = torch.max(x,dim = 1,keepdim = True)
        mean = torch.mean(x,dim = 1,keepdim = True)
        max_avg = torch.cat([avg,mean],dim = 1)
        SA = self.conv(max_avg)
        out = self.sigmoid(SA)
        return x*out

if __name__ == '__main__':
    x = torch.randn((1,64,40,50))
    SAM = Spatial_Attention_Mechanisms(kernel_size=7)
    x= SAM.forward(x)
    print(x.size())
