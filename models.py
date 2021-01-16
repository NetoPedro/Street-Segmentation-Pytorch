import torch
from torch import nn
from torchvision import models

class UNetBlockA(nn.Module):
    def __init__(self,channel_in,channel_out):
        super(UNetBlockA,self).__init__()
        self.sequential1 = torch.nn.Sequential(
            nn.Conv2d(channel_in,channel_out,3,padding=1),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel_out,channel_out,3,padding=1),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(inplace=True)

        )
    def forward(self,x):
        return self.sequential1(x)

class UNetBlockB(nn.Module):
    def __init__(self,channel_in,channel_out,biliniar=False):
        super(UNetBlockB,self).__init__()
        if biliniar:
            self.sequential1 = torch.nn.Sequential(
                nn.Conv2d(channel_in,channel_out,3,padding=1),
                nn.BatchNorm2d(channel_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel_out,channel_out,3,padding=1),
                nn.BatchNorm2d(channel_out),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            )
        else:
            self.up = nn.ConvTranspose2d(channel_in,channel_in//2,2,stride=2,padding=1)
            self.sequential1 = torch.nn.Sequential(
                nn.Conv2d(channel_in,channel_out,3,padding=1),
                nn.BatchNorm2d(channel_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel_out,channel_out,3,padding=1),
                nn.BatchNorm2d(channel_out),
                nn.ReLU(inplace=True),
            )
    def forward(self,x,x2):
        x = self.up(x)
        diffY = x2.size()[2] - x.size()[2]
        diffX = x2.size()[3] - x.size()[3]
        x = torch.nn.functional.pad(x, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        #x= x + x2
        x = torch.cat((x,x2),dim=1)
        return self.sequential1(x)




class UNet(nn.Module):

    def __init__(self,output_dim=1,pretrained = False):
        super(UNet,self).__init__()
        self.pretrained = pretrained
        if pretrained:
            self.conv1 = models.resnet34(pretrained=True).conv1
            self.conv1.stride=1
            self.bn1 = models.resnet34(pretrained=True).bn1
            self.sequential1 = models.resnet34(pretrained=True).layer1
            self.sequential2 = models.resnet34(pretrained=True).layer2
            self.sequential3 = models.resnet34(pretrained=True).layer3
            self.sequential4 = models.resnet34(pretrained=True).layer4
            self.sequential5 = UNetBlockA(512,1024)
        else:
            self.sequential1 = UNetBlockA(3,64)
            self.sequential2 = UNetBlockA(64,128)
            self.sequential3 = UNetBlockA(128,256)
            self.sequential4 = UNetBlockA(256,512)
            self.sequential5 = UNetBlockA(512,1024)
            self.sequential6 = UNetBlockA(1024,2048)
            self.sequential7 = UNetBlockB(2048,1024)

        self.sequential8 = UNetBlockB(1024,512)
        self.sequential9 = UNetBlockB(512,256)
        self.sequential10 = UNetBlockB(256,128)
        self.sequential11 = UNetBlockB(128,64)
        self.final_conv = nn.Conv2d(64,output_dim,1)      
        self.pool = nn.MaxPool2d(2)  
    def forward(self,x):
        if self.pretrained:
            x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
            x1 = self.sequential1(x)
            x2 = self.sequential2(x1)
            x3 = self.sequential3(x2)
            x4 = self.sequential4(x3)
            x5 = self.sequential5(x4)

            x5 = self.sequential8(x5,x4)
            x5 = self.sequential9(x5,x3)
            x5 = self.sequential10(x5,x2)
            x5 = self.sequential11(x5,x1)
            x5 = self.final_conv(x5)
            return x5

        x1 = self.sequential1(x)
        x2 = self.sequential2(self.pool(x1))
        x3 = self.sequential3(self.pool(x2))
        x4 = self.sequential4(self.pool(x3))
        x5 = self.sequential5(self.pool(x4))
        x6 = self.sequential6(x5)

        x6 = self.sequential7(x6,x5)
        x6 = self.sequential8(x6,x4)
        x6 = self.sequential9(x6,x3)
        x6 = self.sequential10(x6,x2)
        x6 = self.sequential11(x6,x1)
        x6 = self.final_conv(x6)
        return x6



class FCN:
    pass