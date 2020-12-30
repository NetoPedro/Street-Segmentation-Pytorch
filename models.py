import torch
from torch import nn


class UNetBlockA(nn.Module):
    def __init__(self,channel_in,channel_out):
        self.sequential1 = torch.nn.Sequential(
            nn.Conv2d(channel_in,channel_out,3)
            nn.ReLU()
            nn.Conv2d(channel_out,channel_out,3)
            nn.ReLU()
            nn.Conv2d(channel_out,channel_out,3)
            nn.ReLU()
        )
    def forward(self,x):
        return self.sequential1(x)

class UNet(nn.Module):

    def __init__(self,output_dim=1):
        self.sequential1 = torch.nn.Sequential(
            nn.Conv2d(3,64,3)
            nn.ReLU()
            nn.Conv2d(64,64,3)
            nn.ReLU()
            nn.Conv2d(64,64,3)
            nn.ReLU()
        )
        self.sequential2 = torch.nn.Sequential(
            nn.Conv2d(64,128,3)
            nn.ReLU()
            nn.Conv2d(128,128,3)
            nn.ReLU()
            nn.Conv2d(128,128,3)
            nn.ReLU()
        )

        self.sequential3 = torch.nn.Sequential(
            nn.Conv2d(64,128,3)
            nn.ReLU()
            nn.Conv2d(128,128,3)
            nn.ReLU()
            nn.Conv2d(128,128,3)
            nn.ReLU()
        )

        self.sequential4 = torch.nn.Sequential(
            nn.Conv2d(64,128,3)
            nn.ReLU()
            nn.Conv2d(128,128,3)
            nn.ReLU()
            nn.Conv2d(128,128,3)
            nn.ReLU()
        )

        self.sequential5 = torch.nn.Sequential(
            nn.Conv2d(64,128,3)
            nn.ReLU()
            nn.Conv2d(128,128,3)
            nn.ReLU()
            nn.Conv2d(128,128,3)
            nn.ReLU()
        )
        
    def forward(self,x):
        pass

