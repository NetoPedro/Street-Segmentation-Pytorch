import models
import train

net = models.UNet(output_dim=5,pretrained=True)
train.train(net)
