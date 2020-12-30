import models
import train

net = models.UNet(output_dim=1)
train.train(net)
