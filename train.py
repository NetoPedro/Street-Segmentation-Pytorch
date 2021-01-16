import torch 
import data_loader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from collections import defaultdict

class_weights = torch.tensor([0.1,0.3,0.2,0.3,0.01]).cuda()


class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )

def train(net):
    scaler = torch.cuda.amp.GradScaler()
    net.cuda()
    net.train()
    
    data_transforms = {
    'train': A.Compose(
    [
        A.RandomCrop(400, 400),
        A.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.15, rotate_limit=20, p=0.5),
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5),
        A.RandomBrightnessContrast(p=0.6),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),]),

    'val': A.Compose(
    [   
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),]),
    }

    path = "/home/pedro/Documents/Datasets/bdd100k/seg/"

    trainloader = torch.utils.data.DataLoader(data_loader.Bdd100kDataset(path,mode="train",transforms=data_transforms["train"]), batch_size=8,
                                            shuffle=True, num_workers=4,pin_memory=True)

    validationloader = torch.utils.data.DataLoader(data_loader.Bdd100kDataset(path,mode="val",transforms=data_transforms["val"]), batch_size=4,
                                            shuffle=False, num_workers=4,pin_memory=True)
    
    optimizer = torch.optim.Adam(net.parameters(),lr=5e-4)
    
    best_dice = 0

    for epoch in range(0,300):  # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        running_IoU= 0.0
        metric_monitor = MetricMonitor()
        stream = tqdm(trainloader)
        loss_func = CrossEntropyLoss2d()
        for i, (ipt,target) in enumerate(stream, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = ipt.cuda(non_blocking=True),target.cuda(non_blocking=True)

            # zero the parameter gradients
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():

                outputs = net(inputs)
                dice_score_ = dice_score(outputs,labels)
                labels = torch.argmax(labels,dim=1)
                dice,focal = criterion2(outputs, labels)
                loss = dice+focal*0.7

            scaler.scale(loss).backward()
            scaler.step(optimizer)

            # Updates the scale for next iteration
            scaler.update()

            metric_monitor.update("Focal", focal.item())
            metric_monitor.update("Dice", dice.item())
            metric_monitor.update("Dice_", dice_score_.item())
            stream.set_description(
            "Epoch: {epoch}. Train.      {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor))
            
        
        
        net.eval()
        trans = transforms.ToPILImage()

        with torch.no_grad():
            seen_images = 0
            running_loss = 0.0
            metric_monitor = MetricMonitor()
            stream = tqdm(validationloader)
            for i, (ipt,target,path) in enumerate(stream, 0):
            # get the inputs; data is a list of [inputs, labels]
                inputs, labels = ipt.cuda(non_blocking=True),target.cuda(non_blocking=True)
                with torch.cuda.amp.autocast():
                    outputs = net(inputs)
                    loss = dice_score(outputs, labels)
                seen_images +=ipt.shape[0]

                running_loss += loss.item() *ipt.shape[0]

                if i % 100 == 0:
                    display_prediction(path[0],outputs[0,:,:,:].cpu(),epoch,i,train=False)
                    display_prediction(path[0],labels[0,:,:,:].cpu(),epoch,i+1,train=False)

                metric_monitor.update("Loss", loss.item())
                stream.set_description("Epoch: {epoch}. Train.      {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor))

            print('[%d] dice: %.3f ' %
                    (epoch + 1, running_loss / seen_images))

            if best_dice < running_loss:
                best_dice = running_loss
                torch.save(net, "./model_"+str(epoch)+".pth")

def criterion(y,target,epsilon=1e-6):
    cel = FocalLoss(weight=class_weights)
    return soft_dice(y,target,epsilon) ,  cel(y,target)

def criterion2(y,target,epsilon=1e-6):
    cel = torch.nn.CrossEntropyLoss()
    return soft_dice(y,target,epsilon) ,  cel(y,target)

class FocalLoss(torch.nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight 

    def forward(self, input, target):

        ce_loss = torch.nn.functional.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

def soft_dice(y,target,eps=1e-6):
    """
    https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py#L131-L175
    """
    num_classes = y.shape[1]
    true_1_hot = torch.eye(num_classes)[target.squeeze(1)]
    true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
    probas = torch.nn.functional.softmax(y, dim=1)
    true_1_hot = true_1_hot.type(y.type())
    dims = (0,) + tuple(range(2, target.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)


def dice_score(y,target,eps=1e-6):
    num_classes = y.shape[1]
    target = torch.argmax(target,dim=1)
    true_1_hot = torch.eye(num_classes)[target.squeeze(1)]
    true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
    probas = torch.nn.functional.softmax(y, dim=1)
    max_idx = torch.argmax(probas, 1, keepdim=True)
    one_hot = torch.FloatTensor(probas.shape).cuda()
    one_hot.zero_()
    one_hot = torch.nn.functional.one_hot(torch.squeeze(max_idx,dim=1),num_classes).permute((0,3,1,2))
    true_1_hot = true_1_hot.type(y.type())
    dims = (0,) + tuple(range(2, target.ndimension()))
    intersection = torch.sum(one_hot * true_1_hot, dims)
    cardinality = torch.sum(one_hot + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return dice_loss




def display_prediction(path,prediction,epoch,i,train=True):
    image = np.load(path)

    image = np.squeeze(image)
    new_image = np.copy(image)
    prediction = np.squeeze(prediction.numpy())
    mask = np.argmax(prediction, axis=0)
   
    new_image[mask[:,:]==0, :] = [0,255,0]
    # sidewalk = blue
    new_image[mask[:,:]==1, :] = [0,0,255]
    # pedestrians = yellow
    new_image[mask[:,:]==2, :] = [255,255,0]
    # vehicles = red
    new_image[mask[:,:]==3, :] = [255,0,0]
    
    new_image = Image.blend(Image.fromarray(image, mode='RGB').convert('RGBA'),
                            Image.fromarray(new_image, mode='RGB').convert('RGBA'),
                            alpha=0.6)
    
    plt.imshow(new_image, interpolation='nearest')
    if not(train):
        epoch = "validation_" + str(epoch) 
    plt.savefig("images/sample_"+str(epoch)+"_"+str(i)+".png")