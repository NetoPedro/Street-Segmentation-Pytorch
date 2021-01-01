import torch 
import data_loader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt
import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2
def train(net):
    
    net.cuda()
    net.train()
    unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    data_transforms = {
    'train': A.Compose(
    [
        #A.PadIfNeeded(min_height=500, min_width=1000),
        #A.RandomCrop(500, 1000),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),]),

    'val': A.Compose(
    [   #A.PadIfNeeded(min_height=500, min_width=1000),
        #A.CenterCrop(500, 1000),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),]),
    }

    path = "/home/pedro/Documents/Datasets/bdd100k/seg/"

    trainloader = torch.utils.data.DataLoader(data_loader.Bdd100kDataset(path,mode="train",transforms=data_transforms["train"]), batch_size=1,
                                            shuffle=True, num_workers=4)

    validationloader = torch.utils.data.DataLoader(data_loader.Bdd100kDataset(path,mode="val",transforms=data_transforms["val"]), batch_size=1,
                                            shuffle=False, num_workers=2)

    optimizer = torch.optim.Adam(net.parameters(),lr=1e-2)
    print(len(trainloader))
    size = len(trainloader)
    for epoch in range(30):  # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        running_IoU= 0.0
        for i, (ipt,target) in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = ipt.cuda(),target.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 800 == 799:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f ' %
                    (epoch + 1, i + 1, running_loss / 800))
                running_loss = 0.0
                unormed = unorm(ipt[0,:,:,:])
                display_prediction(unormed,unorm(outputs[0,:,:,:].cpu().detach()),epoch,i)
                display_prediction(unormed,target[0,:,:,:],epoch,i+1)
                
        torch.save(net, "./model.pth")
        
        net.eval()
        trans = transforms.ToPILImage()
        with torch.no_grad():
            seen_images = 0
            running_loss = 0.0
            for i, (ipt,target) in enumerate(validationloader, 0):
            # get the inputs; data is a list of [inputs, labels]
                inputs, labels = ipt.cuda(),target.cuda()
                outputs = net(inputs)
                loss = dice_score(outputs, labels)
                seen_images +=ipt.shape[0]
                # print statistics
                running_loss += loss.item() *ipt.shape[0]
                if i % 600 == 0:
                    display_prediction(unorm(ipt[0,:,:,:]),unorm(outputs[0,:,:,:].cpu()),epoch,i,train=False)
                    #trans(torch.squeeze(outputs[0,:,:,:])).show()
                    #trans(torch.squeeze(labels[0,:,:,:])).show()
                #video.write(cv2.cvtColor(np.array(trans(torch.squeeze(outputs[0,:,:,:]))), cv2.COLOR_RGB2BGR)) 
            print('[%d] dice: %.3f ' %
                    (epoch + 1, running_loss / seen_images))

def criterion(y,target,epsilon=1e-6):
    cel = torch.nn.CrossEntropyLoss()
    target = torch.argmax(target,dim=1)
    return 0.6 * soft_dice(y,target,epsilon) + 0.4 * cel(y,target)

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
    true_1_hot = true_1_hot.type(y.type())
    dims = (0,) + tuple(range(2, target.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return dice_loss



class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor



def display_prediction(image,prediction,epoch,i,train=True):
   
    image = np.squeeze(image.numpy()*255).astype(np.uint8).transpose((1,2,0))
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