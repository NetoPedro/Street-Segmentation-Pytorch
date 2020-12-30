import torch 
import data_loader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageDraw
import cv2
import matplotlib.pyplot as plt
def train(net):
    
    net.cuda()
    net.train()

    data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((720,1280)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize([0.5, 0.5, 0.5], [0.1, 0.1, 0.1])
    ]),
    'val': transforms.Compose([
        transforms.Resize((720,1280)),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        #transforms.Normalize([0.5, 0.5, 0.5], [0.1, 0.1, 0.1])
    ]),
    }

    path = "/home/pedro/Documents/Datasets/bdd100k/seg/"

    trainloader = torch.utils.data.DataLoader(data_loader.Bdd100kDataset(path,mode="train",transforms=data_transforms["train"]), batch_size=1,
                                            shuffle=True, num_workers=1)

    validationloader = torch.utils.data.DataLoader(data_loader.Bdd100kDataset(path,mode="val",transforms=data_transforms["val"]), batch_size=1,
                                            shuffle=False, num_workers=2)

    optimizer = torch.optim.Adam(net.parameters(),lr=1e-3)
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
            loss = soft_dice(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f ' %
                    (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
                display_prediction(ipt[0,:,:,:],outputs[0,:,:,:].cpu().detach(),epoch,i)
                
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
                if i % 200 == 0:
                    display_prediction(ipt[0,:,:,:],outputs[0,:,:,:].cpu(),epoch,i,train=False)
                    #trans(torch.squeeze(outputs[0,:,:,:])).show()
                    #trans(torch.squeeze(labels[0,:,:,:])).show()
                #video.write(cv2.cvtColor(np.array(trans(torch.squeeze(outputs[0,:,:,:]))), cv2.COLOR_RGB2BGR)) 
            print('[%d] dice: %.3f ' %
                    (epoch + 1, running_loss / seen_images))

def criterion(y,target,epsilon=1e-6):
    cel = nn.CrossEntropyLoss()
    soft_dice(y,target,epsilon) * cel(y,target)

def soft_dice(y,target,epsilon=1e-6):
    numerator = 2. * torch.sum(y*target,dim=tuple(range(1,len(target.shape))))
    denominator = torch.sum(torch.square(y) + torch.square(target),dim=tuple(range(1,len(target.shape))))
    return 1 - torch.mean((numerator+epsilon)/(denominator+epsilon))


def dice_score(y,target,epsilon=1e-6):
    numerator = 2. * torch.sum(y*target,dim=tuple(range(1,len(target.shape))))
    denominator = torch.sum(torch.square(y) + torch.square(target),dim=tuple(range(1,len(target.shape))))
    return torch.mean((numerator+epsilon)/(denominator+epsilon))




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