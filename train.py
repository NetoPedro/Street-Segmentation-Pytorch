import torch 
import data_loader
import torchvision.transforms as transforms

def train(net):
    
    net.cuda()
    net.train()

    data_transforms = {
    'train': transforms.Compose([
        #transforms.RandomResizedCrop((512,512)),
        transforms.Resize((360,640)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.1, 0.1, 0.1])
    ]),
    'val': transforms.Compose([
        transforms.Resize((360,640)),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.1, 0.1, 0.1])
    ]),
    }

    path = "/home/pedro/Documents/Datasets/bdd100k/seg/"

    trainloader = torch.utils.data.DataLoader(data_loader.Bdd100kDataset(path,mode="train",transforms=data_transforms["train"]), batch_size=4,
                                            shuffle=True, num_workers=4)

    validationloader = torch.utils.data.DataLoader(data_loader.Bdd100kDataset(path,mode="val"), batch_size=1,
                                            shuffle=False, num_workers=2)

    optimizer = torch.optim.Adam(net.parameters(),lr=1e-4)
    print(len(trainloader))
    for epoch in range(30):  # loop over the dataset multiple times

        running_loss = 0.0
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
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
                torch.save(net, "./model.pth")


def soft_dice(y,target,epsilon=1e-6):
    numerator = 2. * torch.sum(y*target,dim=tuple(range(1,len(target.shape)-1)))
    denominator = torch.sum(torch.square(y) + torch.square(target),dim=tuple(range(1,len(target.shape)-1)))
    return 1 - torch.mean((numerator+epsilon)/(denominator+epsilon))
