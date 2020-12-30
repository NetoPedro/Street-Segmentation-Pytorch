import torch 
import data_loader


def train():

    path = "/home/pedro/Documents/Datasets/bdd100k/seg/"

    trainloader = torch.utils.data.DataLoader(data_loader.Bdd100kDataset(path,mode="train"), batch_size=8,
                                            shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(data_loader.Bdd100kDataset(path,mode="val"), batch_size=8,
                                            shuffle=False, num_workers=2)

    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

    for epoch in range(30):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = soft_dice(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

def soft_dice(y,target,epsilon=1e-6):
    numerator = 2. * torch.sum(y*target,dim=tuple(range(1,len(target.shape)-1)))
    denominator = torch.sum(torch.square(y) + torch.square(target),dim=tuple(range(1,len(target.shape)-1)))
    return 1 - torch.mean((numerator+epsilon)/(denominator+epsilon))
