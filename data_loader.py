import torch
from torch import nn
from torch.utils.data import Dataset
import glob



class CityscapeDataset(Dataset):

    def __init__(self):
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass

class Bdd100kDataset(Dataset):

    def __init__(self,path,mode="train"):
        self.images_path = dict()
        for image in glob.glob(path+"images/"+mode+"/*"):
            self.images_path[image.split("/")[-1].split(".")[0]] = (image,None)

        if mode != "test":
            for label in glob.glob(path+"color_labels/"+mode+"/*"):
                self.images_path[label.split("/")[-1].split("_")[0]] = (self.images_path[label.split("/")[-1].split("_")[0]][0],label)     
        self.images_path = list(self.images_path.values())
    def __getitem__(self, item):
        return self.images_path[item]

    def __len__(self):
        return len(self.images_path)

dataset = {
    "cityescape": CityscapeDataset,
    "bdd100": Bdd100kDataset 
}


path = "/home/pedro/Documents/Datasets/bdd100k/seg/"
dataset = Bdd100kDataset(path)
print(len(dataset))
print(dataset[1])
dataset = Bdd100kDataset(path,"val")
print(len(dataset))
print(dataset[1])
dataset = Bdd100kDataset(path,"test")
print(len(dataset))
print(dataset[1])
