import torch
from torch import nn
from torch.utils.data import Dataset
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

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
        img=mpimg.imread(self.images_path[item][0]).transpose((2,0,1))
        label = None
        if self.images_path[item][1] != None:
            label = mpimg.imread(self.images_path[item][1]).transpose((2,0,1))
            label = torch.nn.functional.one_hot(label,40)

        #TODO data augmentation
        return img, label

    def __len__(self):
        return len(self.images_path)

dataset = {
    "cityescape": CityscapeDataset,
    "bdd100": Bdd100kDataset 
}


path = "/home/pedro/Documents/Datasets/bdd100k/seg/"
dataset = Bdd100kDataset(path)

