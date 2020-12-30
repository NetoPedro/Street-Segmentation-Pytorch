import torch
from torch import nn
from torch.utils.data import Dataset
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
class CityscapeDataset(Dataset):

    def __init__(self):
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass

class Bdd100kDataset(Dataset):

    def __init__(self,path,mode="train",transforms = None,):
        super(Bdd100kDataset,self).__init__()
        self.images_path = dict()
        self.transforms = transforms
        for image in glob.glob(path+"images/"+mode+"/*"):
            self.images_path[image.split("/")[-1].split(".")[0]] = (image,None)

        if mode != "test":
            for label in glob.glob(path+"color_labels/"+mode+"/*"):
                self.images_path[label.split("/")[-1].split("_")[0]] = (self.images_path[label.split("/")[-1].split("_")[0]][0],label)     
        self.images_path = list(self.images_path.values())
    def __getitem__(self, item):
        img=Image.fromarray(mpimg.imread(self.images_path[item][0]) )
        label = None
        if self.images_path[item][1] != None:
            label = Image.fromarray((mpimg.imread(self.images_path[item][1])* 255).astype(np.uint8))
            label = label.convert("RGB")
        if self.transforms != None:
            img = self.transforms(img)
            if label != None:
                
                label = self.transforms(label)
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

