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

    def __init__(self,path,mode="train",transforms = None):
        super(Bdd100kDataset,self).__init__()
        self.images_path = dict()
        self.transform = transforms
        for image in glob.glob(path+"images/"+mode+"/*"):
            self.images_path[image.split("/")[-1].split(".")[0]] = (image,None)

        if mode != "test":
            for label in glob.glob(path+"color_labels/"+mode+"/*"):
                self.images_path[label.split("/")[-1].split("_")[0]] = (self.images_path[label.split("/")[-1].split("_")[0]][0],label)     
        self.images_path = list(self.images_path.values())
        self.set_blue = set()
    def __getitem__(self, item):
        img=mpimg.imread(self.images_path[item][0])
        label = None
        if self.images_path[item][1] != None:
            
            label = (mpimg.imread(self.images_path[item][1]) * 255).astype(np.uint8)
            
            #    self.set_blue.add(pixel)
            if label.shape[2] == 4:
                label = label[:,:,:3]
            label = prepare_labels(label)
            
            #label = Image.fromarray(image)
            #label = label.convert("RGB")
        if self.transform is not None:
            if not(label is None):  
                transformed = self.transform(image=img, mask=label)
                img = transformed["image"]
                label = transformed["mask"]
            else:
                transformed = self.transform(image=img)
                img = transformed["image"]
        return img, torch.transpose(label,2,1).transpose(0,1)

    def __len__(self):
        return len(self.images_path)

dataset = {
    "cityescape": CityscapeDataset,
    "bdd100": Bdd100kDataset 
}


path = "/home/pedro/Documents/Datasets/bdd100k/seg/"
dataset = Bdd100kDataset(path)

def replace_labels(image):
    image = np.where(image[:,:] == [119, 11, 32],[0, 0, 142],image[:,:])
    image = np.where(image[:,:] == [0, 60, 100],[0, 0, 142],image[:,:])
    image = np.where(image[:,:] == [0, 0, 90],[0, 0, 142],image[:,:])
    image = np.where(image[:,:] == [0, 0, 230],[0, 0, 142],image[:,:])
    image = np.where(image[:,:] == [0, 0, 110],[0, 0, 142],image[:,:])
    image = np.where(image[:,:] == [0, 80, 100],[0, 0, 142],image[:,:])
    image = np.where(image[:,:] == [0, 0, 70],[0, 0, 142],image[:,:])
    return image


labels_list = [
    #("unlabeled", 0, 255, "void", 0, False, True, (0, 0, 0)),
    #("dynamic", 1, 255, "void", 0, False, True, (111, 74, 0)),
    #("ego vehicle", 2, 255, "void", 0, False, True, (0, 0, 0)),
    #("ground", 3, 255, "void", 0, False, True, (81, 0, 81)),
    #("static", 4, 255, "void", 0, False, True, (0, 0, 0)),
    #("parking", 5, 255, "flat", 1, False, True, (250, 170, 160)),
    #("rail track", 6, 255, "flat", 1, False, True, (230, 150, 140)),
    ("road", 7, 0, "road", 1, False, False, (128, 64, 128)),
    ("sidewalk", 8, 1, "sidewalk", 1, False, False, (244, 35, 232)),
    #("bridge", 9, 255, "construction", 2, False, True, (150, 100, 100)),
    #("building", 10, 2, "construction", 2, False, False, (70, 70, 70)),
    #("fence", 11, 4, "construction", 2, False, False, (190, 153, 153)),
    #("garage", 12, 255, "construction", 2, False, True, (180, 100, 180)),
    #(
    #    "guard rail", 13, 255, "construction", 2, False, True, (180, 165, 180)
    #),
    #("tunnel", 14, 255, "construction", 2, False, True, (150, 120, 90)),
    #("wall", 15, 3, "construction", 2, False, False, (102, 102, 156)),
    #("banner", 16, 255, "object", 3, False, True, (250, 170, 100)),
    #("billboard", 17, 255, "object", 3, False, True, (220, 220, 250)),
    #("lane divider", 18, 255, "object", 3, False, True, (255, 165, 0)),
    #("parking sign", 19, 255, "object", 3, False, False, (220, 20, 60)),
    #("pole", 20, 5, "object", 3, False, False, (153, 153, 153)),
    #("polegroup", 21, 255, "object", 3, False, True, (153, 153, 153)),
    #("street light", 22, 255, "object", 3, False, True, (220, 220, 100)),
    #("traffic cone", 23, 255, "object", 3, False, True, (255, 70, 0)),
    #(
    #    "traffic device", 24, 255, "object", 3, False, True, (220, 220, 220)
    #),
    #("traffic light", 25, 6, "object", 3, False, False, (250, 170, 30)),
    #("traffic sign", 26, 7, "object", 3, False, False, (220, 220, 0)),
    #("traffic sign frame", 27,255,"object",3,False,True,(250, 170, 250)),
    #("terrain", 28, 9, "nature", 4, False, False, (152, 251, 152)),
    #("vegetation", 29, 8, "nature", 4, False, False, (107, 142, 35)),
    #("sky", 30, 10, "sky", 5, False, False, (70, 130, 180)),
    ("person", 31, 11, "human", 6, True, False, (220, 20, 60)),
    ("rider", 32, 12, "human", 6, True, False, (255, 0, 0)),
    ("bicycle", 33, 18, "human", 7, True, False, (119, 11, 32)),
    ("bus", 34, 15, "vehicle", 7, True, False, (0, 60, 100)),
    ("car", 35, 13, "vehicle", 7, True, False, (0, 0, 142)),
    ("caravan", 36, 255, "vehicle", 7, True, True, (0, 0, 90)),
    ("motorcycle", 37, 17, "vehicle", 7, True, False, (0, 0, 230)),
    ("trailer", 38, 255, "vehicle", 7, True, True, (0, 0, 110)),
    ("train", 39, 16, "vehicle", 7, True, False, (0, 80, 100)),
    ("truck", 40, 14, "vehicle", 7, True, False, (0, 0, 70)),
]
labels_dic = {}
for _,_,_,a,_,_,_,b in labels_list: 
        labels_dic.setdefault(a, []).append(b) 

def prepare_labels(img):
    new_image = np.zeros((img.shape[0], img.shape[1], len(labels_dic)+1))
    
    new_image[:,:,0] = (img[:,:] == labels_dic["road"][0])[:,:,0]
    new_image[:,:,1] = (img[:,:] == labels_dic["sidewalk"][0])[:,:,0]
    new_image[:,:,2] = np.logical_or.reduce((img[:,:] == labels_dic["human"][0],
                                            img[:,:] == labels_dic["human"][1],
                                            img[:,:] == labels_dic["human"][2]))[:,:,0]
    new_image[:,:,3] = np.logical_or.reduce((img[:,:] == labels_dic["vehicle"][0],
                                            img[:,:] == labels_dic["vehicle"][1],
                                            img[:,:] == labels_dic["vehicle"][2],
                                            img[:,:] == labels_dic["vehicle"][3],
                                            img[:,:] == labels_dic["vehicle"][4],
                                            img[:,:] == labels_dic["vehicle"][5],
                                            img[:,:] == labels_dic["vehicle"][6]))[:,:,0]
    
    else_mask = np.logical_not(np.logical_or.reduce((new_image[:,:,3], new_image[:,:,2],
                                                     new_image[:,:,1], new_image[:,:,0])))
    new_image[:,:,4] = else_mask
    
    return new_image.astype(np.float32)