import torch
from torch import nn
from torch.utils.data import Dataset
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
from patchify import patchify
from skimage import io

class Bdd100kDataset(Dataset):

    def __init__(self,path,mode="train",transforms = None,prepare_dataset = False):
        super(Bdd100kDataset,self).__init__()
            if prepare_dataset:
            aux_paths = dict()
            self.mode = mode
            self.transform = transforms
            for image in glob.glob(path+"images/"+mode+"/*"):
                aux_paths[image.split("/")[-1].split(".")[0]] = (image,None)

            if mode != "test":
                for label in glob.glob(path+"color_labels/"+mode+"/*"):
                    aux_paths[label.split("/")[-1].split("_")[0]] = (aux_paths[label.split("/")[-1].split("_")[0]][0],label)     
            aux_paths = list(aux_paths.values())
            self.set_blue = set()
            
            self.images_path= []

            for path_image, path_label in aux_paths:
                img=io.imread(path_image)
                patches_image = [img]

                label = io.imread(path_label)
                if label.shape[2] == 4:
                    label = label[:,:,:3]
                patches_label = [label]

                path_image = path_image.replace("images","images_patched").split(".")[0]
                path_label = path_label.replace("color","threshold").split(".")[0]
                for i,(patch_image,patch_label) in enumerate(zip(patches_image,patches_label)):
                    patch_label = prepare_labels(patch_label,"")
                    path_label_aux = path_label+str(i)
                    np.save(path_label_aux, patch_label) 

                    path_image_aux = path_image+str(i)
                    np.save(path_image_aux, patch_image) 
                    
                    self.images_path.append((path_image_aux + ".npy", path_label_aux + ".npy"))
            
            
            
        else:
            aux_paths = dict()
            self.mode = mode
            self.transform = transforms
            for image in glob.glob(path+"images_patched/"+mode+"/*"):
                aux_paths[image.split("/")[-1].split(".")[0]] = (image,None)

            if mode != "test":
                for label in glob.glob(path+"threshold_labels/"+mode+"/*"):
                    aux_paths[label.split("/")[-1].replace("_train_threshold","").split(".")[0]] = (aux_paths[label.split("/")[-1].replace("_train_threshold","").split(".")[0]][0],label)     
            aux_paths = list(aux_paths.values())
            self.set_blue = set()
            
            self.images_path = aux_paths
        
        
        

    def __getitem__(self, item):
        img= np.load(self.images_path[item][0])
        label = None
        if self.images_path[item][1] != None:
            path = self.images_path[item][1]
            label = np.load(path)
            
            
        if self.transform is not None:
            if not(label is None):  
                transformed = self.transform(image=img, mask=label)
                img = transformed["image"]
                label = transformed["mask"]
            else:
                transformed = self.transform(image=np.array(img))
                img = transformed["image"]
        if self.mode == "train":
            return img, torch.transpose(label,2,1).transpose(0,1)
        else:
            return img, torch.transpose(label,2,1).transpose(0,1),self.images_path[item][0]
        
    def __len__(self):
        return len(self.images_path)



labels_list = [
   
    ("road", 7, 0, "road", 1, False, False, (128, 64, 128)),
    ("sidewalk", 8, 1, "sidewalk", 1, False, False, (244, 35, 232)),
    #("person", 31, 11, "human", 6, True, False, (220, 20, 60)),
    ("rider", 32, 12, "human", 6, True, False, (255, 0, 0)),
    ("bicycle", 33, 18, "vehicle", 7, True, False, (119, 11, 32)),
    ("bus", 34, 15, "vehicle", 7, True, False, (0, 60, 100)),
    ("car", 35, 13, "vehicle", 7, True, False, (0, 0, 142)),
    ("caravan", 36, 255, "vehicle", 7, True, True, (0, 0, 90)),
    ("motorcycle", 37, 17, "vehicle", 7, True, False, (0, 0, 230)),
    ("trailer", 38, 255, "vehicle", 7, True, True, (0, 0, 110)),
    ("truck", 40, 14, "vehicle", 7, True, False, (0, 0, 70)),
]
labels_dic = {}
for _,_,_,a,_,_,_,b in labels_list: 
        labels_dic.setdefault(a, []).append(b) 

def prepare_labels(img,path):
    new_image = np.zeros((img.shape[0], img.shape[1], len(labels_dic)+1))
    new_image[:,:,0] = (img[:,:] == labels_dic["road"][0])[:,:,0]
    new_image[:,:,1] = (img[:,:] == labels_dic["sidewalk"][0])[:,:,0]
    new_image[:,:,2] = (img[:,:] == labels_dic["human"][0])[:,:,0]
    new_image[:,:,3] = np.logical_or.reduce((np.all(img[:,:] == labels_dic["vehicle"][0],axis=-1),
                                            np.all(img[:,:] == labels_dic["vehicle"][1],axis=-1),
                                            np.all(img[:,:] == labels_dic["vehicle"][2],axis=-1),
                                            np.all(img[:,:] == labels_dic["vehicle"][3],axis=-1),
                                            np.all(img[:,:] == labels_dic["vehicle"][4],axis=-1),
                                            np.all(img[:,:] == labels_dic["vehicle"][5],axis=-1),
                                            np.all(img[:,:] == labels_dic["vehicle"][6],axis=-1)))
    
    else_mask = np.logical_not(np.logical_or.reduce((new_image[:,:,3], new_image[:,:,2],
                                                     new_image[:,:,1], new_image[:,:,0])))
    new_image[:,:,4] = else_mask
   

    return new_image.astype(np.uint8)