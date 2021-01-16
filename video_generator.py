import cv2
import torch
import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt

import ffmpeg
import imutils
def check_rotation(path_video_file):
    # https://stackoverflow.com/questions/53097092/frame-from-video-is-upside-down-after-extracting
    # this returns meta-data of the video file in form of a dictionary
    meta_dict = ffmpeg.probe(path_video_file)
    # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
    # we are looking for
    rotate_code = None
    rotate = meta_dict.get('streams', [dict(tags=dict())])[0].get('tags', dict()).get('rotate', 0)
    return round(int(rotate) / 90.0) * 90





videos = ["/home/pedro/Documents/Datasets/bdd100k/videos/train/00d79c0a-23bea078.mov","/home/pedro/Documents/Datasets/bdd100k/videos/train/00a1176f-5121b501.mov",'/home/pedro/Documents/Datasets/bdd100k/videos/train/01a0fe55-298688cb.mov']
net = torch.load("model_149.pth").eval()
with torch.no_grad():
    transform = A.Compose(
        [   #A.PadIfNeeded(min_height=500, min_width=1000),
            #A.CenterCrop(500, 1000),
            #A.Resize(400, 650),
            #A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),])
    for i, path in enumerate(videos):
        print(i)
        vidcap = cv2.VideoCapture(path)
        out = cv2.VideoWriter('outpy'+str(i)+'.avi', cv2.VideoWriter_fourcc(*'XVID'), 30,( int(vidcap.get(4)), int(vidcap.get(3))))
        rotateCode = check_rotation(path)
        print(int(vidcap.get(3)))
        print(int(vidcap.get(4)))
        success,image = vidcap.read()
        count = 0
        while success:
            if rotateCode is not None:

                image = imutils.rotate_bound(image, rotateCode)
          
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            new_image = np.copy(image)
            transformed = transform(image=np.array(image))
            image = transformed["image"]
            
            image = image.cuda().unsqueeze(0)
            output = net(image).cpu().numpy()
            image = np.squeeze(image.cpu().numpy())
            output = np.squeeze(output)
            mask = np.argmax(output, axis=0)
            new_image[mask[:,:]==0, :] = [0,255,0]
            new_image[mask[:,:]==1, :] = [0,0,255]
            new_image[mask[:,:]==2, :] = [255,255,0]
            new_image[mask[:,:]==3, :] = [255,0,0]
            image = image.transpose((1,2,0))
            new_image = Image.blend(Image.fromarray(image, mode='RGB').convert('RGBA'),
                                        Image.fromarray(new_image, mode='RGB').convert('RGBA'),
                                        alpha=0.6)
            #print(np.array(new_image).transpose(1,0,2).shape)
            
            out.write(np.array(new_image)[:,:,:3])
            
            success,image = vidcap.read()
            #print('Read a new frame: ', success)
            count += 1


        vidcap.release()
        out.release()


