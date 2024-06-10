from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import numpy as np
import random 

from PIL import Image, ImageFile
from data.utils import StaticColorJitter

tf_list = []    
tf_list.append(transforms.ToTensor())    


#     tf_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])) #imagenet values
tf_list.append(transforms.Normalize([0.47264798, 0.47641314, 0.46798028], [0.07805742, 0.0770264 , 0.08050214])) #imagenet values
tf = transforms.Compose(tf_list)

class VizDoomData(Dataset):
    def __init__(self, data, istest = False, augvars = {"RAND_EQ": False,
                                                        "RAND_AC": False,
                                                        "RAND_GN": False,
                                                        "RAND_SIG": 5.0}):
        self.data = data 
        self.TEST = istest
        self.av = augvars
    def __len__(self):
        return len(self.data)

    def gauss_noise_tensor(self, img, rand_num):
        assert isinstance(img, torch.Tensor)
        
        if rand_num < 0.5:
            return img
        
        dtype = img.dtype
        if not img.is_floating_point():
            img = img.to(torch.float32)
        
        sigma = self.av["RAND_SIG"]
        
        out = img + sigma * torch.randn_like(img)
        
        if out.dtype != dtype:
            out = out.to(dtype)
            
        return out

    def load_img(self, img_path, color_transform):       
        ppad = 56 
        border = (220-ppad, 160-ppad, 444+ppad, 384+ppad) # left, up, right, bottom [224x224]
        img = Image.open(img_path).crop(border)  
        img = img.resize((224,224))
        img = color_transform(img)

        if not self.TEST:         
            if self.av["RAND_EQ"]:
                img = self.re(img, self.rr)
            if self.av["RAND_AC"]:
                img = self.ra(img, self.rr)
            

        imgarr = tf(img)

        if not self.TEST and "depth" not in img_path:            
            if self.av["RAND_GN"]:
                imgarr = self.gauss_noise_tensor(imgarr, self.rr)

        
        img.close()
        return imgarr


    def __getitem__(self,index):
        imgs = self.data.iloc[index]["img_path"]
        actions = self.data.iloc[index]["img_actions"]
        
        if not self.TEST:
            color_jitter = StaticColorJitter(brightness= 0.2, 
                                          contrast=0.2, 
                                          saturation=0.2, 
                                          hue=0.2)
            
            color_transform = StaticColorJitter.get_params(color_jitter.brightness, color_jitter.contrast, color_jitter.saturation,
                color_jitter.hue)

        img_batch = torch.stack([self.load_img(idx, color_transform) for idx in imgs])
        img_batch = img_batch.reshape(3, 35, 3, 224,224) #secs, frames, channels, h, w 
        #TODO PARAMETERIZE THIS LATER!!!
        actions = torch.tensor(actions).reshape(3,35,7)

        return img_batch , actions