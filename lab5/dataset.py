import json
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import os
import numpy as np

def get_iCLEVR_data(root_folder,mode):
    if mode == 'train':
        data = json.load(open(os.path.join(root_folder,'train.json')))
        obj = json.load(open(os.path.join(root_folder,'objects.json')))
        img = list(data.keys())
        cond = list(data.values())
        for i in range(len(cond)): # for i-th image
            for j in range(len(cond[i])): # for j-th cond in i-th image
                cond[i][j] = obj[cond[i][j]]
            # one-hot encodding
            tmp = np.zeros(len(obj))
            tmp[cond[i]] = 1
            cond[i] = tmp
        return np.squeeze(img), np.squeeze(cond), len(obj)
    else:
        data = json.load(open(os.path.join(root_folder,'test.json')))
        obj = json.load(open(os.path.join(root_folder,'objects.json')))
        cond = data
        for i in range(len(cond)):
            for j in range(len(cond[i])):
                cond[i][j] = obj[cond[i][j]]
            tmp = np.zeros(len(obj))
            tmp[cond[i]] = 1
            cond[i] = tmp
        return None, cond, len(obj)


class ICLEVRDataset(data.Dataset):
    def __init__(self, root_folder, mode, image_side_length=None):
        self.root_folder = root_folder
        self.mode = mode
        self.img_list, self.cond_list, self.condition_size  = get_iCLEVR_data(root_folder,mode) # self.condition_size = 24
        if self.mode == 'train':
            self.transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), transforms.Resize((image_side_length, image_side_length))])
            print("> Found %d images..." % (len(self.img_list)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.cond_list)

    def __getitem__(self, index):
        if self.mode == 'train':
            path = os.path.join(self.root_folder, 'images', self.img_list[index])
            cond = self.cond_list[index]
            img = Image.open(path) # np.array(img).shape is (240,320,4), value is between [0,255]
            img = img.convert('RGB') # remove fourth channel which is the alpha channel controlling the opaque value, np.array(img).shape is (240,320,3), value is between [0,255]
            img = self.transforms(img) # np.array(img).shape is (3,64,64)
            return img, torch.Tensor(cond)
        elif self.mode == 'test':
            cond = self.cond_list[index]
            return torch.Tensor(cond)


