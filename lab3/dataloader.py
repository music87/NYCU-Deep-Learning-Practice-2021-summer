import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import os 
from PIL import Image

def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    elif mode == 'test':
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)

#refer to 
#https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
#https://pytorch.org/tutorials/beginner/basics/transforms_tutorial.html
#https://pytorch.org/vision/stable/transforms.html
class RetinopathyDataset(Dataset):
    def __init__(self, root, mode):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status('train' or 'test')

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        if mode =='test':
            self.transforms=transforms.ToTensor() #without data augmentation
        elif mode == 'train':
            self.transforms=transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),transforms.ToTensor()]) #with data augmentation
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                          =>transforms.ToTensor() converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] 
                         
            step4. Return processed image and label
        """
        #step1
        path = os.path.join(self.root,self.img_name[index] + '.jpeg')
        #step2
        label = self.label[index]
        #step3
        img = Image.open(path) #np.array(img).shape is (512,512,3), value is between [0,255]
        img = self.transforms(img) #type(img) is torch.Tensor, img.shape is (3,512,512), value is between [0,1]
        #step4
        return img, label
