from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import random
import logging
import numpy as np
import cv2 
import torch
from os import path
from PIL import Image
from PIL import ImageEnhance
import albumentations as A
import torchvision.transforms as TT
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torchvision.transforms import functional as TF
from torchvision.datasets.utils import list_files
import torchvision.transforms as transforms

channel_means = [0.536, 0.529, 0.521]
channel_stds  = [0.129, 0.127, 0.125]

def Bextraction(img):
    img = img[0].numpy()
    img1 = img.astype(np.uint8)
    DIAMOND_KERNEL_5 = np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ], dtype=np.uint8)
    img2 =  cv2.dilate(img1, DIAMOND_KERNEL_5).astype(img.dtype)
    img3 = img2 - img
    img3 = np.expand_dims(img3, axis = 0)
    return torch.tensor(img3.copy())

class ImgToTensor(object):
    def __call__(self, img):
        tf = transforms.Compose([transforms.ToTensor(),                                                                                    
                                 transforms.Normalize(channel_means, channel_stds)])
        return tf(img)

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(img).long()

class CrackDataset(Dataset):
    def __init__(self, data_root, split_mode):
        self.data_root = data_root#Dataset root directory.
        self.split_mode = split_mode#Training, validation, or test split.
        self.transforms = transforms
        self.num_return = 2

        self.logger = logging.getLogger('CamCrack810 Dataset')
        self.logger.debug('split_mode is set to %s', self.split_mode)#Log the selected split.

        self.img_totensor  = ImgToTensor()
        self.mask_totensor = MaskToTensor()
        
        if self.split_mode == 'train':
            train_data_root = path.join(self.data_root,  'train')
            self.dataset = CrackPILDataset(data_root=train_data_root)
        elif self.split_mode == 'valid':
            valid_data_root = path.join(self.data_root, 'val')
            self.dataset = CrackPILDataset(data_root=valid_data_root)
        elif self.split_mode == 'test':
            test_data_root = path.join(self.data_root, 'test')
            self.dataset = CrackPILDataset(data_root=test_data_root) 
        else:
            self.logger.error('split_mode must be either "train" or "valid"')
            raise NotImplementedError
    
    def read_bbox_from_txt(self, bbox_path):
        """
        Read bounding boxes from a text file.
        Each line has the format: x_min y_min width height.
        """
        bboxes = []
        with open(bbox_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                x_min = int(parts[0])
                y_min = int(parts[1])
                width = int(parts[2])
                height = int(parts[3])

                # Compute x_max and y_max.
                x_max = x_min + width
                y_max = y_min + height
                bboxes.append([x_min, y_min, x_max, y_max])
        
        return torch.tensor(bboxes, dtype=torch.float32)

    def __getitem__(self, index):
        img, mask = self.dataset[index]  # The underlying dataset returns an image and a mask.
        if img.shape[2] == 4:  
            img = img[:, :, :3]  # Keep only the RGB channels.
        image=TF.to_tensor(img)
        annot=TF.to_tensor(mask)
        
        img_filename = self.dataset.get_image_filename(index)
        bbox_filename = img_filename.replace('.jpg', '.txt').replace('.png', '.txt')  # The extension replacement could be made more robust.
        if self.split_mode == 'train':
            bbox_path = os.path.join(self.data_root, 'train/box', bbox_filename)
        elif self.split_mode == 'valid':
            bbox_path = os.path.join(self.data_root, 'val/box', bbox_filename)
        elif self.split_mode == 'test':
            bbox_path = os.path.join(self.data_root, 'test/box', bbox_filename)

        bboxes = self.read_bbox_from_txt(bbox_path)  # Read the bounding box.
        
        img = img.astype(np.uint8)  # Convert to uint8.
        mask = mask.astype(np.uint8)  # Convert to uint8.
        # Convert to tensors.
        img = self.img_totensor(Image.fromarray(img.copy()))
        mask = self.mask_totensor(mask.copy()).unsqueeze(0)
        # Extract the boundary.
        boundary = Bextraction(mask)
        return {
            'image': image,
            'mask': annot,
            'boundary': boundary,
            'bboxes':bboxes
        }

    def __len__(self):
        return len(self.dataset)

    

class CrackPILDataset(Dataset):
    def __init__(self, data_root,mode = 0):
        self.data_root = path.expanduser(data_root)
        self._image_dir = path.join(self.data_root, 'image')#Path to the image directory.
        self._annot_dir = path.join(self.data_root, 'label')
        self._image_paths = sorted(list_files(self._image_dir,suffix=('.png', '.PNG','.jpg','.JPG'),prefix=True))#Collect and sort image paths (for example, 001.jpg).
        self._annot_paths = sorted(list_files(self._annot_dir,suffix=('.png', '.PNG','.jpg','.JPG'),prefix=True))#Collect and sort mask paths (for example, 001.png).
        self.mode = mode
        assert len(self._image_paths) == len(
            self._annot_paths), 'Crack dataset corrupted'#Ensure the numbers of images and masks match.

    def __getitem__(self, index):#Open the image and mask.
        if self.mode == 0:
            image = Image.open(self._image_paths[index])
            annot = Image.open(self._annot_paths[index], mode='r').convert('L')
            
            image=np.array(image)
            image=image/255
            annot=np.array(annot)
            annot=annot/255
            return image, annot
        else:
            image = Image.open(self._image_paths[index])
            annot = Image.open(self._annot_paths[index])
            image=np.array(image)
            image=image/255
            annot=np.array(annot)
            annot=annot/255
            
            return image, annot
        
    def get_image_filename(self, index):
        return os.path.basename(self._image_paths[index])

    def __len__(self):
        return len(self._image_paths)

