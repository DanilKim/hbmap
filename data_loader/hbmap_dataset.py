import logging
import os
from pathlib import Path
from glob import glob
from PIL import Image
import numpy as np 
import cv2

from torch.utils.data import Dataset
import torch 

mean = np.array([0.77455656, 0.74896831, 0.76683053])
std = np.array([0.25168728, 0.2655022 , 0.26106301])

def img2tensor(img,dtype:np.dtype=np.float32):
    if img.ndim==2 : img = np.expand_dims(img,2)
    img = np.transpose(img,(2,0,1))
    return torch.from_numpy(img.astype(dtype, copy=False))

class HBMapDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, tfms):
        # self.split_file = split_file
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.tfms = tfms

        self.names = [x.split('/')[-1] for x in glob(images_dir + '/*')]
        # with open(self.split_file, 'r') as f:
        #     self.ids = [x.strip() for x in f.readlines()]

        if not self.names:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.names)} examples')

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img = cv2.cvtColor(cv2.imread(os.path.join(self.images_dir, name)), 
                            cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.masks_dir, name), cv2.IMREAD_GRAYSCALE)
        # img = Image.open(os.path.join(self.images_dir, name))
        # mask = Image.open(os.path.join(self.masks_dir, name))

        # assert (img.width == mask.width) and (img.height == mask.height), \
        #     f'Image and mask {name} should be the same size, but are ({img.width}, {img.height}) and ({mask.width}, {mask.height})'

        if self.tfms:
            batch_data = self.tfms(image=img, mask=mask)
            img = batch_data['image']
            mask = batch_data['mask']
            # mask = self.tfms(np.array(mask))
        # return {
        #     'image': batch_data['image'],
        #     'mask': batch_data['mask'].long(),
        #     # 'mask': mask.squeeze(0).long(),
        #     'name': name
        # }
        # return {
        #     'image': img2tensor((img/255.0 - mean)/std),
        #     'mask' : img2tensor(mask)
        # }
        return img2tensor((img/255.0 - mean)/std), img2tensor(mask)