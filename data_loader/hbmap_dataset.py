import logging
import os
import random
from pathlib import Path

from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset


class HBMapDataset(Dataset):
    def __init__(self, split_file: str, images_dir: str, masks_dir: str, tfms, augs):
        self.split_file = split_file
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.tfms = tfms
        self.augs = augs
        if self.augs:
            self.flip = transforms.RandomHorizontalFlip(p=1.0)

        with open(self.split_file, 'r') as f:
            self.ids = [x.strip() for x in f.readlines()]

        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        img = Image.open(os.path.join(self.images_dir, name+'.tiff'))
        mask = Image.open(os.path.join(self.masks_dir, name+'.png'))

        assert (img.width == mask.width) and (img.height == mask.height), \
            f'Image and mask {name} should be the same size, but are ({img.width}, {img.height}) and ({mask.width}, {mask.height})'

        if self.tfms:
            img = self.tfms(img)
            mask = self.tfms(mask)
        
        if self.augs:
            img = self.augs(img)
            flip = random.randint(0,1)
            if flip:
                img = self.flip(img)
                mask = self.flip(mask)
            
        return {
            'image': img,
            'mask': mask.squeeze(0).long(),
            'name': name
        }