import logging
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class HBMapDataset(Dataset):
    def __init__(self, split, split_file: str, images_dir: str, masks_dir: str, tfms, mask_tfms):
        self.split = split
        self.split_file = split_file
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.tfms = tfms
        self.mask_tfms = mask_tfms

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

        assert img.shape[:2] == mask.shape, \
            f'Image and mask {name} should be the same size, but are {img.shape} and {mask.shape}'

        if self.tfms:
            img = self.tfms(img)
        if self.mask_tfms:
            mask = self.mask_tfms(mask)

        return {
            'image': img,
            'mask': mask,
            'name': name
        }