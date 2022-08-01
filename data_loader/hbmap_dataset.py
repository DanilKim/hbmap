import logging
import os
import random
from pathlib import Path

from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class HBMapDataset(Dataset):
    def __init__(self, split_file: str, images_dir: str, masks_dir: str, n_class: int, tfms, augment):
        self.split_file = split_file
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.tfms = tfms
        self.norm = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.augment = augment
        if self.augment:
            self.augs = transforms.Compose([
                AddGaussianNoise(0., 0.5,)
            ])
            self.flip = transforms.RandomHorizontalFlip(p=1.0)

        with open(self.split_file, 'r') as f:
            data = [x.strip().split() for x in f.readlines()]
            self.ids, self.labels = map(list, zip(*data))

        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        cls = self.labels[idx]
        img = Image.open(os.path.join(self.images_dir, name+'.tiff'))
        mask = Image.open(os.path.join(self.masks_dir, name+'.png'))

        assert (img.width == mask.width) and (img.height == mask.height), \
            f'Image and mask {name} should be the same size, but are ({img.width}, {img.height}) and ({mask.width}, {mask.height})'

        if self.tfms:
            img = self.tfms(img)
            mask = self.tfms(mask)
        
        img = self.norm(img)
        if self.augment:
            img = self.augs(img)
            flip = random.randint(0,1)
            if flip:
                img = self.flip(img)
                mask = self.flip(mask)
            
        return {
            'image': img,
            'cls': int(cls),
            'mask': mask.squeeze(0).long(),
            'name': name
        }