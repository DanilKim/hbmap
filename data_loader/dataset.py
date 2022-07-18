import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
import math
from PIL import Image
from torch.utils.data import Dataset
from utils.util import get_tile_bbox


class HBMapDataset(Dataset):
    def __init__(self, split, images_dir: str, masks_dir: str, image_size: int, tile_size: int, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size
        self.tile_size = tile_size
        self.split = split
        self.mask_suffix = mask_suffix

        if self.split == 'train':
            self.tile_hw = math.ceil( image_size * 2 / tile_size )
        else:
            self.tile_hw = math.ceil( image_size / tile_size )
        self.tile_num = self.tile_hw * self.tile_hw

        self.ids = [
            (splitext(file)[0], tid) for file in listdir(images_dir) if not file.startswith('.')
            for tid in range(self.tile_num)
        ]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(img, scale, is_mask):
        #w, h = img.shape
        #newW, newH = int(scale * w), int(scale * h)
        #assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        #img = img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        ##img_ndarray = np.asarray(img)
        img_ndarray = img

        if img_ndarray.ndim == 2:
            img_ndarray = img_ndarray[np.newaxis, ...]
        else:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_mask:
            img_ndarray = img_ndarray / 255

        return img_ndarray

    @staticmethod
    def load_image(filename, tid, image_size, tile_size, overlap=True):
        ext = splitext(filename)[1]
        if ext == '.npy':
            img = np.load(filename)
        elif ext in ['.pt', '.pth']:
            img = torch.load(filename).numpy()
        else:
            img = np.asarray(Image.open(filename))

        if img.shape[0] < image_size:
            pw = math.ceil((image_size - img.shape[0]) / 2)
            img = np.pad(img, ((pw, pw), (pw, pw), (0, 0)), mode='reflect')
        elif img.shape[0] > image_size:
            print(img.shape[0])

        y1, y2, x1, x2 = get_tile_bbox(image_size, tile_size, tid, overlap)
        return img[ y1 : y2, x1 : x2, : ]

    @staticmethod
    def load_mask(filename, tid, image_size, tile_size, overlap=True):
        ext = splitext(filename)[1]
        if ext == '.npy':
            img = np.load(filename)
        elif ext in ['.pt', '.pth']:
            img = torch.load(filename).numpy()
        else:
            img = np.asarray(Image.open(filename))
        
        if img.shape[0] < image_size:
            pw = math.ceil((image_size - img.shape[0]) / 2)
            img = np.pad(img, pw, mode='reflect')
        elif img.shape[0] > image_size:
            print(img.shape[0])

        y1, y2, x1, x2 = get_tile_bbox(image_size, tile_size, tid, overlap)
        return img[ y1 : y2, x1 : x2 ]


    def __getitem__(self, idx):
        name, tid = self.ids[idx]
        mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = self.load_mask(mask_file[0], tid, self.image_size, self.tile_size, self.split=='train')
        img = self.load_image(img_file[0], tid, self.image_size, self.tile_size, self.split=='train')

        assert img.shape[:2] == mask.shape, \
            f'Image and mask {name} should be the same size, but are {img.shape} and {mask.shape}'

        img = self.preprocess(img, 1, is_mask=False)
        mask = self.preprocess(mask, 1, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).float().contiguous(),
            'name': name,
            'tile_id': tid
        }