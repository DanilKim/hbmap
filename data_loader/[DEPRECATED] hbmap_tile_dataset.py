import logging
import os
from pathlib import Path
import math

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class HBMapTileDataset(Dataset):
    def __init__(self, split, split_file: str, images_dir: str, masks_dir: str, tile_size: int, info_csv_file: str, tfms):
        self.split = split
        self.split_file = split_file
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.tile_size = tile_size
        self.tfms = tfms

        self.ids = []
        df = pd.read_csv(info_csv_file)
        with open(self.split_file, 'r') as f:
            ids_raw = [x.strip() for x in f.readlines()]
            for id in ids_raw:
                data = df.loc[df.id==int(id)]
                w, h = data['img_width'].values[0], data['img_height'].values[0]
                n_tile_w = math.ceil(w / tile_size)
                n_tile_h = math.ceil(h / tile_size)
                self.ids += [(id, tid_w, tid_h) for tid_w in range(n_tile_w) for tid_h in range(n_tile_h)]

        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name, tid_w, tid_h = self.ids[idx]
        img = Image.open(os.path.join(self.images_dir, name+'.tiff'))
        mask = Image.open(os.path.join(self.masks_dir, name+'.png'))

        assert (img.width == mask.width) and (img.height == mask.height), \
            f'Image and mask {name} should be the same size, but are ({img.width}, {img.height}) and ({mask.width}, {mask.height})'

        # tile
        img = self.tile_image(img, tid_w, tid_h)
        mask = self.tile_image(mask, tid_w, tid_h)

        if self.tfms:
            img = self.tfms(img)
            mask = self.tfms(mask)
        
        return {
            'image': img,
            'mask': mask.squeeze(0).long(),
            'name': name
        }
    
    def tile_image(self, img: Image, tid_w: int, tid_h: int) -> Image:
        img = np.array(img)
        img = img[tid_h*self.tile_size:(tid_h+1)*self.tile_size, tid_w*self.tile_size:(tid_w+1)*self.tile_size, ...]
        if img.shape[:2] != (self.tile_size, self.tile_size):
            img_ = np.copy(img)
            if len(img.shape)==3:
                img = np.zeros([self.tile_size, self.tile_size, img.shape[2]], dtype=img.dtype)
                img[:img_.shape[0], :img_.shape[1], ...] = img_
            elif len(img.shape)==2:
                img = np.zeros([self.tile_size, self.tile_size], dtype=img.dtype)
                img[:img_.shape[0], :img_.shape[1]] = img_
        return Image.fromarray(img)
