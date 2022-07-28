import logging
import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class HBMapDataset(Dataset):
    def __init__(self, split_file: str, images_dir: str, masks_dir: str, tfms):
        self.split_file = split_file
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.tfms = tfms

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
            
        return {
            'image': img,
            'cls': cls,
            'mask': mask.squeeze(0).long(),
            'name': name
        }