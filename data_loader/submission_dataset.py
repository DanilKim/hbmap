import logging
import os

from torch.utils.data import Dataset
from PIL import Image


class SubmissionDataset(Dataset):
    def __init__(self, images_dir: str, tfms):
        self.images_dir = images_dir
        self.tfms = tfms

        self.ids = os.listdir(self.images_dir)

        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        img = Image.open(os.path.join(self.images_dir, name))
        width, height = img.size

        if self.tfms:
            img = self.tfms(img)
            
        return {
            'image': img,
            'name': name.split('.')[0],
            'size': (width, height)
        }