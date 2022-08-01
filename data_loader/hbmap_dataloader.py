import torch
from torchvision import transforms
from base.base_data_loader import BaseDataLoader
from .hbmap_dataset import HBMapDataset


class HBMapDataLoader(BaseDataLoader):
    """
    HuBMAP data loading using BaseDataLoader
    """
    def __init__(self, images_dir, masks_dir, split_file, batch_size, shuffle, augment, image_size, num_workers=1):
        if image_size:
            tfms = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ])
        else:
            tfms = transforms.Compose([
                transforms.ToTensor(),
            ])
        
        self.dataset = HBMapDataset(split_file, images_dir, masks_dir, tfms, augment)
        super().__init__(self.dataset, batch_size, shuffle, 0.0, num_workers)
