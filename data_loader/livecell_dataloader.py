from torchvision import transforms
import torch

from base.base_data_loader import BaseDataLoader

from .livecell_dataset import LIVECellDataset


class LIVECellDataLoader(BaseDataLoader):
    """
    LIVECell data loading using BaseDataLoader
    """
    def __init__(self, images_dir, masks_dir, split_file, batch_size, shuffle, image_size, num_workers=1):
        if image_size:
            tfms = transforms.Compose([
                transforms.Resize([image_size, image_size]),
                transforms.ToTensor(),
            ])
        else:
            tfms = transforms.Compose([
                transforms.ToTensor(),
            ])

        self.dataset = LIVECellDataset(split_file, images_dir, masks_dir, tfms)
        super().__init__(self.dataset, batch_size, shuffle, 0.0, num_workers)
