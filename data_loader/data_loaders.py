from torchvision import transforms
from base import BaseDataLoader

from .hbmap_dataset import HBMapDataset


class HBMapDataLoader(BaseDataLoader):
    """
    HuBMAP data loading using BaseDataLoader
    """
    def __init__(self, batch_size, split, shuffle, num_workers=1):
        images_dir = '/data/train_images'
        masks_dir = '/data/train_masks'
        split_file = f'/data/{split}.txt'

        self.dataset = HBMapDataset(split, split_file, images_dir, masks_dir)
        super().__init__(self.dataset, batch_size, shuffle, 0.0, num_workers)
