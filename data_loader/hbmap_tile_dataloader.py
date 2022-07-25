from torchvision import transforms

from base.base_data_loader import BaseDataLoader

from .hbmap_tile_dataset import HBMapTileDataset


class HBMapTileDataLoader(BaseDataLoader):
    """
    HuBMAP Tile data loading using BaseDataLoader
    """
    def __init__(self, images_dir, masks_dir, split_file, batch_size, shuffle, tile_size, num_workers=1):
        info_csv_file = '/data/train.csv'

        tfms = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.dataset = HBMapTileDataset(split_file, images_dir, masks_dir, tile_size, info_csv_file, tfms)
        super().__init__(self.dataset, batch_size, shuffle, 0.0, num_workers)
