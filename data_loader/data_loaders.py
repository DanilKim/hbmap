from torchvision import datasets, transforms
from base import BaseDataLoader
from .dataset import HBMapDataset


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class HBMapDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, batch_size, image_size, tile_size, split='train', num_workers=1):
        images_dir = '/data/' + split + '_split_images'
        masks_dir = '/data/' + split + '_split_masks'
        shuffle = split == 'train'
        self.dataset = HBMapDataset(split, images_dir, masks_dir, image_size, tile_size)
        super().__init__(self.dataset, batch_size, shuffle, 0.0, num_workers)
