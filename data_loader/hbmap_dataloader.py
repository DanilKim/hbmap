import torch
from torchvision import transforms
from base.base_data_loader import BaseDataLoader
from .hbmap_dataset import HBMapDataset


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


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
        
        augs = None
        if augment:
            augs = transforms.Compose([
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                AddGaussianNoise(0., 0.5,)
            ])

        self.dataset = HBMapDataset(split_file, images_dir, masks_dir, tfms, augs)
        super().__init__(self.dataset, batch_size, shuffle, 0.0, num_workers)
