import torch 
from torch import Tensor 
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np 
import pandas as pd 
import os 
import rasterio 

from utils.util_train import *
from model.loss import symmetric_lovasz, deep_supervision_symmetric_lovasz 
from model.metric import *
from data_loader.hbmap_dataset import HBMapDataset
from model.unet import UNet
from model.seresnext import UNET_RESNET34
from model.unext50 import UneXt50
from model.seresnext import UNET_RESNET34

from fastai.vision.all import *
from fastai.test_utils import *

bs = 64 
sz = 256 
reduce = 4
num_workers=8
TRAIN = '/data/train_images_256tile'
MASK = '/data/train_masks_256tile'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# split_layers = lambda m: [list(m.enc0.parameters())+list(m.enc1.parameters())+
#                 list(m.enc2.parameters())+list(m.enc3.parameters())+
#                 list(m.enc4.parameters()),
#                 list(m.aspp.parameters())+list(m.dec4.parameters())+
#                 list(m.dec3.parameters())+list(m.dec2.parameters())+
#                 list(m.dec1.parameters())+list(m.fpn.parameters())+
#                 list(m.final_conv.parameters())]

split_layers = lambda m: [
    list(m.conv1.parameters())+
    list(m.bn1.parameters())+
    list(m.layer1.parameters())+
    list(m.layer2.parameters())+
    list(m.layer3.parameters())+
    list(m.layer4.parameters())+
    list(m.center.parameters())+
    list(m.decoder4.parameters())+
    list(m.decoder3.parameters())+
    list(m.decoder2.parameters())+
    list(m.decoder1.parameters())+
    list(m.decoder0.parameters())+
    list(m.upsample4.parameters())+
    list(m.upsample3.parameters())+
    list(m.upsample2.parameters())+
    list(m.upsample1.parameters())+
    list(m.final_conv.parameters())
]

# split_layers = lambda m: [list(m.inc.parameters())+
#                 list(m.down1.parameters())+
#                 list(m.down2.parameters())+
#                 list(m.down3.parameters())+
#                 list(m.down4.parameters())+
#                 list(m.up1.parameters())+
#                 list(m.up2.parameters())+
#                 list(m.up3.parameters())+
#                 list(m.up4.parameters())+
#                 list(m.outc.parameters())]

def _train(args):
    dice = Dice_th_pred(np.arange(0.2, 0.7, 0.01))
    tfms = get_aug()
    dataset = HBMapDataset(TRAIN, MASK, tfms)
    t_len = (len(dataset)//10) * 8
    [t_dataset, v_dataset] = random_split(dataset, [t_len, len(dataset)-t_len])
    data = ImageDataLoaders.from_dsets(t_dataset, v_dataset,
                                        bs=bs,
                                        num_workers=num_workers,
                                        pin_memory=True).cuda()
    model = UNET_RESNET34([256,256],False).cuda()
    # model = UneXt50()
    learn = Learner(data, model, loss_func=symmetric_lovasz,
                    metrics=[Dice_soft(), Dice_th()],
                    splitter=split_layers)

    #continue training full model
    learn.fit_one_cycle(50, lr_max=slice(2e-4,2e-3),
        cbs=SaveModelCallback(monitor='dice_th',comp=np.greater))
    torch.save(learn.model.state_dict(),'/sources/model_seres_v0.pth')


def dataset_set(contain_val, args):
    from albumentations import Compose, Normalize, Resize 
    from albumentations.pytorch import ToTensorV2
    mean = np.array([0.7720342, 0.74582646, 0.76392896])
    std = np.array([0.24745085, 0.26182273, 0.25782376])
    tfms = get_aug()
    dataset = HBMapDataset(TRAIN, MASK, tfms)

    if contain_val: 
        t_len = (len(dataset)//10) * 8
        [t_dataset, v_dataset] = random_split(dataset, [t_len, len(dataset)-t_len])
        t_loader = DataLoader(t_dataset, 
                            batch_size=bs,
                            num_workers=num_workers,
                            shuffle=True,
                            drop_last=True)
        v_loader = DataLoader(v_dataset, 
                            batch_size=bs,
                            num_workers=num_workers)
        return t_loader, v_loader 
    else: 
        loader = DataLoader(dataset, 
                            batch_size=bs,
                            num_workers=num_workers,
                            shuffle=True,
                            drop_last=True)
        return loader, None

def get_aug(p=1.0):
    from albumentations import Compose, Normalize, Resize, HorizontalFlip, VerticalFlip, \
                                RandomRotate90, ShiftScaleRotate, OpticalDistortion, \
                                GridDistortion, IAAPiecewiseAffine, \
                                HueSaturationValue, CLAHE, RandomBrightnessContrast, OneOf
    from albumentations.pytorch import ToTensorV2
    import cv2
    return Compose([
        HorizontalFlip(),
        VerticalFlip(),
        RandomRotate90(),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9, 
                         border_mode=cv2.BORDER_REFLECT),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=.1),
        ], p=0.3),
        OneOf([
            HueSaturationValue(10,15,10),
            CLAHE(clip_limit=2),
            RandomBrightnessContrast(),            
        ], p=0.3),
    ], p=p)

if __name__ == '__main__':

    import argparse
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('--contain_val', default=True, type=int,
                      help='do validation with random split (default: True)')
    args = args.parse_args()
    _train(args)