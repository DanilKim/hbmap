""" Full assembly of the parts to form the complete network """

from .unet_parts import *
from .cnet import CNet


class UNet_CNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_CNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.cnet = CNet(1024, 32, n_classes)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 2)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        cls_logits = self.cnet(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        seg_logits = self.outc(x)
        return cls_logits, seg_logits

    def define_params_for_optimizer(self):
        #seg_params = filter(lambda p: p.requires_grad, model.parameters())
        seg_params = [
            {'params': self.inc.parameters()},
            {'params': self.down1.parameters()},
            {'params': self.down2.parameters()},
            {'params': self.down3.parameters()},
            {'params': self.down4.parameters()},
            {'params': self.up1.parameters()},
            {'params': self.up2.parameters()},
            {'params': self.up3.parameters()},
            {'params': self.up4.parameters()},
            {'params': self.outc.parameters()},
        ]
        cls_params = [
            {'params': self.inc.parameters()},
            {'params': self.down1.parameters()},
            {'params': self.down2.parameters()},
            {'params': self.down3.parameters()},
            {'params': self.down4.parameters()},
            {'params': self.cnet.parameters(), 'lr': 1e-3, 'weigth_decay': 1e-7},
        ]
        return seg_params, cls_params