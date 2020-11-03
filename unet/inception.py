""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import torch

from .unet_parts import *


class InceptionUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(InceptionUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.block1 = InceptionConv(8, 4)
        self.block2 = InceptionConv(16, 8)
        self.block3 = InceptionConv(32, 16)
        self.block4 = InceptionConv(64, 16)

        self.inc = DoubleConv(n_channels, 8)
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
        self.down3 = Down(32, 64)
        factor = 2 if bilinear else 1
        self.down4 = Down(64, 128 // factor)
        self.up1 = UpInception(128+64, 64 // factor, bilinear)
        self.up2 = UpInception(128, 32 // factor, bilinear)
        self.up3 = UpInception(64, 16 // factor, bilinear)
        self.up4 = UpInception(32, 8, bilinear)
        self.outc = OutConv(8, n_classes)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        block1 = self.block1(x1)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)

        x = self.up1(x5, x4, block4)
        # x = torch.cat(x, block4)      
        x = self.up2(x, x3, block3)
        # x = torch.cat(x, block3)
        x = self.up3(x, x2, block2)
        # x = torch.cat(x, block2)
        x = self.up4(x, x1, block1)
        # x = torch.cat(x, block1)
        logits = self.outc(x)
        return logits
