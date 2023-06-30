""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import torch
import torch.nn as nn

from .unet_parts import *


class InceptionUNetMod(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(InceptionUNetMod, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        """ Down sample UNET"""
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DownInception(64, 128)
        self.down2 = DownInception(128, 256)
        self.down3 = DownInception(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = DownInception(512, 1024 // factor)

        """Upsample UNET"""
        self.up1 = UpInception(1024+512, 256 // factor, bilinear)
        self.up2 = UpInception(896, 128 // factor, bilinear)
        self.up3 = UpInception(448, 32 // factor, bilinear)
        self.up4 = UpInception(208, 16, bilinear)
        self.outc = OutConv(16, n_classes)

        """ Segmentation overlap on input image """
        self.mul_avgpool = nn.AdaptiveAveragePool2d((256, 256))

        """Intermediate Chain upsample conv"""
        self.up1 = nn.Sequential(nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2),
                                DoubleConv(in_channels, out_channels))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2),
                                DoubleConv(in_channels, out_channels))


    def forward(self, x):
        print("x", x.shape)
        x1 = self.inc(x)
        print("x1",x1.shape)
        x2 = self.down1(x1)
        print("x2",x2.shape)
        x3 = self.down2(x2)
        print("x3",x3.shape)
        x4 = self.down3(x3)
        print("x4",x4.shape)
        x5 = self.down4(x4)
        print("x5",x5.shape)

        x = self.up1(x5, x4, block4)
        # x = torch.cat(x, block4)      
        x = self.up2(x, x3, block3)
        # x = torch.cat(x, block3)
        x = self.up3(x, x2, block2)
        # x = torch.cat(x, block2)
        x = self.up4(x, x1, block1)
        # x = torch.cat(x, block1)
        x = self.outc(x)
        x_segment = torch.sigmoid(x)

        """ for segmented mask layer """
        x_mul = x_segment * torch.unsqueeze(x_segment, axis = 1)
        x_mul_avgpool = self.mul_avgpool(x_mul)
        
        """ for intermediate upsample layer """
        return x_segment
