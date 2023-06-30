import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet
from unet import UNetMini
from unet import InceptionUNet
from unet import InceptionUNetMod

from utils.dataset import BasicDataset
# from torch.utils.data import DataLoader, random_split4
from torchsummary import summary

if __name__ == "__main__":
    device = "cuda"

    model = InceptionUNetMod(n_channels = 3, n_classes = 2, bilinear =False).to(device)
    print(model)
    print(summary(model, input_size = (3,512, 512)))
