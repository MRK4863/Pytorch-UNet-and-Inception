import torch
from torch import Tensor


def dice_loss(target, pred):
    smooth = 1.

    iflat = torch.flatten(pred.contiguous(), 1)
    tflat = torch.flatten(target.contiguous(), 1)
    intersection = (iflat * tflat).sum(1)
    A_sum = iflat.sum(1)
    B_sum = tflat.sum(1)
    
    loss = ((2. * intersection + smooth) / (A_sum + B_sum + smooth)).mean()
    return loss