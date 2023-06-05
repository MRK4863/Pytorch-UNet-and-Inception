
import argparse
import logging
import time
import copy

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import pickle
import torch.nn.functional as F


from unet import UNet
from unet import UNetMini
from unet import InceptionUNet
from dice_coeff import dice_loss

from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
from torch.optim import lr_scheduler
import torch.optim as optim

DATASET_PATH = "/home/ibrahim/Desktop/Dataset/NEU-Surface/"

# Set the train and validation directory paths
train_directory = DATASET_PATH + "train/"
valid_directory = DATASET_PATH + "val/"
test_directory = DATASET_PATH + "test/"

num_epochs = 50
batch_size = 1
dataset_train = BasicDataset(train_directory,  DATASET_PATH + "masks/")
dataset_valid = BasicDataset(valid_directory,  DATASET_PATH + "masks/")

n_val = int(len(dataset_valid))
n_train = len(dataset_train)
dataset_sizes = {
    'train':n_train,
    'valid':n_val,
}

train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

# Create iterators for data loading
dataloaders = {
    'train': train_loader,
    'valid':val_loader
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


MODEL_NAME = "UNET"
model = UNet(n_channels=1, n_classes=1, bilinear=True)
model = model.to(device)


optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
criterion = nn.BCEWithLogitsLoss()

since = time.time()

best_model_wts = copy.deepcopy(model.state_dict())
best_loss = 999
patience_counter = 0
training_loss, val_loss = [], []
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'valid']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for batch in dataloaders[phase]:
            imgs = batch['image']
            true_masks = batch['mask']  

            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.float32)   

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):

                masks_pred = model(imgs)
                loss = -1*dice_loss(true_masks, masks_pred)
                    

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * imgs.size(0)
            
        if phase == 'train':
            scheduler.step()

        epoch_loss = running_loss / dataset_sizes[phase]

        print('{} Loss: {:.4f}'.format(
            phase, epoch_loss))

        # Record training loss and accuracy for each phase
        if phase == 'train':
            training_loss.append(epoch_loss)
        else:
            val_loss.append(epoch_loss)

        # deep copy the modelmodel_ft
        if phase == 'valid' and epoch_loss < best_loss:
            patience_counter = 0
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
        elif phase=='valid':
            patience_counter +=1

    if patience_counter>=10:
        print('Training ends due to patience_counter:', patience_counter)
        break

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))
print('Best val Loss: {:4f}'.format(best_loss))



# Save the entire model
print("\nSaving the model...")
torch.save(model, MODEL_NAME + ".pth")

history = {}
history['train'] = training_loss
history['val'] = val_loss


with open(MODEL_NAME + ".history", 'wb') as file_pi:
    pickle.dump(history, file_pi)