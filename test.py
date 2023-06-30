
import cv2
import numpy as np
import torch

from utils.dataset import BasicDataset
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

def find_metrics(true_values, estimated):
    
    true_values[true_values>0] = 255
    estimated[estimated>0] = 255
    confusion = confusion_matrix(true_values.ravel(),estimated.ravel())
    if confusion.shape[0] == 1 and confusion.shape[1] == 1:
        return 1,1,1
    
    tn = confusion[0,0]
    fn = confusion[1,0]
    tp = confusion[1,1]
    fp = confusion[0,1]
    
    if np.sum(true_values) == 0 and np.sum(estimated) == 0:
        return 1,1,1

    if 0 == tp:
        return 0,0,0

    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    f1 = (2*precision*recall) / (precision+recall)
    
    return precision,recall,f1



DATASET_PATH = "/home/ibrahim/Desktop/Dataset/NEU-Surface/"


test_directory = DATASET_PATH + "test/"

batch_size = 1
dataset_test = BasicDataset(test_directory,  DATASET_PATH + "masks/")

n_test = int(len(dataset_test))
test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "UNET"

# Model class must be defined somewhere
model = torch.load(MODEL_NAME + ".pth")
model.eval()

model = model.to(device)

precision, recall, f1 = [], [], []
counter = 0

for i, (batch) in enumerate(test_loader):
    img_input = batch['image']
    mask = batch['mask']  

    img_input = img_input.to(device=device, dtype=torch.float32)

    output = model(img_input)

    img = img_input.cpu().detach().numpy()[0]
    img = np.swapaxes(img,0,2)
    img = np.swapaxes(img,0,1)
    # img[:,:,0] = img[:,:,2]
    # print("max: ", np.max(img))
    # img = (img*255).astype(np.uint8)

    pred = output.cpu().detach().numpy()[0][0]
    mask = mask.detach().numpy()[0][0]

    pred[pred>=0.5] = 1
    pred[pred<0.5] = 0
    pred = pred.astype(np.uint8)

    pr, r, f = find_metrics(mask, pred)
    precision.append(pr)
    recall.append(r)
    f1.append(f)

    # print("precision: ", pr)
    # print("recall: ", r)
    # print("f1: ", f)

    counter = counter + 1
    cv2.imshow("img", img)
    cv2.imshow("mask", mask)
    cv2.imshow("pred", pred)

    cv2.waitKey(1)


print("*********\ncount: ", counter)
print("ortalama precision: ", np.mean(precision))
print("ortalama recall: ", np.mean(recall))
print("ortalama f1: ", np.mean(f1))