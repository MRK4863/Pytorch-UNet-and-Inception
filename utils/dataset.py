from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        self.image_paths = []
        for id in self.ids:
            file_list = listdir(imgs_dir + id)
            self.image_paths += [ id +"/"+ filename for filename in file_list]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.image_paths)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, idx):
        img = Image.open(self.imgs_dir + self.image_paths[idx])
        
        pos =  self.image_paths[idx].find("/")
        filename = self.image_paths[idx][pos+1:]
        filename_mask = ""
        if "Cr" in filename:
            filename_mask = filename.replace("Cr", "crazing")
        elif "In" in filename:
            filename_mask = filename.replace("In", "inclusion")
        elif "Pa" in filename:
            filename_mask = filename.replace("Pa", "patches")
        elif "PS" in filename:
            filename_mask = filename.replace("PS", "pitted_surface")
        elif "RS" in filename:
            filename_mask = filename.replace("RS", "rolled-in_scale")
        elif "Sc" in filename:
            filename_mask = filename.replace("Sc", "scratches")

        filename_mask = filename_mask.replace("bmp", "png")
        mask = Image.open(self.masks_dir + filename_mask)


        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')
