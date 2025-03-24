import numpy as np
import torch
import os
import cv2
from torchvision.io import read_image

'''
# downsample and crop to 64 x 64 to create low_res counterpart
def lowRes(image_path, target_path):
    hr_image = cv2.imread(image_path)
    height, width = hr_image.shape[:2]

    if height > 128 or width > 128:
        hr_image = hr_image[0:128, 0:128]

    lr_dim = (int(width * 0.5), int(height * 0.5))
    print(hr_image.shape, lr_dim)
    lr_image = cv2.resize(hr_image, lr_dim, interpolation = cv2.INTER_AREA)

    
    cv2.imwrite(image_path, hr_image)
    cv2.imwrite(target_path, lr_image)

root_dir = 'my-supa-res/data-64x64-128x128/high_res'
target_dir = 'my-supa-res/data-64x64-128x128/low_res'

for filename in os.listdir(root_dir):
    if filename.endswith(".png"):
        lowRes(os.path.join(root_dir, filename), os.path.join(target_dir, filename))
 '''


# Create dataset object for custom paired dataset
class PairedImageDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform_lr=None, transform_hr=None):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.transform_lr = transform_lr
        self.transform_hr = transform_hr

        # List of all filenames
        self.filenames = os.listdir(self.lr_dir)
        self.filenames.sort()

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        # Get the file name
        filename = self.filenames[idx]

        # Build the full path for LR and HR images
        lr_path = os.path.join(self.lr_dir, filename)
        hr_path = os.path.join(self.hr_dir, filename)

        # Load the images
        lr_im = read_image(lr_path)
        hr_im = read_image(hr_path)

        # Apply the transforms
        if self.transform_lr:
            lr_im = self.transform_lr(lr_im)
        if self.transform_hr:
            hr_im = self.transform_hr(hr_im)

        # return the images
        return lr_im, hr_im