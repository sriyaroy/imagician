import numpy as np
import torch
import os
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
import cv2
from PIL import Image
import unet
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# downsample and crop to 64 x 64 to create low_res counterpart
def lowRes(image_path, target_path):
    hr_image = cv2.imread(image_path)
    height, width = hr_image.shape[:2]

    if height > 128:
        hr_image = hr_image[:, 0:128]
    if width > 128:
        hr_image = hr_image[0:128, :]

    lr_dim = (int(width * 0.5), int(height * 0.5))
    lr_image = cv2.resize(hr_image, lr_dim, interpolation = cv2.INTER_AREA)

    cv2.imwrite(image_path, hr_image)
    cv2.imwrite(target_path, lr_image)

root_dir = 'data-64x64-128x128/high_res'
target_dir = 'data-64x64-128x128/low_res'

''' 
for filename in os.listdir(root_dir):
    if filename.endswith(".png"):
        lowRes(os.path.join(root_dir, filename), os.path.join(target_dir, filename))

'''

# Create dataset object for custom paired dataset
computed_mean = [0.5749, 0.5367, 0.4373]
computed_std = [0.1970, 0.1929, 0.2085]

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
        lr_im = Image.open(lr_path).convert('RGB')
        hr_im = Image.open(hr_path).convert('RGB')

        # Apply the transforms
        if self.transform_lr:
            lr_im = self.transform_lr(lr_im)
        if self.transform_hr:
            hr_im = self.transform_hr(hr_im)

        # return the images
        return lr_im, hr_im
    
## Use your custom data loader and apply transforms required
transform_lr = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=computed_mean, std=computed_std)])
transform_hr = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=computed_mean, std=computed_std)])

lr_dir = 'data-64x64-128x128/low_res'
hr_dir = 'data-64x64-128x128/high_res'

dataset = PairedImageDataset(lr_dir, hr_dir, transform_lr, transform_hr)

# Split train/test
train_ratio = 0.8
train_size = int(train_ratio * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

# Quick test to see batch shapes
print(f'Train batch shape: {train_loader.dataset[0][0].shape}') 
print(f'Test batch shape: {test_loader.dataset[0][0].shape}')

### Training Loop ###
model = unet.UNet()

device = torch.device('cpu')
model = model.to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimiser = optim.Adam(model.parameters(), lr=1e-4)
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for lr_imgs, hr_imgs in train_loader:
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)

        optimiser.zero_grad()

        # forward pass
        outputs = model(lr_imgs)

        # Compute loss
        loss = criterion(outputs, hr_imgs)

        # backwards pass and optimisation
        loss.backward()
        optimiser.step()

        running_loss += loss.item() * lr_imgs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}'.format(epoch+1, 10, epoch_loss))

import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_test_samples(model, test_loader, device, num_samples=3):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for lr_imgs, hr_imgs in test_loader:
            # Move images to device
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            outputs = model(lr_imgs)
            
            # Transfer to CPU and convert to numpy arrays
            lr_imgs = lr_imgs.cpu().numpy()
            outputs = outputs.cpu().numpy()
            hr_imgs = hr_imgs.cpu().numpy()

            # Plot a few samples
            for i in range(min(num_samples, lr_imgs.shape[0])):
                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                
                # Convert tensor shape from [C, H, W] to [H, W, C]
                lr_img = np.transpose(lr_imgs[i], (1, 2, 0))
                out_img = np.transpose(outputs[i], (1, 2, 0))
                hr_img = np.transpose(hr_imgs[i], (1, 2, 0))
                
                # Display images
                axs[0].imshow(lr_img)
                axs[0].set_title("Low Resolution Input")
                axs[0].axis("off")
                
                axs[1].imshow(out_img)
                axs[1].set_title("Super Resolved Output")
                axs[1].axis("off")
                
                axs[2].imshow(hr_img)
                axs[2].set_title("High Resolution Ground Truth")
                axs[2].axis("off")
                
                plt.show()
            break  # Only visualize the first batch

# Example usage:
visualize_test_samples(model, test_loader, device, num_samples=3)

## TODO: Compute mean and std for random train test split and normalise
## TODO: Why is the model so poor? Plot loss curve and test on better images
## TODO: Try different loss functions