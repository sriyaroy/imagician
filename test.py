import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import test_loader
import unet

## SET CHECKPOINT TO VALIDATE
model = unet.UNet()
model.load_state_dict(torch.load('checkpoints/2025-05-16-15-20-38.pth'))
model.eval() # Set the model to evaluation mode

device = torch.device('cpu')

# Denormalise images from model

def visualize_test_samples(model, test_loader, device, num_samples=3):
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