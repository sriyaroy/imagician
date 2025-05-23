import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import test_loader
import unet
from sewar.full_ref import ssim
import os
import datetime

## SET CHECKPOINT TO VALIDATE
folder_path = input('Enter model-runs folder path: ')
chkpt_name = input('Enter checkpoint name: ')
model = unet.UNet()
model.load_state_dict(torch.load('model-runs/' + folder_path + '/checkpoints/' + chkpt_name + '.pth'))
model.eval() # Set the model to evaluation mode

# Save the outputs to folder
def save_test_samples(model, test_loader, num_samples=10):
    saved = 0
    with torch.no_grad():    
        for lr_imgs, hr_imgs in test_loader:
            outputs = model(lr_imgs).cpu().numpy()
            hr_imgs = hr_imgs.cpu().numpy()
            lr_imgs = lr_imgs.cpu().numpy()

            batch_size = outputs.shape[0]

            for i in range(batch_size):
                if saved >= num_samples:
                    return
                
                # Convert tensor shape from [C, H, W] to [H, W, C]
                lr_img = np.transpose(lr_imgs[i], (1, 2, 0))
                out_img = np.transpose(outputs[i], (1, 2, 0))
                hr_img = np.transpose(hr_imgs[i], (1, 2, 0))
                
                # Save images
                plt.imsave('model-runs/' + folder_path + '/samples/test/'+ f'out_{i}.png', out_img)
                plt.imsave('model-runs/' + folder_path + '/samples/test/'+ f'hr_{i}.png', hr_img)
                plt.imsave('model-runs/' + folder_path + '/samples/test/'+ f'lr_{i}.png', lr_img)

                saved += 1

def visualize_test_samples(model, test_loader, num_samples=3):
    with torch.no_grad():
        for lr_imgs, hr_imgs in test_loader:
            outputs = model(lr_imgs)
            
            # Transfer to CPU and convert to numpy arrays
            lr_imgs = lr_imgs.cpu().numpy()
            outputs = outputs.cpu().numpy()
            hr_imgs = hr_imgs.cpu().numpy()

            '''
            # Plot a few samples
            for i in range(3):
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
            break # Only visualize the first batch
            '''

## USAGE
save_test_samples(model, test_loader, num_samples=10)

#visualize_test_samples(model, test_loader, num_samples=3)