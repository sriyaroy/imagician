import numpy as np
import torch
import unet
import torch.nn as nn
import torch.optim as optim
from dataset import train_loader, test_loader
import datetime
from torch.utils.tensorboard import SummaryWriter
import os

# Set up folders to save outputs
datetime = datetime.datetime.now()
date = datetime.strftime('%Y-%m-%d')
time = datetime.strftime('%H-%M-%S')

## REMEMBER TO SET THE FOLLOWING:
unique_chkpt_name = input('Enter checkpoint name: ')
num_epochs = int(input('Enter number of epochs: '))

if not os.path.exists(f'model-runs/{date}-{time}-{unique_chkpt_name}'):
    os.makedirs(f'model-runs/{date}-{time}-{unique_chkpt_name}')
    os.makedirs(f'model-runs/{date}-{time}-{unique_chkpt_name}/checkpoints') # Create folder to store checkpoints
    os.makedirs(f'model-runs/{date}-{time}-{unique_chkpt_name}/samples') # To save test dataset & (eventually) validation dataset
    os.makedirs(f'model-runs/{date}-{time}-{unique_chkpt_name}/samples/test')

if train_loader.batch_size != test_loader.batch_size:
    raise ValueError('Train and test batches must be of the same size')

### Training Loop ###
writer = SummaryWriter()
model = unet.UNet()

device = torch.device('mps')
model = model.to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimiser = optim.Adam(model.parameters(), lr=1e-4)

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
        writer.add_scalar('Loss/train', loss, epoch)

        # backwards pass and optimisation
        loss.backward()
        optimiser.step()

        running_loss += loss.item() * lr_imgs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}'.format(epoch+1, 10, epoch_loss))

# Save the trained model
writer.flush()
writer.close()
torch.save(model.state_dict(), f'model-runs/{date}-{time}-{unique_chkpt_name}/checkpoints/epoch_{num_epochs}.pth')