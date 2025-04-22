import numpy as np
import torch
import unet
import torch.nn as nn
import torch.optim as optim
from dataset import train_loader, test_loader
import datetime


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

# Save the trained model
datetime = datetime.datetime.now()
date = datetime.strftime('%Y-%m-%d')
time = datetime.strftime('%H-%M-%S')
torch.save(model.state_dict(), f'checkpoints/{date}-{time}.pth')