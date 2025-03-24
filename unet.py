import torch
import torch.nn as nn
import numpy as np

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.enc1 = nn.Sequential(nn.Conv2d(3, 64, 3, padding=1),
                                  nn.ReLU(),
                                  nn.BatchNorm2d(64))
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1),
                                  nn.ReLU(),
                                  nn.BatchNorm2d(128))
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1),
                                  nn.ReLU(),
                                  nn.BatchNorm2d(256))
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1),
                                  nn.ReLU(),
                                  nn.BatchNorm2d(128))

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1),
                                  nn.ReLU(),
                                  nn.BatchNorm2d(64))

        self.up3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.final_conv = nn.Conv2d(32, 3, 1)
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        enc1 = self.enc1(x)
        p1 = self.pool1(enc1)

        enc2 = self.enc2(p1)
        p2 = self.pool2(enc2)

        b = self.bottleneck(p2)

        u1 = self.up1(b)
        cat1 = torch.cat([u1, enc2], dim=1)
        d1 = self.dec1(cat1)

        u2 = self.up2(d1)
        cat2 = torch.cat([u2, enc1], dim=1)
        d2 = self.dec2(cat2)

        u3 = self.up3(d2)
        out = self.final_conv(u3)
        out = self.final_activation(out)
        return out
