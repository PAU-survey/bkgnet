#!/usr/bin/env python

import torch
import torch.nn as nn

class CNN_model(nn.Module):
    """Network architecture for the background estimation."""

    def __init__(self):
        super(CNN_model, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=10, stride=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=2, stride=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=2, stride=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=2, stride=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = nn.Sequential(torch.nn.Linear(512*2*2+4, 1), torch.nn.LeakyReLU(0.2))

    def forward(self, x, coordx, coordy,max_values,band):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        # Embedding with the coordinate of the CCD(coordx,coordy) and the maximum flux
        # value within the postage stamp.
        out = out.view(-1,512*2*2)
        x2  = coordx
        x3 = coordy
        x4 = max_values
        x5 = band
        out = torch.cat((out,x2,x3, x4, x5), dim = 1)
        out = self.fc1(out)

        return out
