#!/usr/bin/env python

import torch
from torch import nn

class CNN_model(nn.Module):
    def __init__(self):
        super(CNN_model, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=9, stride=1, padding = 4),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=7, stride=1, padding = 3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding = 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))


        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=2, stride=1, padding = 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))


        self.fc1 = nn.Sequential(nn.BatchNorm1d(512*3*3+9),torch.nn.Linear(512*3*3+9, 1000), torch.nn.LeakyReLU(0.2))
        self.fc5 = nn.Sequential(nn.BatchNorm1d(1000),torch.nn.Linear(1000, 500), torch.nn.LeakyReLU(0.2))
        self.fc6 = nn.Sequential(nn.BatchNorm1d(500),torch.nn.Linear(500, 100), torch.nn.LeakyReLU(0.2))
        self.fc7 = nn.Sequential(torch.nn.Linear(100, 2))


        self.embed = nn.Embedding(40, 5)



    def forward(self, x, coordx, coordy,max_values,band,stdv):

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        out = out.view(-1,512*3*3)
        x2  = coordx
        x3 = coordy
        x4 = max_values

        # Dimention mismatch...
        band = band.squeeze(1)
        x5 = self.embed(band)
        x6 = stdv.unsqueeze(1)


        out = torch.cat((out,x2,x3, x4, x5, x6), dim = 1)
        out = self.fc1(out)
        out = self.fc5(out)
        out = self.fc6(out)
        out = self.fc7(out)

        return out
