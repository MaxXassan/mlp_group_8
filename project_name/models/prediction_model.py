import torch.nn as nn
import torch
import matplotlib.pyplot as plt


class ConvNet(nn.Module):
    #constructor of the ConvNet class, using cross entropy as its loss function.
    #the rest is specified by the person creating an instance of this class.
    #model architecture inspired by https://doi.org/10.37398/jsr.2020.640251.
    def __init__(self, device):
        super(ConvNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.25),
            nn.Conv2d(64, 64, 2),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Flatten(),
            nn.Linear(64*12*12, 64),
            nn.Dropout(p=0.25),
            nn.Linear(64, 10)
        )

        self.train_losses = []
        self.test_losses = []
        self.accuracies = []
        self.device = device 
        self.model = None
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None

    def forward(self, x):
        return self.layers(x)

    