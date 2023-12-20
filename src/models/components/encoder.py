from typing import List
import torch
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # Initial Convolution Block
        self.conv1 = nn.Conv2d(12, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Pyramid Convolution Blocks
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # Spatial size halved
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1) # Spatial size halved again

        # Flatten layer
        self.flatten_size = 256 * 30 * 30

        # Latent space representation
        self.fc_mu = nn.Linear(self.flatten_size, 256)
        self.fc_logvar = nn.Linear(self.flatten_size, 256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

