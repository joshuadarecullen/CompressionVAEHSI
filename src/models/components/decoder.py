from typing import Tuple
import torch
from torch import nn

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(256, 256 * 30 * 30)

        # Upsampling Blocks
        self.conv5 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.conv7 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(32)
        self.conv8 = nn.ConvTranspose2d(32, 12, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        z = self.fc(z)
        z = z.view(z.size(0), 256, 30, 30)  # Reshape to the tensor before flattening
        z = F.relu(self.bn5(self.conv5(z)))
        z = F.relu(self.bn6(self.conv6(z)))
        z = F.relu(self.bn7(self.conv7(z)))
        z = torch.sigmoid(self.conv8(z))  # Using sigmoid for final layer to normalize the output
        return z

