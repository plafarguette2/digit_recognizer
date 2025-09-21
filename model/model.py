"""
Define the architecture of a lightweight Convolutional Neural Network (CNN) 
for handwritten digit recognition using the MNIST dataset.

The Architecture:
    - Convolution 1: input channels = 1 (grayscale image in MNIST), output channels = 32, kernel size = 3x3
    - ReLU activation + MaxPool (2x2)
    - Convolution 2: input channels = 32, output channels = 64, kernel size = 3x3
    - ReLU activation + MaxPool (2x2)
    - Flatten: [batch size, 64, 7, 7] → [batch_size, 64*7*7]
    - Fully connected 1: 64*7*7 → 128
    - Fully connected 2: 128 → 10 class  (for digits 0–9)

"""

import torch.nn as nn
import torch.nn.functional as F

class my_model(nn.Module):
    def __init__(self):
        """
        Initialize all layers.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """
        Define the forward pass of the cnn.

        Args : 
        - x (torch.Tensor): Batch of MNIST images, shape [batch size, 1, 28, 28].

        Return : 
        - torch.tensor : Raw class scores, shape [batch size, 10]
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7) # flattening to feed fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x