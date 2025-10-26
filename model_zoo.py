# model_zoo.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseCNN(nn.Module):
    """
    A simple Convolutional Neural Network for MNIST classification.
    (Readable, multi-line forward pass)
    """
    def __init__(self):
        super(BaseCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Block 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Classifier
        x = x.view(-1, 64 * 7 * 7) # Flatten
        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class Autoencoder(nn.Module):
    """
    A simple Convolutional Autoencoder for MNIST.
    (Readable, multi-line forward pass)
    """
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # --- Encoder ---
        self.enc1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1) 
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- Decoder ---
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2) 
        self.dec1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.up2 = nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2)
        self.dec2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        # --- Encode ---
        x = self.enc1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.enc2(x)
        x = F.relu(x)
        x = self.pool2(x) # Bottleneck
        
        # --- Decode ---
        x = self.up1(x)
        x = F.relu(x)
        x = self.dec1(x)
        x = F.relu(x)
        
        x = self.up2(x)
        x = F.relu(x)
        x = self.dec2(x)
        
        # Use Tanh for the final output
        x = torch.tanh(x)
        
        return x