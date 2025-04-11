import torch
import torch.nn as nn
import torch.nn.functional as F

class NormalConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NormalConv2D, self).__init__()
        # Standard 3x3 convolution
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Apply the 3x3 convolution
        x = self.conv3x3(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SimpleCNN, self).__init__()
        # Standard convolution with 3x3 kernel
        self.conv = NormalConv2D(in_channels, 16)
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling with kernel size 2 and stride 2

        # Calculate the size of the fully connected input dynamically
        self.fc = nn.Linear(16 * 14 * 14, num_classes)  # Adjusted for 28x28 input after pooling

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.pool(x)  # Apply max pooling (28x28 -> 14x14)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

# Example usage:
model = SimpleCNN(in_channels=3, num_classes=10)  # For RGB images like CIFAR-10
print(model)

# Input for testing (RGB image)
x = torch.randn(1, 3, 28, 28)  # Batch of 1, 3 channels (RGB), 28x28 image
output = model(x)
print(output.shape)
