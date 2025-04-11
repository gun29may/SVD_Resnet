import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomConv2D, self).__init__()
        # Define the 3x1 and 1x3 kernels for multi-channel input
        self.kernel3x1 = nn.Parameter(torch.randn(out_channels, in_channels, 3, 1))
        self.kernel1x3 = nn.Parameter(torch.randn(out_channels, in_channels, 1, 3))

    def forward(self, x):
        # Multiply the 3x1 and 1x3 kernels to get a 3x3 kernel
        # Element-wise multiplication across input channels
        combined_kernel = self.kernel3x1 * self.kernel1x3
        
        # Apply convolution using the constructed 3x3 kernel
        x = F.conv2d(x, combined_kernel, padding=1)  # Padding=1 to maintain spatial dimensions
        return x

class SimpleCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SimpleCNN, self).__init__()
        # Custom convolution with 3x1 and 1x3 kernels for RGB input (in_channels=3)
        self.custom_conv = CustomConv2D(in_channels, 16)
        self.fc = nn.Linear(16 * 14 * 14, num_classes)  # Assuming input size is 28x28 (like CIFAR-10)

    def forward(self, x):
        x = self.custom_conv(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
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
