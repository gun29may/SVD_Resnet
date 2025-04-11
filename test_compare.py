import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define both models with two convolutional layers
class NormalConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NormalConv2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

class CustomConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomConv2D, self).__init__()
        self.kernel3x1_1 = nn.Parameter(torch.randn(out_channels, in_channels, 3, 1))
        self.kernel1x3_1 = nn.Parameter(torch.randn(out_channels, in_channels, 1, 3))
        self.kernel3x1_2 = nn.Parameter(torch.randn(out_channels, out_channels, 3, 1))
        self.kernel1x3_2 = nn.Parameter(torch.randn(out_channels, out_channels, 1, 3))

    def forward(self, x):
        combined_kernel1 = self.kernel3x1_1 * self.kernel1x3_1
        x = F.relu(F.conv2d(x, combined_kernel1, padding=1))
        combined_kernel2 = self.kernel3x1_2 * self.kernel1x3_2
        x = F.relu(F.conv2d(x, combined_kernel2, padding=1))
        return x

# Extract the 3x3 kernels from both convolution layers and compare
def extract_kernels_and_compare(normal_model, custom_model):
    # Extract the kernels from the first conv layer of the normal model
    normal_kernel1 = normal_model.conv1.weight.data[0, 0].cpu().numpy()  # Shape: (3, 3)
    # Extract the kernels from the second conv layer of the normal model
    normal_kernel2 = normal_model.conv2.weight.data[0, 0].cpu().numpy()  # Shape: (3, 3)
    
    # Get the 3x1 and 1x3 kernels from the custom model's first conv layer and calculate their product
    kernel_3x1_1 = custom_model.kernel3x1_1.data[0, 0].cpu().numpy()  # Shape: (3, 1)
    kernel_1x3_1 = custom_model.kernel1x3_1.data[0, 0].cpu().numpy()  # Shape: (1, 3)
    custom_3x3_kernel1 = (kernel_3x1_1 @ kernel_1x3_1).reshape(3, 3)  # Matrix multiplication to get (3,3)
    
    # Get the 3x1 and 1x3 kernels from the custom model's second conv layer and calculate their product
    kernel_3x1_2 = custom_model.kernel3x1_2.data[0, 0].cpu().numpy()  # Shape: (3, 1)
    kernel_1x3_2 = custom_model.kernel1x3_2.data[0, 0].cpu().numpy()  # Shape: (1, 3)
    custom_3x3_kernel2 = (kernel_3x1_2 @ kernel_1x3_2).reshape(3, 3)  # Matrix multiplication to get (3,3)

    # Print kernels for the first layer
    print("First Conv Layer - Normal Conv2D 3x3 Kernel:")
    print(normal_kernel1)
    print("\nFirst Conv Layer - Custom Conv2D 3x3 Kernel (Product of 3x1 and 1x3):")
    print(custom_3x3_kernel1)

    # Print difference for the first layer
    diff1 = normal_kernel1 - custom_3x3_kernel1
    print("\nDifference for First Conv Layer:")
    print(diff1)

    # Print kernels for the second layer
    print("\nSecond Conv Layer - Normal Conv2D 3x3 Kernel:")
    print(normal_kernel2)
    print("\nSecond Conv Layer - Custom Conv2D 3x3 Kernel (Product of 3x1 and 1x3):")
    print(custom_3x3_kernel2)

    # Print difference for the second layer
    diff2 = normal_kernel2 - custom_3x3_kernel2
    print("\nDifference for Second Conv Layer:")
    print(diff2)

# Initialize both models
normal_model = NormalConv2D(3, 16)
custom_model = CustomConv2D(3, 16)

# Initialize with Xavier Uniform
def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

initialize_weights(normal_model)
initialize_weights(custom_model)

# Compare the kernels and their differences for both convolutional layers
extract_kernels_and_compare(normal_model, custom_model)
