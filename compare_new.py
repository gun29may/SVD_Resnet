import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define both models (from earlier)
class NormalConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NormalConv2D, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv3x3(x)
        return x

class CustomConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomConv2D, self).__init__()
        self.kernel3x1 = nn.Parameter(torch.randn(out_channels, in_channels, 3, 1))
        self.kernel1x3 = nn.Parameter(torch.randn(out_channels, in_channels, 1, 3))

    def forward(self, x):
        combined_kernel = self.kernel3x1 * self.kernel1x3
        x = F.conv2d(x, combined_kernel, padding=1)
        return x

# Extract the 3x3 kernels and compare
def extract_kernels_and_compare(normal_model, custom_model):
    # Get the 3x3 kernel from the normal model's Conv2D layer
    normal_kernel = normal_model.conv.conv3x3.weight.data[0, 0].cpu().numpy()  # Shape: (3, 3)
    
    # Get the 3x1 and 1x3 kernels from the custom model and calculate their product
    kernel_3x1 = custom_model.conv.kernel3x1.data[0, 0].cpu().numpy()  # Shape: (3, 1)
    kernel_1x3 = custom_model.conv.kernel1x3.data[0, 0].cpu().numpy()  # Shape: (1, 3)
    
    # Compute the 3x3 kernel from the separable convolutions
    custom_3x3_kernel = (kernel_3x1 @ kernel_1x3).reshape(3, 3)  # Matrix multiplication to get (3,3)
    
    # Print both kernels
    print("Normal Conv2D 3x3 Kernel:")
    print(normal_kernel)
    
    print("\nCustom Conv2D 3x3 Kernel (Product of 3x1 and 1x3):")
    print(custom_3x3_kernel)

    # Calculate and print the differences between the corresponding elements
    diff = normal_kernel - custom_3x3_kernel
    print("\nDifference between Normal and Custom 3x3 Kernels:")
    print(diff)

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

# Compare the kernels and their differences
extract_kernels_and_compare(normal_model, custom_model)
