import torch
import torch.nn as nn
import torch.nn.functional as F

# CustomConv2D Class with Support for All Kernel Sizes
class CustomConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1, bias=True):
        super(CustomConv2D, self).__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        
        assert isinstance(kernel_size, tuple) and len(kernel_size) == 2, "Kernel size must be a tuple (height, width)"
        
        self.stride = stride
        self.padding = padding
        self.bias = bias
        
        # Initialize separable kernels based on input kernel size
        self.kernel3x1 = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size[0], 1))
        self.kernel1x3 = nn.Parameter(torch.randn(out_channels, in_channels, 1, kernel_size[1]))
        
        if self.bias:
            self.bias_param = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias_param = None

    def forward(self, x):
        kernel3x1 = self.kernel3x1.to(x.device)
        kernel1x3 = self.kernel1x3.to(x.device)
        
        # Multiply kernels before performing convolution
        combined_kernel = kernel3x1 * kernel1x3  # Element-wise multiplication

        # Perform convolution with the combined kernel
        x = F.conv2d(x, combined_kernel, stride=self.stride, padding=self.padding)
        
        if self.bias_param is not None:
            x = x + self.bias_param.view(1, -1, 1, 1)
        
        return x

# BasicBlock for ResNet-18
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, kernel_size=3, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = CustomConv2D(in_planes, planes, kernel_size=kernel_size, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = CustomConv2D(planes, planes, kernel_size=kernel_size)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            # Adjusting the shortcut to match dimensions
            self.shortcut = nn.Sequential(
                CustomConv2D(in_planes, self.expansion * planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Ensure the dimensions match
        out = F.relu(out)
        return out

# ResNet-18 Model with CustomConv2D supporting different kernel sizes
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, kernel_size=3, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = CustomConv2D(3, 64, kernel_size=kernel_size, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], kernel_size, stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], kernel_size, stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], kernel_size, stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], kernel_size, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, kernel_size, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, kernel_size, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Instantiate ResNet-18 with CustomConv2D supporting any kernel size
def ResNet18(num_classes=1000, kernel_size=3):
    return ResNet(BasicBlock, [2, 2, 2, 2], kernel_size=kernel_size, num_classes=num_classes)

# Example usage:
# model = ResNet18(num_classes=1000, kernel_size=5)  # Change kernel_size as needed
model = ResNet18(num_classes=1000, kernel_size=5).cuda()
from torchsummary import summary
summary(model, (3, 224, 224))
