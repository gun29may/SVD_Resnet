import torch
import torch.nn as nn
from functools import partial
import torch
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm

# Autopadding Conv2d definition
class AutoPadding(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)

# Partial for conv3x3
conv3x3 = partial(AutoPadding, kernel_size=3, bias=False)
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

    
    
# Function to return activations
def activations(activation):
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
    ])[activation]

# ResidualBlock base class
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super(ResidualBlock, self).__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.activate = activations(activation)
        self.shortcut = nn.Identity()
    
    def forward(self, x):
        residual = x
        if self.shortcut_check():
            residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        return self.activate(x)

    def shortcut_check(self):
        return self.in_channels != self.out_channels

# ResNetResidualBlock
class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, downsampling=1, conv=conv3x3, *args, **kwargs):
        super(ResNetResidualBlock, self).__init__(in_channels, out_channels, *args, **kwargs)
        self.expansion, self.downsampling, self.conv = expansion, downsampling, conv
        
        # Define shortcut with downsampling if in_channels != out_channels
        self.shortcut = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels * self.expansion, kernel_size=(1,1), stride=self.downsampling, bias=False),
            nn.BatchNorm2d(self.out_channels * self.expansion)
        ) if self.shortcut_check() else nn.Identity()

    def expanded_channels(self):
        return self.out_channels * self.expansion

# Helper function to stack Conv2d + BatchNorm
def conv_temp(in_channels, out_channels, conv, *args, **kwargs):
    return nn.Sequential(
        conv(in_channels, out_channels, *args, **kwargs),
        nn.BatchNorm2d(out_channels)
    )


# ResNetBasicBlock
class ResNetBasicBlock(ResNetResidualBlock):
    expansion = 1
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super(ResNetBasicBlock, self).__init__(in_channels, out_channels, *args, **kwargs)
        
        # Main conv blocks
        self.blocks = nn.Sequential(
            conv_temp(self.in_channels, self.out_channels, conv=self.conv, bias=False, stride=self.downsampling),
            activations(self.activation),
            conv_temp(self.out_channels, self.expanded_channels(), conv=self.conv, bias=False)
        )

# Test case to check if the block works correctly
# dummy = torch.ones((1, 32, 224, 224))
# block = ResNetBasicBlock(32, 64)
# output = block(dummy)

# print(f"Output shape: {output.shape}")
# print(block)


class ResNetBottleNeckBlock(ResNetResidualBlock):
    expansion = 4
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, expansion=4, *args, **kwargs)
        self.blocks = nn.Sequential(
           conv_temp(self.in_channels, self.out_channels, self.conv, kernel_size=1),
             activations(self.activation),
             conv_temp(self.out_channels, self.out_channels, self.conv, kernel_size=3, stride=self.downsampling),
             activations(self.activation),
             conv_temp(self.out_channels, self.expanded_channels(), self.conv, kernel_size=1),
        )

# dummy = torch.ones((1, 32, 10, 10))

# block = ResNetBottleNeckBlock(32, 64)
# block(dummy).shape
# print(block)

class ResNetLayer(nn.Module):
    """
    A ResNet layer composed by `n` blocks stacked one after the other
    """
    def __init__(self, in_channels, out_channels, block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        # 'We perform downsampling directly by convolutional layers that have a stride of 2.'
        downsampling = 2 if in_channels != out_channels else 1
        self.blocks = nn.Sequential(
            block(in_channels , out_channels, *args, **kwargs, downsampling=downsampling),
            *[block(out_channels * block.expansion, 
                    out_channels, downsampling=1, *args, **kwargs) for _ in range(n - 1)]
        )

    def forward(self, x):
        x = self.blocks(x)
        return x

# dummy = torch.ones((1, 64, 48, 48))

# layer = ResNetLayer(64, 128, block=ResNetBasicBlock, n=3)
# print(layer(dummy).shape)

class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by layers with increasing features.
    """
    def __init__(self, in_channels=3, blocks_sizes=[64, 128, 256, 512], deepths=[2,2,2,2], 
                 activation='relu', block=ResNetBasicBlock, *args, **kwargs):
        super().__init__()
        self.blocks_sizes = blocks_sizes
        
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, self.blocks_sizes[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.blocks_sizes[0]),
            activations(activation),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        self.in_out_block_sizes = list(zip(blocks_sizes, blocks_sizes[1:]))
        self.blocks = nn.ModuleList([ 
            ResNetLayer(blocks_sizes[0], blocks_sizes[0], n=deepths[0], activation=activation, 
                        block=block,*args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion, 
                          out_channels, n=n, activation=activation, 
                          block=block, *args, **kwargs) 
              for (in_channels, out_channels), n in zip(self.in_out_block_sizes, deepths[1:])]       
        ])
        
        
    def forward(self, x):
        x = self.gate(x)
        for block in self.blocks:
            x = block(x)
        return x
    

class ResnetDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.decoder = nn.Linear(in_features, n_classes)

    def forward(self, x):
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.decoder(x)
        return x
    

class ResNet(nn.Module):
    
    def __init__(self, in_channels, n_classes, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels, *args, **kwargs)
        self.decoder = ResnetDecoder(self.encoder.blocks[-1].blocks[-1].expanded_channels(), n_classes)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

def resnet18(in_channels, n_classes, block=ResNetBasicBlock, *args, **kwargs):
    return ResNet(in_channels, n_classes, block=block, deepths=[2, 2, 2, 2], *args, **kwargs)

from torchsummary import summary

model = resnet18(3, 10).cuda()
summary(model, (3, 224, 224))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=6000, shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=6000, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
custom_model = model

# Training the normal_model
net = custom_model.to('cuda')
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

for epoch in tqdm(range(1000)):  # Loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/2000:.5f}")
    running_loss = 0.0

print('Finished Training Normal Model')

# Evaluate the model accuracy on test data
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)  # Move data to GPU           
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test images: {100 * correct / total}%')

    # Optionally, train the custom model by replacing `net = normal_model` with `net = custom_model` and rerun the training loop