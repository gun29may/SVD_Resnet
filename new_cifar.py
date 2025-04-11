
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=7000, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=7000, shuffle=False, num_workers=2)

# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Define the original CNN with 3x3 kernel instead of 5x5
import torch
import torch.nn as nn
import torch.nn.functional as F

# CustomConv2D Class with Support for All Kernel Sizes
class CustomConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1, bias=False):
        super(CustomConv2D, self).__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        
        assert isinstance(kernel_size, tuple) and len(kernel_size) == 2, "Kernel size must be a tuple (height, width)"
        
        self.stride = stride
        self.padding = padding
        self.bias = bias
        
        # Initialize separable kernels based on input kernel size
        self.kernel3x1 = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size[0], 1))
        self.kernel1x3 = nn.Parameter(torch.empty(out_channels, in_channels, 1, kernel_size[1]))
        nn.init.kaiming_normal_(self.kernel3x1, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.kernel1x3, mode='fan_out', nonlinearity='relu')
        
        if self.bias:
            self.bias_param = nn.Parameter(torch.randn(out_channels))
        else:
            self.bias_param = None

    def forward(self, x):
        kernel3x1 = self.kernel3x1.to(x.device)
        kernel1x3 = self.kernel1x3.to(x.device)
        
        # Multiply kernels before performing convolution
        combined_kernel = kernel3x1 * kernel1x3  

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
    def __init__(self, block, num_blocks, kernel_size=5, num_classes=1000):
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

def ResNet34(num_classes=1000, kernel_size=5):
    return ResNet(BasicBlock, [3, 4, 6, 3], kernel_size=kernel_size, num_classes=num_classes)

# Example usage:
# model = ResNet18(num_classes=1000, kernel_size=5)  # Change kernel_size as needed

custom_model = ResNet18(num_classes=10, kernel_size=3).cuda()
# custom_model.load_weights('new_resnet4_latest.pth')


# Training the normal_model
net = custom_model.to('cuda')
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9,weight_decay=0.0001)

# for epoch in tqdm(range(10000)):  # Loop over the dataset multiple times
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         inputs, labels = data
#         inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
#         optimizer.zero_grad()
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#     print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/2000:.7f}")
#     running_loss = 0.0

# print('Finished Training Normal Model')
# torch.save(custom_model.state_dict(), 'new_resnet3.pth')
best_loss = float('inf')  # Initialize the best loss to a high value

# for epoch in tqdm(range(10000)):  # Loop over the dataset multiple times
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         inputs, labels = data
#         inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
#         optimizer.zero_grad()
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#     # Calculate average loss for the epoch
#     avg_loss = running_loss / len(trainloader)

#     # Save the most recent weights
#     torch.save(net.state_dict(), 'new_resnet5_latest100.pth')

#     # Save the best weights based on loss
#     if avg_loss < best_loss:
#         best_loss = avg_loss
#         torch.save(net.state_dict(), 'new_resnet5_best_100.pth')
#     print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/2000:.7f}")


#     # print(f"Epoch {epoch + 1}, Loss: {avg_loss:.7f}")

# print('Finished Training Normal Model')
# # Evaluate the model accuracy on test data
# correct = 0
# total = 0
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         images, labels = images.to(device), labels.to(device)  # Move data to GPU           
#         outputs = net(images)
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(f'Accuracy of the network on the test images: {100 * correct / total}%')

# Optionally, train the custom model by replacing `net = normal_model` with `net = custom_model` and rerun the training loop
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

# Initialize TensorBoard writer
# state_dict = torch.load('new_resnet_34_best_100_val_1.pth')
# custom_model.load_state_dict(state_dict,strict=False)
writer = SummaryWriter(log_dir='./runs/experiment7')

best_loss = float('inf')
best_acc=float(0)

for epoch in tqdm(range(10000)):  # Loop over the dataset multiple times
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

    # Calculate average loss for the epoch
    avg_loss = running_loss / len(trainloader)

    # Log average loss to TensorBoard
    writer.add_scalar('Training Loss', avg_loss, epoch)

    # Save the most recent weights
    torch.save(net.state_dict(), 'half_resnet_18_latest100_1_.pth')

    # Save the best weights based on loss
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(net.state_dict(), 'new_resnet_18_best_100_1_5.pth')

    print(f"Epoch {epoch+1}, Loss: {avg_loss:.7f}")

    if (epoch + 1) % 5 == 0:
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

        accuracy = 100 * correct / total
        print(f'Accuracy on test images after {epoch+1} epochs: {accuracy}%')
        if accuracy > best_acc:
            best_acc = accuracy
            
            torch.save(net.state_dict(), 'new_resnet_ _best_100_val_1_5.pth')

        # Log accuracy to TensorBoard
        writer.add_scalar('Validation Accuracy', accuracy, epoch)
    if (epoch + 1) % 50 == 0:
        correct = 0
        total = 0
        with torch.no_grad():
            for data in trainloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)  # Move data to GPU           
                outputs = net(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy on train images after {epoch+1} epochs: {accuracy}%')

# Close TensorBoard writer
writer.close()

print('Finished Training Normal Model')
