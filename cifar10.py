
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=10000, shuffle=False, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=10000, shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Define the original CNN with 3x3 kernel instead of 5x5
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 3)  # Changed to 3x3 kernel
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 3)  # Changed to 3x3 kernel
            self.fc1 = nn.Linear(16 * 6 * 6, 120)  # Updated input size
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 6 * 6)  # Flatten
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # Define the custom convolution layer
    class CustomConv2D(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(CustomConv2D, self).__init__()
            self.kernel3x1 = nn.Parameter(torch.randn(out_channels, in_channels, 3, 1))
            self.kernel1x3 = nn.Parameter(torch.randn(out_channels, in_channels, 1, 3))
            # self.kernel3x1_2 = nn.Parameter(torch.randn(out_channels, out_channels, 3, 1))
            # self.kernel1x3_2 = nn.Parameter(torch.randn(out_channels, out_channels, 1, 3))
            # self.pool = nn.MaxPool2d(2,2)

        def forward(self, x):
            # First convolution
            combined_kernel1 = self.kernel3x1 * self.kernel1x3
            x = F.conv2d(x, combined_kernel1)
            # x = F.relu(x)
            # x = self.pool(x)
            # # Second convolution
            # combined_kernel2 = self.kernel3x1_2 * self.kernel1x3_2
            # x = F.conv2d(x, combined_kernel2)
            # x = F.relu(x)
            # x = self.pool(x)
            return x

    # Define the custom CNN
    class CustomSimpleCNN(nn.Module):
        def __init__(self):
            super(CustomSimpleCNN, self).__init__()
            self.conv = CustomConv2D(3, 16)
            self.fc1 = nn.Linear(16 * 6 * 6, 120)
            self.fc2 = nn.Linear(120,84)
            self.fc3 = nn.Linear(84,10)

        def forward(self, x):
            x = self.conv(x)
            x = x.view(x.size(0), -1)  
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # Step 3: Initialize models
    # normal_model = Net()
    custom_model = Net()

    # Training the normal_model
    net = custom_model.to('cuda')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

    for epoch in range(100):  # Loop over the dataset multiple times
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
        print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/2000:.3f}")
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