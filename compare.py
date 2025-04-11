import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt  # Importing Matplotlib for plotting
torch.manual_seed(42)

num_samples = 10000
input_data = torch.randn(num_samples, 3, 28, 28)
labels = torch.randint(0, 10, (num_samples,))

# Step 2: Define both models
# class NormalConv2D(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(NormalConv2D, self).__init__()
#         self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

#     def forward(self, x):
#         x = self.conv3x3(x)
#         return x

# class SimpleCNN(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super(SimpleCNN, self).__init__()
#         self.conv = NormalConv2D(in_channels, 16)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc = nn.Linear(16 * 14 * 14, num_classes)

#     def forward(self, x):
#         x = self.conv(x)
#         x = F.relu(x)
#         x = self.pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x
class NormalConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NormalConv2D, self).__init__()
        self.conv3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv3x3_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv3x3_1(x)
        x = F.relu(x)
        x = self.conv3x3_2(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv = NormalConv2D(in_channels, 16)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 14 * 14, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Custom separable convolution
# class CustomConv2D(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(CustomConv2D, self).__init__()
#         self.kernel3x1 = nn.Parameter(torch.randn(out_channels, in_channels, 3, 1))
#         self.kernel1x3 = nn.Parameter(torch.randn(out_channels, in_channels, 1, 3))

#     def forward(self, x):
#         combined_kernel = self.kernel3x1 * self.kernel1x3
#         x = F.conv2d(x, combined_kernel, padding=1)
#         return x

# class CustomSimpleCNN(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super(CustomSimpleCNN, self).__init__()
#         self.conv = CustomConv2D(in_channels, 16)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc = nn.Linear(16 * 14 * 14, num_classes)

#     def forward(self, x):
#         x = self.conv(x)
#         x = F.relu(x)
#         x = self.pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x
class CustomConv2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CustomConv2D, self).__init__()
        self.kernel3x1 = nn.Parameter(torch.randn(out_channels, in_channels, 3, 1))
        self.kernel1x3 = nn.Parameter(torch.randn(out_channels, in_channels, 1, 3))
        self.kernel3x1_2 = nn.Parameter(torch.randn(out_channels, out_channels, 3, 1))
        self.kernel1x3_2 = nn.Parameter(torch.randn(out_channels, out_channels, 1, 3))
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        combined_kernel1 = self.kernel3x1 * self.kernel1x3
        x = F.conv2d(x, combined_kernel1, padding=1)
        x = F.relu(x)
        x = self.pool(x)
        combined_kernel2 = self.kernel3x1_2 * self.kernel1x3_2
        x = F.conv2d(x, combined_kernel2, padding=1)
        x = F.relu(x)        
        return x

class CustomSimpleCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CustomSimpleCNN, self).__init__()
        self.conv = CustomConv2D(in_channels, 16)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 14 * 14, num_classes)

    def forward(self, x):
        x = self.conv(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Step 3: Initialize models
normal_model = SimpleCNN(in_channels=3, num_classes=10)
custom_model = CustomSimpleCNN(in_channels=3, num_classes=10)

def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

initialize_weights(normal_model)
initialize_weights(custom_model)

# Step 4: Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer_normal = optim.SGD(normal_model.parameters(), lr=0.01, momentum=0.9)
optimizer_custom = optim.SGD(custom_model.parameters(), lr=0.001, momentum=0.9)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move models to the specified device (GPU or CPU)
normal_model.to(device)
custom_model.to(device)

# Move data to the specified device (GPU or CPU)
input_data = input_data.to(device)
labels = labels.to(device)

# Step 5: Training loop
# def train_model(model, optimizer, data, labels, epochs=5):
#     model.train()
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, labels)
#         loss.backward()
#         optimizer.step()
#         print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
def train_model(model, optimizer, data, labels, batch_size=10000, epochs=5):
    model.train()
    num_batches = data.size(0) // batch_size
    for epoch in range(epochs):
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            batch_data = data[start_idx:end_idx]
            batch_labels = labels[start_idx:end_idx]

            optimizer.zero_grad()
            output = model(batch_data)
            loss = criterion(output, batch_labels)
            loss.backward()
            optimizer.step()
            
            print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{num_batches}], Loss: {loss.item():.4f}")
    return loss.item()
# Train both models
print("Training Normal CNN:")
loss_cnn=train_model(normal_model, optimizer_normal, input_data, labels,epochs=100)


print("\nTraining Custom CNN:")
loss_bakchodi=train_model(custom_model, optimizer_custom, input_data, labels,epochs=100)
print(loss_cnn,loss_bakchodi)
plt.figure(figsize=(10, 5))
plt.plot(loss_bakchodi, label='Normal CNN Loss', color='blue')
plt.plot(loss_cnn, label='Custom CNN Loss', color='orange')
plt.title('Loss Over Time')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()
torch.save(normal_model.state_dict(), 'normal_cnn_weights.pth')
torch.save(custom_model.state_dict(), 'custom_cnn_weights.pth')
# Step 6: Compare weights after training
def compare_weights(model1, model2):
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if torch.equal(param1, param2):
            print(f"Weights in {name1} and {name2} are the same.")
        else:
            print(f"Weights in {name1} and {name2} are different.")

print("\nComparing Weights:")
compare_weights(normal_model, custom_model)
