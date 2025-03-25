import torch
print(torch.__version__)  # Should print the installed PyTorch version
print("CUDA Available:", torch.cuda.is_available())  # True if GPU is detected


import torch
import torch.nn as nn
import torch.optim as optim

# Define the CNN model in PyTorch
class PathPlanningCNN(nn.Module):
    def __init__(self):
        super(PathPlanningCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # Flattened size after pooling
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 20)  # Output: 10 waypoints (x, y)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # Linear activation for regression output
        return x

# Create model instance
model = PathPlanningCNN()

# Define loss function (MSE for regression) and optimizer (Adam)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Print model architecture
print(model)
