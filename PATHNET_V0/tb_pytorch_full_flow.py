import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import os

# === Step 1: Dataset Definition ===

class GridPathDataset(Dataset):
    def __init__(self, data_dir):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npy")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = np.load(self.files[idx], allow_pickle=True).item()
        grid = sample["grid"].astype(np.float32)    # Shape: (10, 10)
        path = sample["path"]

        # Normalize and pad/truncate path to 10 waypoints
        path = np.pad(path, ((0, max(0, 10 - len(path))), (0, 0)), mode='constant')[:10]
        path = path.astype(np.float32).flatten()    # Shape: (20,)

        grid = grid[np.newaxis, :, :]               # Add channel dimension (1, 10, 10)
        return torch.tensor(grid), torch.tensor(path)

# === Step 2: Load Dataset and Split ===

data_dir = "/home/thh3/dev/MPAi/PATHNET_V0/grid_dataset" 
dataset = GridPathDataset(data_dir)

train_size = int(0.8 * len(dataset))
val_size   = int(0.1 * len(dataset))
test_size  = len(dataset) - train_size - val_size

train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_set, batch_size=32)
test_loader  = DataLoader(test_set, batch_size=32)

# === Step 3: Model Definition ===

class PathPlanningCNN(nn.Module):
    def __init__(self):
        super(PathPlanningCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool  = nn.MaxPool2d(2)
        self.fc1   = nn.Linear(64 * 5 * 5, 128)
        self.fc2   = nn.Linear(128, 64)
        self.fc3   = nn.Linear(64, 20)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# === Step 4: Training Setup ===

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PathPlanningCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# === Step 5: Training Loop ===

def evaluate(model, loader):
    model.eval()
    total_loss = 0
    total_mae = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            mae = torch.abs(pred - y).mean()
            total_loss += loss.item()
            total_mae += mae.item()
    return total_loss / len(loader), total_mae / len(loader)

EPOCHS = 50
for epoch in range(EPOCHS):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

    val_loss, val_mae = evaluate(model, val_loader)
    print(f"Epoch {epoch+1:02d} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f}")

