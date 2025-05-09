
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
import numpy as np
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the data
adj_matrices = np.load("adj_matrices.npy", allow_pickle=True)

data = np.load("eeg_dataset.npz")

node_features = data["X"]  # shape: [N, C, T]
labels = data["labels"]  # shape: [N] or [N, 1]
if labels.min() != 0:
    labels = labels - labels.min()  # shift to start from 0
#labels = np.load("labels.npy", allow_pickle=True)
#node_features = np.load("node_features.npy", allow_pickle=True)

# Prepare graph data
graph_list = []
for i in range(len(adj_matrices)):
    adj = adj_matrices[i]
    features = node_features[i]
    label = labels[i]

    edge_index = torch.tensor(np.array(adj.nonzero()), dtype=torch.long)
    x = torch.tensor(np.array(features), dtype=torch.float)
    y = torch.tensor(np.array(label), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    graph_list.append(data)

# DataLoader
train_data, val_data = train_test_split(graph_list, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

class CNNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.norm1 = torch.nn.BatchNorm1d()
        self.conv2 = torch.nn.Conv1d(16, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # x: [num_nodes, signal_length]
        x = x.unsqueeze(1)  # [num_nodes, 1, signal_length]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x.mean(dim=2)  # Pool over time -> [num_nodes, out_channels]

# GCN model
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)  # Pooling to get graph-level embedding
        x = self.lin(x)
        return x

# Initialize model, optimizer, and loss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(input_dim=321, hidden_dim=128, output_dim=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

# Training loop
for epoch in range(1, 2):
    model.train()
    total_loss = 0
    correct = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (out.argmax(dim=1) == data.y).sum().item()
    train_acc = correct / len(train_data)
    train_losses.append(total_loss)
    train_accuracies.append(train_acc)

    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            val_loss += criterion(out, data.y).item()
            correct += (out.argmax(dim=1) == data.y).sum().item()
    val_acc = correct / len(val_data)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch:02d}, Train Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}, "
          f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

# Plot
epochs = range(1, 2)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label="Train Accuracy")
plt.plot(epochs, val_accuracies, label="Val Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs")
plt.legend()

plt.tight_layout()
plt.savefig("training_plot.png")
plt.show()