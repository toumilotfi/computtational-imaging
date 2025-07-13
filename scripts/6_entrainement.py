import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os

os.makedirs('dataset/test_reconstructions', exist_ok=True)

# Dataset personnalisé
class CTReconstructionDataset(Dataset):
    def __init__(self, folder_xk, folder_xstar):
        self.files = os.listdir(folder_xk)
        self.folder_xk = folder_xk
        self.folder_xstar = folder_xstar
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        fname = self.files[idx]
        x_k = np.load(os.path.join(self.folder_xk, fname))
        x_star = np.load(os.path.join(self.folder_xstar, fname))
        
        # Convert to tensors
        x_k = torch.tensor(x_k, dtype=torch.float32).unsqueeze(0)  # 1 channel
        x_star = torch.tensor(x_star, dtype=torch.float32).unsqueeze(0)
        
        return x_k, x_star

# UNet simplifié (à compléter ou prendre une version préexistante)
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        # Exemple très simple, à étendre
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 1, 3, padding=1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        return x

# Hyperparamètres
batch_size = 4
lr = 1e-3
epochs = 20

# Dossiers
folder_xk = "dataset/train_reconstructions/x_k"
folder_xstar = "dataset/train_reconstructions/x_star"






# Dataset + DataLoader
dataset = CTReconstructionDataset(folder_xk, folder_xstar)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Modèle, loss, optim
model = SimpleUNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Entraînement
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for x_k, x_star in loader:
        optimizer.zero_grad()
        out = model(x_k)
        loss = criterion(out, x_star)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(loader):.6f}")

# Sauvegarder modèle
torch.save(model.state_dict(), "rising_unet.pth")
print("Entraînement terminé, modèle sauvegardé.")

