"""Experiment: improved CNN for Fashion-MNIST (Naloga 3).
Reproduces the exact data split from dn2.ipynb (seed 42, 10k train / 2k test)
and trains an improved CNN to beat the 85.26% baseline (CNN7).
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import time

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {device}")

# --- Data: same split as the notebook (cell 28) ---
norm = transforms.Normalize((0.5,), (0.5,))

# Augmented transform for training: small shifts + horizontal flip.
# Fashion-MNIST classes are left-right symmetric, so flipping is safe.
train_transform = transforms.Compose([
    transforms.RandomCrop(28, padding=2),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    norm,
])
test_transform = transforms.Compose([transforms.ToTensor(), norm])

train_polni_aug = datasets.FashionMNIST(root='./data', train=True, download=True, transform=train_transform)
train_polni_plain = datasets.FashionMNIST(root='./data', train=True, download=True, transform=test_transform)
test_polni = datasets.FashionMNIST(root='./data', train=False, download=True, transform=test_transform)

# Exactly the same indices as in the notebook
np.random.seed(42)
train_idx_nn = np.random.choice(len(train_polni_aug), 10000, replace=False)
test_idx_nn = np.random.choice(len(test_polni), 2000, replace=False)

train_set_aug = Subset(train_polni_aug, train_idx_nn.tolist())
test_set = Subset(test_polni, test_idx_nn.tolist())


# --- Model: VGG-style CNN, two convolutions per block ---
class ImprovedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 28x28 -> 14x14
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            # Block 2: 14x14 -> 7x7
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            # Block 3: 7x7 -> 3x3
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3 * 3, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def train_improved(model, epochs=30, batch_size=128, lr=1e-3):
    loader = DataLoader(train_set_aug, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    model.train()
    for epoch in range(epochs):
        running = 0.0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
        scheduler.step()
        print(f"  epoch {epoch+1}/{epochs}  loss {running/len(loader):.4f}", flush=True)


def evaluate(model):
    loader = DataLoader(test_set, batch_size=256, shuffle=False)
    model.eval()
    correct = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            pred = model(inputs).argmax(1)
            correct += (pred == labels).sum().item()
    return correct / len(test_set)


# --- Stability over 5 seeds (same protocol as the notebook) ---
seeds = [42, 66, 12, 7, 100]
results = []
for seed in seeds:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ImprovedCNN().to(device)
    t0 = time.time()
    print(f"\nSeed {seed}")
    train_improved(model)
    acc = evaluate(model)
    results.append(acc)
    print(f"Seed {seed}: accuracy {acc:.4f}  ({time.time()-t0:.0f}s)", flush=True)

print("\n=== ImprovedCNN results ===")
print(f"Accuracies: {[f'{a:.4f}' for a in results]}")
print(f"Mean: {np.mean(results):.4f}")
print(f"Std:  {np.std(results):.4f}")
