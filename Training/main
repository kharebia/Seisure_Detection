# ============================================================
# EEG Seizure Detection Pipeline: .mat → Spectrograms → CNN Classification
# Fully GPU-accelerated using PyTorch
# ============================================================

import os, glob, shutil
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# -------------------------------
# 1. GPU Setup
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    print("GPU name:", torch.cuda.get_device_name(0))

# -------------------------------
# 2. Paths
# -------------------------------
root_dir = "C:/Users/khare/OneDrive/Documents/Memristive_Hardware/Data"
out_dir  = "C:/Users/khare/OneDrive/Documents/Memristive_Hardware/Output"
os.makedirs(out_dir, exist_ok=True)

# -------------------------------
# 3. Convert .mat → Spectrograms (GPU)
# -------------------------------
def save_spectrogram_gpu(eeg_1d, fs, out_path):
    eeg_tensor = torch.tensor(eeg_1d, dtype=torch.float32, device=device)
    Zxx = torch.stft(eeg_tensor, n_fft=256, hop_length=128, win_length=256, return_complex=True)
    S = torch.log1p(torch.abs(Zxx))
    S_cpu = S.cpu().numpy()

    plt.imshow(S_cpu, aspect='auto', origin='lower', cmap='viridis')
    plt.axis('off')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

all_files = glob.glob(os.path.join(root_dir, "*/.mat"), recursive=True)
print("Total .mat files found:", len(all_files))

filtered_files = [f for f in all_files if "ictal" in f.lower() or "interictal" in f.lower()]
print("Filtered files:", len(filtered_files))

fs = 400
count = 0
for f in filtered_files:
    fname = os.path.basename(f).lower()
    mat = sio.loadmat(f)
    eeg = next((mat[k] for k in ["data", "eeg", "X", "samples", "signal"] if k in mat), None)
    if eeg is None:
        continue

    eeg_ch = eeg.squeeze()[0]
    label = "ictal" if "ictal" in fname else "interictal" if "interictal" in fname else "unknown"
    label_dir = os.path.join(out_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    out_path = os.path.join(label_dir, fname.replace(".mat", ".png"))
    save_spectrogram_gpu(eeg_ch, fs, out_path)

    count += 1
    if count % 500 == 0:
        print(f"Processed {count} files...")

print("Spectrogram conversion complete. Total saved:", count)

# -------------------------------
# 4. Dataset & Dataloaders
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root=out_dir, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
print("Classes:", dataset.classes)

# -------------------------------
# 5. Load Pretrained Model
# -------------------------------
model_name = "shufflenet"

if model_name == "shufflenet":
    model = models.shufflenet_v2_x0_5(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
elif model_name == "mobilenet":
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
elif model_name == "regnet":
    model = models.regnet_x_400mf(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)

model = model.to(device)

# -------------------------------
# 6. Loss & Optimizer
# -------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# -------------------------------
# 7. Training Loop (GPU)
# -------------------------------
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    gpu_mem = torch.cuda.memory_allocated() / 1024**2 if device.type == "cuda" else 0
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} - GPU Memory: {gpu_mem:.2f} MiB")

# -------------------------------
# 8. Evaluation
# -------------------------------
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# -------------------------------
# 9. Save Model
# -------------------------------
torch.save(model.state_dict(), "/kaggle/working/eeg_model_gpu.pth")
print("Model saved to /kaggle/working/eeg_model_gpu.pth")
