import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from tqdm import tqdm
from utils import plot_metrics
from triplet_dataset import TripletFaceDataset # ✅ Use your custom dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
train_dir = '../facedataset_split/train'
val_dir = '../facedataset_split/eval'
checkpoint_dir = '../outputs/checkpoints'
plot_dir = '../outputs/plots'
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# Transforms
transform = transforms.Compose([
transforms.Resize((160, 160)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.5]*3, std=[0.5]*3) # Optional normalization
])

# Dataset (uses triplets directly)
train_dataset = TripletFaceDataset(train_dir, transform=transform, num_triplets=10000)
val_dataset = TripletFaceDataset(val_dir, transform=transform, num_triplets=2000)

# DataLoader (no shuffling needed, dataset already randomized)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Model (embedding network)
model = InceptionResnetV1(pretrained='vggface2', classify=False).to(device)

# Freeze BatchNorms
for module in model.modules():
if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
module.eval()
for param in module.parameters():
param.requires_grad = False

model.train()

# Loss and Optimizer
triplet_loss = nn.TripletMarginLoss(margin=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training config
epochs = 10
train_losses = []
val_losses = []

# Training loop
for epoch in range(epochs):
model.train()
epoch_loss = 0

for anchor, positive, negative in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

emb_anchor = model(anchor)
emb_pos = model(positive)
emb_neg = model(negative)

loss = triplet_loss(emb_anchor, emb_pos, emb_neg)
optimizer.zero_grad()
loss.backward()
optimizer.step()
epoch_loss += loss.item()

avg_train_loss = epoch_loss / len(train_loader)
train_losses.append(avg_train_loss)
print(f"Train Loss: {avg_train_loss:.4f}")

# Validation loop
model.eval()
val_loss = 0.0
with torch.no_grad():
for anchor, positive, negative in val_loader:
anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
emb_anchor = model(anchor)
emb_pos = model(positive)
emb_neg = model(negative)

loss = triplet_loss(emb_anchor, emb_pos, emb_neg)
val_loss += loss.item()

avg_val_loss = val_loss / len(val_loader)
val_losses.append(avg_val_loss)
print(f"Val Loss: {avg_val_loss:.4f}")

# Save best model
if epoch == 0 or avg_val_loss < min(val_losses[:-1]):
torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))

# Plot
plot_metrics(train_losses, val_losses, save_path=plot_dir)

