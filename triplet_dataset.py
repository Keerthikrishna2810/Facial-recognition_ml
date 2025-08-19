import torch
from torch.utils.data import Dataset
import random
import os
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from collections import defaultdict

class TripletFaceDataset(Dataset):
def __init__(self, root_dir, transform=None, num_triplets=10000, image_size=(160, 160)):
self.root_dir = root_dir
self.num_triplets = num_triplets
self.image_size = image_size

# Define default transform if not provided
self.transform = transform or transforms.Compose([
transforms.Resize(self.image_size),
transforms.ToTensor(),
])

self.class_to_images = defaultdict(list)

# Populate valid image paths
for cls in os.listdir(root_dir):
cls_path = os.path.join(root_dir, cls)
if not os.path.isdir(cls_path):
continue
for img_name in os.listdir(cls_path):
img_path = os.path.join(cls_path, img_name)
if self._is_valid_image(img_path):
self.class_to_images[cls].append(img_path)

# Keep only classes with at least 2 images
self.class_to_images = {
k: v for k, v in self.class_to_images.items() if len(v) >= 2
}

self.classes = list(self.class_to_images.keys())
assert len(self.classes) >= 2, "Need at least 2 classes with 2+ images each."

def __len__(self):
return self.num_triplets

def __getitem__(self, idx):
anchor_class = random.choice(self.classes)
negative_class = random.choice([c for c in self.classes if c != anchor_class])

anchor_img_path, positive_img_path = random.sample(self.class_to_images[anchor_class], 2)
negative_img_path = random.choice(self.class_to_images[negative_class])

anchor = self._load_and_transform(anchor_img_path)
positive = self._load_and_transform(positive_img_path)
negative = self._load_and_transform(negative_img_path)

return anchor, positive, negative

def _load_and_transform(self, path):
try:
image = Image.open(path).convert("RGB")
except (UnidentifiedImageError, FileNotFoundError):
print(f"[Warning] Skipping unreadable image: {path}")
image = Image.new('RGB', self.image_size)

return self.transform(image)

def _is_valid_image(self, path):
try:
with Image.open(path) as img:
img.verify()
return True
except:
return False
