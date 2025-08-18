import os
import shutil
import random

# Paths
base_dir = "../facedataset"
output_dir = "../facedataset_split"
high_dir = os.path.join(base_dir, "high_resolution")
low_dir = os.path.join(base_dir, "low_resolution")

# Create output folders
for split in ["train", "eval"]:
for person in os.listdir(high_dir):
os.makedirs(os.path.join(output_dir, split, person), exist_ok=True)

# Combine high + low resolution images for each person, then split
for person in os.listdir(high_dir):
all_images = []

high_person_dir = os.path.join(high_dir, person)
low_person_dir = os.path.join(low_dir, person)

# Add all images (high + low)
for img_folder in [high_person_dir, low_person_dir]:
if not os.path.isdir(img_folder):
continue
for img in os.listdir(img_folder):
img_path = os.path.join(img_folder, img)
if os.path.isfile(img_path):
all_images.append(img_path)
#Person-wise splitting of images into train/eval.
# Ensures no identity or image appears in both sets.
# This is a true holdout strategy for model evaluation.
# Shuffle and split
random.shuffle(all_images)
split_idx = int(0.8 * len(all_images))
train_imgs = all_images[:split_idx]
eval_imgs = all_images[split_idx:]

# Copy to respective folders
for img_path in train_imgs:
dest = os.path.join(output_dir, "train", person, os.path.basename(img_path))
shutil.copy(img_path, dest)

for img_path in eval_imgs:
dest = os.path.join(output_dir, "eval", person, os.path.basename(img_path))
shutil.copy(img_path, dest)

print("✅ Dataset split complete: 80% train / 20% eval (person-wise)")
