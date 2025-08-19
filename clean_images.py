from PIL import Image
import os

def is_valid_image(filepath):
try:
with Image.open(filepath) as img:
img.verify()
return True
except:
return False

base_path = "../facedataset_split"
for split in ["train", "eval"]:
split_path = os.path.join(base_path, split)
for person in os.listdir(split_path):
person_path = os.path.join(split_path, person)
for file in os.listdir(person_path):
full_path = os.path.join(person_path, file)
if not is_valid_image(full_path):
print(f"❌ Removing corrupt image: {full_path}")
os.remove(full_path)
print("✅ Cleaning complete.")
To run:
python3 clean_images.py
