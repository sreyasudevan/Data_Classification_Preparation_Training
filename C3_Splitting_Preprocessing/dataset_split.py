"""
Task 3: Dataset Splitting and Preprocessing (C3)
Author: Sreya Sudevan

This script splits the cleaned and balanced dataset into train, validation, and test sets,
while applying optional preprocessing such as resizing and normalization.
"""

import os
import random
import shutil
from PIL import Image
from tqdm import tqdm

# === Configuration ===
base_path = '/content/drive/MyDrive/archive/augmented_dataset'
output_path = '/content/drive/MyDrive/archive/final_dataset'
split_ratios = (0.7, 0.15, 0.15)
image_size = (224, 224)

# Create folders
splits = ['train', 'val', 'test']
for split in splits:
    for cls in os.listdir(base_path):
        os.makedirs(os.path.join(output_path, split, cls), exist_ok=True)

def preprocess_image(src_path, dest_path):
    """Resize and normalize image before saving."""
    try:
        img = Image.open(src_path).convert("RGB")
        img = img.resize(image_size)
        img.save(dest_path)
    except Exception as e:
        print(f" Error processing {src_path}: {e}")

for cls in os.listdir(base_path):
    cls_path = os.path.join(base_path, cls)
    if not os.path.isdir(cls_path):
        continue

    images = os.listdir(cls_path)
    random.shuffle(images)

    train_split = int(len(images) * split_ratios[0])
    val_split = int(len(images) * split_ratios[1])

    datasets = {
        'train': images[:train_split],
        'val': images[train_split:train_split+val_split],
        'test': images[train_split+val_split:]
    }

    for split, img_list in datasets.items():
        print(f"\nProcessing {cls} -> {split} ({len(img_list)} images)")
        for img_file in tqdm(img_list):
            src = os.path.join(cls_path, img_file)
            dest = os.path.join(output_path, split, cls, img_file)
            preprocess_image(src, dest)

print("\nDataset splitting and preprocessing completed!")
