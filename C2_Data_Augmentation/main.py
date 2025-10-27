"""
Task 2: Class Balancing and Augmentation (C2)
Author: Sreya Sudevan

This script balances image dataset classes by applying augmentation (rotation, flip, brightness, contrast).
It ensures each class has equal representation before training.
"""

import os
import json
import random
from PIL import Image, ImageEnhance
from tqdm import tqdm
from google.colab import drive
drive.mount('/content/drive')

base_path = '/content/drive/MyDrive/archive'
cleaned_train_path = os.path.join(base_path, 'train')
augmented_path = os.path.join(base_path, 'augmented_dataset')

os.makedirs(augmented_path, exist_ok=True)


with open(os.path.join(base_path, 'classes.txt'), 'r') as f:
    classes = [line.strip() for line in f.readlines()]

def augment_image(img):
    # Random rotation
    angle = random.randint(-20, 20)
    img = img.rotate(angle)

    # Random horizontal flip
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Random brightness adjustment
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))

    # Random contrast adjustment
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))

    return img
    
class_counts = {cls: len(os.listdir(os.path.join(cleaned_train_path, cls))) for cls in classes}
max_count = max(class_counts.values())

report_before = class_counts.copy()
report_after = {}

for cls in classes:
    print(f"\nProcessing class: {cls}")
    src_folder = os.path.join(cleaned_train_path, cls)
    dest_folder = os.path.join(augmented_path, cls)
    os.makedirs(dest_folder, exist_ok=True)

    # Skip if source folder missing
    if not os.path.exists(src_folder):
        print(f"Skipping {cls} - folder not found")
        continue

    images = os.listdir(src_folder)
    if len(images) == 0:
        print(f" No images found in {cls}")
        continue

    current_count = len(images)
    needed = max_count - current_count

    # Copy original images safely
    for img_file in images:
        img_path = os.path.join(src_folder, img_file)
        dest_path = os.path.join(dest_folder, img_file)
        try:
            img = Image.open(img_path)
            img.save(dest_path)
        except Exception as e:
            print(f" Skipped {img_file} in {cls}: {e}")
            continue

    #  Balancing augmentation
    if needed > 0:
        for i in tqdm(range(needed), desc=f"Balancing {cls}"):
            try:
                img_file = random.choice(images)
                img_path = os.path.join(src_folder, img_file)
                img = Image.open(img_path)
                aug_img = augment_image(img)
                aug_name = f"{cls}_aug_{i+1:04d}.jpg"
                aug_img.save(os.path.join(dest_folder, aug_name))
            except Exception as e:
                print(f" Augmentation failed for {cls}: {e}")
                continue

    # Diversity augmentation (for all classes)
    extra_aug = int(current_count * 0.1)
    for i in tqdm(range(extra_aug), desc=f"Extra variety for {cls}"):
        try:
            img_file = random.choice(images)
            img_path = os.path.join(src_folder, img_file)
            img = Image.open(img_path)
            aug_img = augment_image(img)
            aug_name = f"{cls}_extra_{i+1:04d}.jpg"
            aug_img.save(os.path.join(dest_folder, aug_name))
        except Exception as e:
            print(f" Extra augmentation failed for {cls}: {e}")
            continue

    report_after[cls] = len(os.listdir(dest_folder))
    print(f"Finished {cls}: total {report_after[cls]} images")

# Save summary report
report = {
    "before_augmentation": report_before,
    "after_augmentation": report_after
}
with open(os.path.join(base_path, 'augmentation_report.json'), 'w') as f:
    json.dump(report, f, indent=4)

print("\nAugmentation completed.")
print("Summary report saved to:", os.path.join(base_path, 'augmentation_report.json'))
