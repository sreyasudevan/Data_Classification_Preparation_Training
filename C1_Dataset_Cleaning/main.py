"""
Task 1: Dataset Cleaning and Validation (C1)
Author: Sreya Sudevan

This script ensures that the image classification dataset is clean, uniform, and ready for model training.
It:
- Removes corrupted/unreadable files
- Converts inconsistent file formats to .jpg
- Ensures class structure matches classes.txt
- Generates a report of valid and removed images
"""

import os
from PIL import Image, UnidentifiedImageError
import json
from tqdm import tqdm
base_path = '/content/drive/MyDrive/archive'

train_path = os.path.join(base_path, 'train')
test_path = os.path.join(base_path, 'test')

print("Train classes:", os.listdir(train_path))
print("Test classes:", os.listdir(test_path))
classes = ['angry','disgust','fear','happy','neutral','sad','surprise']
classes_file = os.path.join(base_path, 'classes.txt')

with open(classes_file, 'w') as f:
    f.write('\n'.join(classes))

print("classes.txt created at:", classes_file)

def is_image_file(filename):
    return filename.lower().endswith(('.jpg', '.jpeg', '.png'))

def ensure_jpg(img_path, converted_files=None):
    """Convert to jpg if not already, and log converted files"""
    if not img_path.lower().endswith(".jpg"):
        new_path = os.path.splitext(img_path)[0] + ".jpg"
        try:
            with Image.open(img_path).convert("RGB") as img:
                img.save(new_path, "JPEG")
            os.remove(img_path)
            if converted_files is not None:
                converted_files.append(os.path.basename(img_path))
            return new_path
        except Exception as e:
            print(f"Error converting {img_path}: {e}")
            return None
    return img_path
    
def clean_dataset(dataset_path, classes_file, output_report, converted_log, removed_log):
    with open(classes_file, 'r') as f:
        valid_classes = [line.strip() for line in f.readlines()]

    report = {}
    converted_files = []
    removed_files = []

    for class_name in valid_classes:
        class_folder = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_folder):
            print(f"⚠️ Missing folder for class: {class_name}")
            continue

        valid_images = 0
        removed_images = 0

        for img_file in tqdm(os.listdir(class_folder), desc=f"Checking {class_name}"):
            img_path = os.path.join(class_folder, img_file)

            # Remove non-image files
            if not is_image_file(img_path):
                os.remove(img_path)
                removed_files.append(img_file)
                removed_images += 1
                continue

            # Verify image and convert if needed
            try:
                with Image.open(img_path) as img:
                    img.verify()  # check corruption
                new_path = ensure_jpg(img_path, converted_files)
                if new_path:
                    valid_images += 1
                else:
                    removed_files.append(img_file)
                    removed_images += 1
            except (UnidentifiedImageError, OSError, ValueError):
                os.remove(img_path)
                removed_files.append(img_file)
                removed_images += 1

        report[class_name] = {
            "valid_images": valid_images,
            "removed_corrupted_or_invalid": removed_images
        }

    # Save JSON report
    with open(output_report, 'w') as f:
        json.dump(report, f, indent=4)

    # Save converted and removed files to text logs
    with open(converted_log, 'w') as f:
        f.write('\n'.join(converted_files))
    with open(removed_log, 'w') as f:
        f.write('\n'.join(removed_files))

    # Print summary
    print(f"\nCleaning completed for {dataset_path}")
    print(f"Report saved to: {output_report}")
    print(f"Converted files logged in: {converted_log} ({len(converted_files)})")
    print(f"Removed files logged in: {removed_log} ({len(removed_files)})")

    return report
train_report_file = os.path.join(base_path, 'train_clean_report.json')
test_report_file = os.path.join(base_path, 'test_clean_report.json')

train_converted_log = os.path.join(base_path, 'train_converted.txt')
train_removed_log = os.path.join(base_path, 'train_removed.txt')

test_converted_log = os.path.join(base_path, 'test_converted.txt')
test_removed_log = os.path.join(base_path, 'test_removed.txt')

train_report = clean_dataset(train_path, classes_file, train_report_file, train_converted_log, train_removed_log)
test_report = clean_dataset(test_path, classes_file, test_report_file, test_converted_log, test_removed_log)

print("\nTrain dataset report:")
print(train_report)
print("\nTest dataset report:")
print(test_report)
