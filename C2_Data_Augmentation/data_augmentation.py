"""
Task 2: Class Balancing and Augmentation (C2)


Description:

This script balances an image dataset across all classes using augmentation techniques.  
It ensures that each class has approximately the same number of images to prevent model bias during training.

Key Operations:
1. Reads all class names from `classes.txt`.
2. Calculates the number of images in each class.
3. Applies augmentation (rotation, flipping, brightness, contrast) to underrepresented classes.
4. Saves augmented images to a new directory: `augmented_dataset`.
5. Generates a JSON report summarizing image counts before and after augmentation.
"""

import os
import json
import random
from PIL import Image, ImageEnhance
from tqdm import tqdm
from google.colab import drive

# Mount Google Drive (required for accessing dataset stored on Drive)
drive.mount('/content/drive')

# Define dataset paths
base_path = '/content/drive/MyDrive/archive'
cleaned_train_path = os.path.join(base_path, 'train')          # Input: cleaned dataset (from Task 1)
augmented_path = os.path.join(base_path, 'augmented_dataset')  # Output: augmented dataset

# Create augmented dataset directory if not present
os.makedirs(augmented_path, exist_ok=True)

# Load class names from classes.txt
with open(os.path.join(base_path, 'classes.txt'), 'r') as f:
    classes = [line.strip() for line in f.readlines()]


def augment_image(img):
    """
    Apply random augmentation transformations to a given image.

    Augmentations applied:
    - Random rotation between -20° and +20°
    - Random horizontal flip
    - Random brightness adjustment (±20%)
    - Random contrast adjustment (±20%)

    Parameters:
        img (PIL.Image): Input image object.

    Returns:
        PIL.Image: Augmented image.
    """
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


# Count number of images in each class
class_counts = {cls: len(os.listdir(os.path.join(cleaned_train_path, cls))) for cls in classes}
max_count = max(class_counts.values())  # Target count for balancing

report_before = class_counts.copy()  # Record pre-augmentation image counts
report_after = {}                    # To store post-augmentation image counts


# Iterate through each class to balance and augment
for cls in classes:
    print(f"\nProcessing class: {cls}")
    src_folder = os.path.join(cleaned_train_path, cls)
    dest_folder = os.path.join(augmented_path, cls)
    os.makedirs(dest_folder, exist_ok=True)

    # Skip if class folder does not exist
    if not os.path.exists(src_folder):
        print(f"Skipping {cls} - folder not found.")
        continue

    images = os.listdir(src_folder)
    if len(images) == 0:
        print(f"No images found in {cls}.")
        continue

    current_count = len(images)
    needed = max_count - current_count  # Number of additional images required to balance

    # Step 1: Copy all original images to augmented folder
    for img_file in images:
        img_path = os.path.join(src_folder, img_file)
        dest_path = os.path.join(dest_folder, img_file)
        try:
            img = Image.open(img_path)
            img.save(dest_path)
        except Exception as e:
            print(f"Skipped {img_file} in {cls}: {e}")
            continue

    # Step 2: Apply augmentation to balance dataset size
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
                print(f"Augmentation failed for {cls}: {e}")
                continue

    # Step 3: Add extra 10% variety augmentations for every class
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
            print(f"Extra augmentation failed for {cls}: {e}")
            continue

    # Record final image count
    report_after[cls] = len(os.listdir(dest_folder))
    print(f"Finished {cls}: total {report_after[cls]} images.")


# Save final augmentation summary report
report = {
    "before_augmentation": report_before,
    "after_augmentation": report_after
}

with open(os.path.join(base_path, 'augmentation_report.json'), 'w') as f:
    json.dump(report, f, indent=4)

# Display final summary
print("\nAugmentation completed successfully.")
print("Summary report saved to:", os.path.join(base_path, 'augmentation_report.json'))
