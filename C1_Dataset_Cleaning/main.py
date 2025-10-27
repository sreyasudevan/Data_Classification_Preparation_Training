"""
Task 1: Dataset Cleaning and Validation (C1)


Description:

This script ensures that the image classification dataset is clean, uniform, and ready for model training.

It performs the following operations:
1. Removes corrupted or unreadable image files.
2. Converts all image formats (e.g., PNG, JPEG) to `.jpg` for consistency.
3. Ensures the dataset’s folder structure matches the class names listed in `classes.txt`.
4. Generates a detailed JSON report summarizing valid and removed images per class.
5. Logs all converted and removed image filenames in separate `.txt` files.
"""

import os
from PIL import Image, UnidentifiedImageError
import json
from tqdm import tqdm

# Define dataset base path (adjust this if running outside Colab)
base_path = '/content/drive/MyDrive/archive'

# Define train and test directories
train_path = os.path.join(base_path, 'train')
test_path = os.path.join(base_path, 'test')

print("Train classes:", os.listdir(train_path))
print("Test classes:", os.listdir(test_path))

# Create a reference file (classes.txt) containing valid class names
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
classes_file = os.path.join(base_path, 'classes.txt')

with open(classes_file, 'w') as f:
    f.write('\n'.join(classes))

print("classes.txt created at:", classes_file)


def is_image_file(filename):
    """
    Check whether a given file is a valid image file.

    Parameters:
        filename (str): Path or name of the file to check.

    Returns:
        bool: True if the file has an image extension, False otherwise.
    """
    return filename.lower().endswith(('.jpg', '.jpeg', '.png'))


def ensure_jpg(img_path, converted_files=None):
    """
    Convert a given image to .jpg format if it isn't already.
    Deletes the original file after conversion.

    Parameters:
        img_path (str): Path of the image to convert.
        converted_files (list): Optional list to record names of converted files.

    Returns:
        str or None: New image path if converted or already a .jpg file;
                     None if conversion fails.
    """
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
    """
    Clean the dataset by validating, converting, and removing invalid images.

    Parameters:
        dataset_path (str): Path to the dataset directory (train/test).
        classes_file (str): Path to the text file containing valid class names.
        output_report (str): Path to save the JSON cleaning report.
        converted_log (str): Path to save names of converted images.
        removed_log (str): Path to save names of removed/corrupted images.

    Returns:
        dict: Summary report of the cleaning process (valid and removed counts per class).
    """
    # Load valid class names
    with open(classes_file, 'r') as f:
        valid_classes = [line.strip() for line in f.readlines()]

    report = {}
    converted_files = []
    removed_files = []

    # Process each class folder
    for class_name in valid_classes:
        class_folder = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_folder):
            print(f"⚠️ Missing folder for class: {class_name}")
            continue

        valid_images = 0
        removed_images = 0

        # Iterate through each file in the class folder
        for img_file in tqdm(os.listdir(class_folder), desc=f"Checking {class_name}"):
            img_path = os.path.join(class_folder, img_file)

            # Remove non-image files
            if not is_image_file(img_path):
                os.remove(img_path)
                removed_files.append(img_file)
                removed_images += 1
                continue

            # Validate image integrity and convert to .jpg if necessary
            try:
                with Image.open(img_path) as img:
                    img.verify()  # Check if the image can be opened
                new_path = ensure_jpg(img_path, converted_files)
                if new_path:
                    valid_images += 1
                else:
                    removed_files.append(img_file)
                    removed_images += 1
            except (UnidentifiedImageError, OSError, ValueError):
                # Remove corrupted or unreadable files
                os.remove(img_path)
                removed_files.append(img_file)
                removed_images += 1

        # Store results for the current class
        report[class_name] = {
            "valid_images": valid_images,
            "removed_corrupted_or_invalid": removed_images
        }

    # Save the summary report as JSON
    with open(output_report, 'w') as f:
        json.dump(report, f, indent=4)

    # Save converted and removed file names as logs
    with open(converted_log, 'w') as f:
        f.write('\n'.join(converted_files))
    with open(removed_log, 'w') as f:
        f.write('\n'.join(removed_files))

    # Display summary
    print(f"\nCleaning completed for {dataset_path}")
    print(f"Report saved to: {output_report}")
    print(f"Converted files logged in: {converted_log} ({len(converted_files)})")
    print(f"Removed files logged in: {removed_log} ({len(removed_files)})")

    return report


# File paths for reports and logs
train_report_file = os.path.join(base_path, 'train_clean_report.json')
test_report_file = os.path.join(base_path, 'test_clean_report.json')

train_converted_log = os.path.join(base_path, 'train_converted.txt')
train_removed_log = os.path.join(base_path, 'train_removed.txt')

test_converted_log = os.path.join(base_path, 'test_converted.txt')
test_removed_log = os.path.join(base_path, 'test_removed.txt')

# Execute cleaning for both train and test datasets
train_report = clean_dataset(train_path, classes_file, train_report_file, train_converted_log, train_removed_log)
test_report = clean_dataset(test_path, classes_file, test_report_file, test_converted_log, test_removed_log)

# Print final summary
print("\n=== Train Dataset Report ===")
print(json.dumps(train_report, indent=4))
print("\n=== Test Dataset Report ===")
print(json.dumps(test_report, indent=4))
