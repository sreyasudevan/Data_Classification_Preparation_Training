## Task 2 — Dataset Augmentation
# Overview

This task increases dataset diversity and balances class distribution by applying image augmentations such as rotations, flips, and brightness adjustments.
The result is a more robust and balanced dataset suitable for model training.

# Prerequisites

Python 3.8 or higher

Cleaned dataset from Task 1 (archive/train/)

Required libraries: os, cv2, numpy, PIL, random, json

Google Drive if using Colab

# Setup Instructions

Mount Google Drive:

from google.colab import drive
drive.mount('/content/drive')

Ensure cleaned dataset exists at:
/content/drive/MyDrive/archive/train


# Install dependencies:

pip install opencv-python pillow numpy

# Steps to Run the App

Run:

python data_augmentation.py


The script will:

Apply augmentations (rotation, flip, brightness, etc.)

Generate balanced samples across emotion classes

Save augmented images into:

/content/drive/MyDrive/archive/augmented_dataset

# Output

augmented_dataset/ — Balanced, augmented dataset

augmentation_report.json — Summary of images before and after augmentation
