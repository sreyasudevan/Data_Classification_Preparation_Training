## Task 3 — Dataset Split and Preprocessing
# Overview

This task splits the augmented dataset into training, validation, and testing sets (default 70/15/15) and resizes all images to a uniform size for model input.
It ensures proper organization and consistent image dimensions for efficient model training.

# Prerequisites

Python 3.8 or higher

Augmented dataset from Task 2 (archive/augmented_dataset/)

Required libraries: os, cv2, random, json, shutil

Google Drive if using Colab

# Setup Instructions

Mount Google Drive:

from google.colab import drive
drive.mount('/content/drive')

Ensure the augmented dataset is located at:
/content/drive/MyDrive/archive/augmented_dataset


# Install dependencies:

pip install opencv-python

Steps to Run the App

Execute the script:

python dataset_split.py


The script will:

Split each emotion class into training, validation, and test sets

Resize all images to (224x224) by default

Save the final dataset in:

/content/drive/MyDrive/archive/final_split_dataset

# Output

final_split_dataset/train/

final_split_dataset/val/

final_split_dataset/test/

split_report.json — Summary of dataset distribution
