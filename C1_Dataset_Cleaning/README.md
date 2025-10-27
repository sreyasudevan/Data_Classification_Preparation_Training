## Task 1 — Dataset Cleaning
# Overview

This task cleans the dataset by removing corrupted or unreadable image files, converting all image formats to .jpg, and validating the dataset structure.
It ensures that the data is consistent and ready for augmentation or model training.

# Prerequisites

Python 3.8 or higher

Required libraries: os, cv2, json, PIL, imghdr

# Dataset structure:

archive/
├── train/
└── test/

# Setup Instructions

Mount Google Drive if using Google Colab:

from google.colab import drive
drive.mount('/content/drive')


Place your dataset inside:

/content/drive/MyDrive/archive/train
/content/drive/MyDrive/archive/test


# Install dependencies:

pip install opencv-python pillow

# Steps to Run the App

Execute the script:

python data_cleaning.py


The script will:

Verify and validate all image files

Convert non-JPG files to .jpg

Delete corrupted or invalid files

Generate logs and reports
