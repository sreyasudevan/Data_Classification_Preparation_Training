# Data_Classification_Preparation_Training
Complete dataset cleaning, augmentation, and preprocessing pipeline 


This repository provides a complete image dataset preparation pipeline for machine learning classification tasks.  
It is divided into three major modules:

1. **C1_Dataset_Cleaning** – Cleans and validates image data.
2. **C2_Data_Augmentation** – Balances classes using data augmentation.
3. **C3_Splitting_Preprocessing** – Splits and preprocesses the dataset for training.

---

## Features
- Detects and removes corrupted or invalid image files.
- Converts image formats to a consistent standard (.jpg).
- Balances dataset classes using augmentation.
- Splits dataset into train, validation, and test sets.
- Resizes and prepares images for model input.

---

## Folder Structure

Data_Classification_Preparation_Training/
├── T1_Data_Cleaning/
│ ├── data_cleaning.py
│ └── README.md
│
├── T2_Data_Augmentation/
│ ├── data_augmentation.py
│ └── README.md
│
├── T3_Dataset_Split/
│ ├── dataset_split.py
│ └── README.md
│
├── TASK_REPORT.md
├── requirements.txt
└── README.md (Main project README)


---

## Prerequisites
- Python 3.10 or higher
- PIL (Pillow), tqdm, json, os libraries
- Google Colab or local Python environment
- Required libraries:
os, cv2, numpy, PIL, shutil, json, random

---

## Setup Instructions
```bash
# Clone this repository
git clone https://github.com/sreyasudevan/Data_Classification_Preparation_Training.git
cd Data_Classification_Preparation_Training

# Install dependencies
pip install -r requirements.txt
