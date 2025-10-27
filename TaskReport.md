# Overview

This project focuses on preparing an image dataset for deep learning–based classification models.
The process was divided into three main tasks:

Dataset Cleaning
Dataset Augmentation
Dataset Splitting and Preprocessing

Each step ensures that the dataset becomes clean, balanced, and properly structured for high-quality model training.

# Task 1: Dataset Cleaning
Purpose

The main goal of this task was to clean and validate the dataset before training.
It ensures that only valid, readable, and properly formatted images remain in the dataset.

Key Steps

Verified image integrity using the Pillow library.
Converted non-JPG files (e.g., PNG, JPEG) into standardized .jpg format.
Removed corrupted, unreadable, and non-image files.
Generated detailed JSON reports and text logs of all conversions and deletions.

What I Learned

How to validate image datasets for deep learning applications.
The importance of file consistency and proper data handling.
Efficient use of libraries like PIL, os, and tqdm for automation.

Challenges Faced

Handling corrupted images that caused read errors during batch processing.
Ensuring smooth execution when class folders were missing or empty.
Managing Drive I/O speed when working in Colab.

# Task 2: Dataset Augmentation
Purpose

To increase dataset diversity and balance the number of samples across emotion classes.
Augmentation helps reduce overfitting and improve model generalization.

Key Steps

Applied random rotations, flips, brightness, and contrast adjustments using the Pillow library.
Balanced each class to the size of the largest class by generating augmented samples.
Created additional 10% “extra” augmentations to increase diversity.
Stored augmented images in a new folder and generated a summary report.

What I Learned

How to use image augmentation to balance classes effectively.
The impact of transformations like rotation and brightness on dataset variability.
How to automate dataset balancing dynamically using Python.

Challenges Faced

Memory limitations when augmenting large datasets.
Ensuring augmentations preserved essential image features without distortion.
Maintaining consistent image formats across all classes.

# Task 3: Dataset Split and Preprocessing
Purpose

To split the final dataset into training, validation, and testing subsets and resize images to uniform dimensions suitable for CNN-based models.

Key Steps

Split dataset into 70% training, 15% validation, and 15% testing.
Resized all images to a standard size (224×224 pixels).
Preserved class folder structures across all splits.
Generated a JSON summary report detailing image counts per split and per class.

What I Learned

How to perform stratified data splits programmatically.
The importance of uniform image dimensions for neural network inputs.
Use of automation for dataset structuring before training.
Challenges Faced
Handling uneven class distributions during random splits.
Managing drive space and runtime limits when processing large image batches.
Ensuring resized images maintained aspect ratio and clarity.

# Overall Learning Outcomes

Developed a complete preprocessing pipeline for image classification.
Gained experience with data cleaning, augmentation, and dataset management.
Learned to automate repetitive dataset preparation tasks efficiently.
Strengthened understanding of dataset quality’s impact on model performance.

# Technologies Used

Python 3.8+
Pillow (PIL) for image manipulation
OpenCV for image resizing
tqdm for progress tracking
Google Colab for execution and Drive integration

# Conclusion

By completing these three interconnected tasks, the dataset is now fully cleaned, balanced, and split for model training.
This project demonstrates a practical, end-to-end pipeline that transforms raw, unstructured image data into a ready-to-train dataset for deep learning applications.
