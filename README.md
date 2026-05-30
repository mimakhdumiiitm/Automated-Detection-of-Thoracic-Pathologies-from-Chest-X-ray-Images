# Thoracic Pathology Detection from Chest X-Rays

This repository contains a PyTorch-based Deep Learning pipeline designed to classify 20 distinct thoracic pathologies from chest X-ray images.

> **Course Project:** This project was developed for **NPPE 1** of the *Introduction to Deep Learning and Gen AI* course, part of the **IIT Madras BS Degree in Data Science and Applications**.

---

## 🔗 Links

* **Kaggle Competition:** [View Competition Layout & Data](https://www.kaggle.com/competitions/26-t-1-dl-gen-ainppe-1)
* **My Notebook/Work:** [View Full Implementation on Kaggle](https://www.kaggle.com/code/mimakhdumiiitm/22f3001418-notebook-26t1)

---

## 🛠️ Key Features

* **Backbone Architecture:** Built using an **EfficientNet-B3** pretrained model with a custom multi-class classification head.
* **Imbalance Handling:** Utilizes a custom **Focal Loss** implementation combined with a `WeightedRandomSampler` to handle extreme class disparities across the 20 diseases (e.g., highly dominant 'No Finding' labels).
* **Data Augmentation:** Features an image pipeline with geometric and photometric transforms (`RandomHorizontalFlip`, `RandomRotation`, `ColorJitter`, `RandomAffine`).
* **Robust Inference:** Implements **Test-Time Augmentation (TTA)** across multiple passes (original, flipped, and rotated) to smooth and stabilize final class predictions via averaged softmax probabilities.
