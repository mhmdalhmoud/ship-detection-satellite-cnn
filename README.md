# Ship Detection in Satellite Imagery

Detecting ships in satellite image chips with a custom convolutional neural network, compared against a classical HOG + SVM baseline, and extended to locate ships in a larger scene with a sliding window.

Built by **Mohammed Alhmoud** and **Laith Sweis** for the Computer Vision course (Phase II), B.Sc. Artificial Intelligence & Data Science, Al-Ahliyya Amman University, 2025/2026.

## Problem

Finding ships in satellite imagery by hand is slow. The dataset is also imbalanced: 3 of every 4 images contain no ship, so a naive model can score well by always answering "no ship" while missing the ships that matter.

## Dataset

[Ships in Satellite Imagery](https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery) (Kaggle, PlanetScope imagery): 4,000 RGB image chips of 80 × 80 pixels, 1,000 labelled "ship" and 3,000 "no ship". The dataset is not included in this repository.

## Approach

1. **Preprocessing:** images read with OpenCV, converted from BGR to RGB, scaled to [0, 1], and split into stratified train / validation / test sets (80 / 10 / 10) so every split keeps the 1:3 class ratio.
2. **Handling the imbalance:** a first model was trained with balanced class weights (scikit-learn `compute_class_weight`). The final model uses horizontal and vertical flip augmentation on the training set only and a reduced learning rate (Adam, 0.0001), which gave stable training.
3. **Custom CNN (TensorFlow / Keras):** four 3 × 3 convolution layers (32, 32, 64, 64 filters) with batch normalization, max pooling and dropout, followed by a 128-unit dense layer and a sigmoid output.
4. **Baseline:** HOG features (scikit-image) with an RBF-kernel SVM (scikit-learn).
5. **Localization:** a sliding window (80 × 80, configurable step) runs the CNN over a larger image and draws a bounding box wherever the ship probability is above a threshold. The demo in the notebook uses a scene assembled from dataset chips.

## Results

| Model | Test images | Accuracy | Ship recall | Ship precision |
|---|---|---|---|---|
| Custom CNN | 400 | 97% | 99% (99 of 100 ships) | 92% |
| HOG + SVM baseline | 800 | 99% | 96% | 98% |

The CNN missed only 1 of 100 ships, at the cost of more false alarms. The HOG + SVM baseline had higher overall accuracy on this dataset but missed more ships. The two were evaluated on different test splits (400 and 800 images), so the figures are not a strict head-to-head comparison.

## Tech stack

Python · TensorFlow / Keras · OpenCV · scikit-learn · scikit-image · NumPy · pandas · Matplotlib · Seaborn · Google Colab (Tesla T4 GPU)

## How to run

The notebook was written for Google Colab.

1. Download the dataset from Kaggle and save the archive to your Google Drive as `MyDrive/shipsnet.zip` (the path the notebook expects).
2. Open `CV_Project_PhaseII.ipynb` in Google Colab and select a GPU runtime.
3. Run the cells in order. The notebook mounts Google Drive, extracts the dataset, trains both models and prints the classification reports.

Some printed messages in the notebook are in Arabic.

## Repository contents

- `CV_Project_PhaseII.ipynb`: full pipeline, including preprocessing, the CNN, the HOG + SVM baseline, evaluation and the sliding-window demo.

## Authors

- Mohammed Alhmoud · [Portfolio](https://mhmdalhmoud.github.io) · [LinkedIn](https://www.linkedin.com/in/mohammed-alhmoud-559348234)
