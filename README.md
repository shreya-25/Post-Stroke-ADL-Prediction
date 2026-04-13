# Post-Stroke ADL Prediction

This repository presents a machine learning project focused on **predicting post-stroke Activities of Daily Living (ADL) outcomes** using clinical data. The project explores how supervised learning models can be used to classify rehabilitation-related outcomes from patient features and compares the effect of dimensionality reduction using **Principal Component Analysis (PCA)**.

The repository includes a complete Python-based workflow for preprocessing data, applying PCA, training multiple machine learning models, evaluating classification performance, and visualizing results. The current repository contains `Final_Code.py`, `stroke data 2.xlsx`, `scree plot.png`, and `feature importance ranking.png`.

## Project Overview

Post-stroke recovery varies across patients, and early prediction of ADL-related outcomes can help clinicians and researchers better understand rehabilitation patterns and support decision-making. This project builds a comparative machine learning pipeline to predict ADL outcomes from structured stroke-related data.

The workflow includes:

- data loading from an Excel dataset
- feature scaling
- train-test split
- dimensionality reduction using PCA
- classification using multiple machine learning models
- confusion matrix and classification report evaluation
- cross-validation for model comparison

## Repository Structure

```bash
Post-Stroke-ADL-Prediction/
│
├── Final_Code.py                  # Main end-to-end machine learning pipeline
├── stroke data 2.xlsx             # Input dataset
├── scree plot.png                 # PCA explained variance visualization
├── feature importance ranking.png # Feature importance result visualization
└── README.md
