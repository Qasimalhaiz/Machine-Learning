# ARTI308 – Lab 8: Instance-Based Classification using K-Nearest Neighbors

This repository contains a structured, end-to-end predictive machine learning pipeline implementing the **K-Nearest Neighbors (KNN)** classification algorithm on a highly dimensional, anonymized quantitative profile dataset.

## Project Objective
The core objective of this project is to build an optimized, robust classification model that predicts a binary categorical output (`TARGET CLASS`) based on a series of feature variables. The workflow systematically focuses on data scaling prerequisites and hyperparameter tuning using the Elbow Method to identify the optimal neighbor count ($K$) that minimizes classification error rates.

---

## Dataset Profile (`KNN_Project_Data.csv`)
The dataset consists of anonymized continuous numerical features serving as predictors for classification:
* **Features (`XVPM`, `GWYH`, `TRAT`, etc.)**: Highly dimensional, continuous numerical variables requiring normalization due to variance in raw coordinate scales.
* **`TARGET CLASS` (Target Variable)**: The binary ground-truth categorical label (0 or 1) to be classified by the instance-based model.

---

## Technical Methodology & Pipeline Architecture

### 1. Exploratory Data Auditing
* Loading the feature matrix and assessing structural data configurations.
* Auditing baseline column profiles to confirm the absence of null, missing, or corrupt records.

### 2. Feature Standardization (Critical for Distance-Based Estimators)
Because the K-Nearest Neighbors classifier computes the geometric Euclidean distance between data points to evaluate similarity, features with inherently larger mathematical scales can artificially bias and dominate the model's decision boundaries.
* Utilizing `StandardScaler` from `scikit-learn` to center all independent feature parameters around a mean of 0 with a standard deviation of 1.
* Transforming raw input vectors into a uniformly balanced, un-biased coordinate feature workspace.

### 3. Training & Hyperparameter Optimization (The Elbow Method)
* Partitioning the normalized feature coordinates ($X$) and objective vectors ($y$) into independent **70% Training** and **30% Testing** allocations to prevent evaluation data leakage.
* Conducting a systematic hyperparameter sweep by testing every successive neighbor value from $K=1$ through $K=40$.
* Computing the mean classification error rate at each step and generating an **Error Rate vs. K Value** line graph to visually pinpoint the "Elbow" threshold—the exact point where the testing error stabilizes at its absolute lowest plateau.

### 4. Classification Assessment
Quantifying model precision and error distributions on hidden validation matrices using dual-performance scoring blocks:
* **Confusion Matrix:** Isolating exact counts for True Positives, True Negatives, False Positives, and False Negatives to detect directional classification biases.
* **Classification Report:** Computing exact performance ratios across precision, recall, and comprehensive $F_1$-score dimensions.

---

## Technical Stack
* **Language:** Python
* **Data Manipulation & Engineering:** Pandas, NumPy
* **Data Visualization & Plotting:** Matplotlib, Seaborn
* **Predictive Machine Learning Engine:** Scikit-learn (`preprocessing.StandardScaler`, `neighbors.KNeighborsClassifier`, `metrics.classification_report`)
