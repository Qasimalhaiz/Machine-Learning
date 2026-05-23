# ARTI308 – Lab 7: Binary Classification using Logistic Regression

This repository features a complete end-to-end data processing and supervised machine learning pipeline that utilizes **Logistic Regression** to analyze user demographic profiles and predictive web activity.

## Project Objective
The main goal is to build and evaluate a binary classification model that accurately determines whether an individual website visitor will click on a targeted advertisement banner (`Clicked on Ad`). This provides actionable, data-driven optimization strategies for digital marketing teams.

---

## Dataset Attributes (`advertising.csv`)
The evaluation dataset contains 1,000 distinct internet user entries tracking the following metrics:
* **`Daily Time Spent on Site`**: Active user session lengths on the platform recorded in minutes.
* **`Age`**: Customer baseline age profile in years.
* **`Area Income`**: Average geographic salary distribution level of the consumer's residential region.
* **`Daily Internet Usage`**: Average daily minutes spent browsing the web globally by the visitor.
* **`Ad Topic Line`**: Text string assigned to the targeted advertisement banner headline.
* **`City` / `Country`**: Geographical position identifiers of the site traffic source.
* **`Male`**: Binary gender feature tag (1 = Male, 0 = Female).
* **`Timestamp`**: High-precision date and time marker when the ad interaction or bounce log took place.
* **`Clicked on Ad` (Target Label)**: Supervised ground-truth classification marker (1 = Interacted/Clicked, 0 = Bounced/No Click).

---

## Technical Methodology

### 1. Exploratory Data Analysis & Feature Selection
* Building distributed histograms tracking user demographic categories like `Age` splits.
* Generating Seaborn pairplots mapping multi-variable distribution patterns isolated explicitly by the target label (`Clicked on Ad`).
* Removing unneeded, text-heavy categorical dimensions (`Ad Topic Line`, `City`, `Country`, `Timestamp`) to establish a clean, continuous numerical array.

### 2. Predictive Optimization 
* Partitioning the independent inputs ($X$) and objective labels ($y$) into structured training and testing sets.
* Fitting an ordinary `LogisticRegression` classifier from `scikit-learn` with increased iteration limits to guarantee maximum convergence accuracy.

### 3. Comprehensive Model Evaluation
Evaluating the classifier's predictive boundaries on hidden test matrices using structural scoring blocks:
* **Confusion Matrix**: Capturing exact True Positive, True Negative, False Positive, and False Negative counts.
* **Classification Report**: Computing direct, highly accurate performance values for precision, recall, and overall $F_1$-score ratios.

---

## Core Technical Stack
* **Language**: Python
* **Data Cleansing**: Pandas, NumPy
* **Data Visualization**: Matplotlib, Seaborn
* **Predictive ML Framework**: Scikit-learn (linear_model, model_selection, metrics)
