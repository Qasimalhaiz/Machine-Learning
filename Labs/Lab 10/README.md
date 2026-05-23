# ARTI308 – Lab 10: Multi-Class Classification & Hyperparameter Optimization Using Support Vector Machines (SVM)

This repository contains a structured, end-to-end predictive machine learning pipeline that implements a **Support Vector Machine (SVM)** classifier optimized via an exhaustive grid search grid to categorize multi-class morphological biological features.

## Project Objective
The primary goal of this project is to build, evaluate, and tune a robust **Support Vector Classifier (SVC)** using the classic **Iris Flower Dataset**. The workflow systematically focuses on identifying maximum-margin separating hyperplanes and tuning regularization and non-linear kernel parameters ($C$ and $\gamma$) to maximize structural classification accuracy.

---

## Dataset Profile (`Iris Dataset`)
The dataset evaluates morphological variations across 150 unique floral samples evenly distributed across three distinct target classifications:
* **Target Species**: `Iris Setosa`, `Iris Virginica`, and `Iris Versicolor`.
* **Quantitative Morphological Features**: 
  * `Sepal Length` (cm)
  * `Sepal Width` (cm)
  * `Petal Length` (cm)
  * `Petal Width` (cm)

---

## Technical Methodology & Pipeline Architecture

### 1. Exploratory Data Analysis & Dimensionality Inspection
* Utilizing Seaborn **Pairplots** to map multi-variable coordinate spaces, evaluating the cluster density, linear separability, and geometric boundaries between different target species.
* Generating two-dimensional **Kernel Density Estimate (KDE) plots** to isolate and analyze structural probability distributions across sepal dimensions.
* **Analytical Inferences:** Visual analytics systematically proved that while `Iris Versicolor` and `Iris Virginica` exhibit overlapping boundary coordinates, the `Iris Setosa` species remains perfectly linearly separable from the other two clusters.

### 2. Baseline Model Architecture
* Partitioning data structures into an independent **70% Training** split and a **30% Testing** validation split to strictly isolate parameter learning from performance generalization metrics.
* Instantiating an ordinary Support Vector Classifier (`SVC`) from Scikit-learn utilizing a Radial Basis Function (`rbf`) kernel to project non-linear boundaries into higher-dimensional vector spaces.

### 3. Exhaustive Hyperparameter Tuning via Grid Search
To optimize boundary soft-margins and prevent structural underfitting or overfitting, an automated `GridSearchCV` pipeline was configured to test 25 unique hyperparameter pairings across an exhaustive parameter grid:
* **Regularization Parameter ($C$):** `[0.1, 1, 10, 100, 1000]` — Controls the operational trade-off between maximizing the decision boundary margin width and minimizing training vector classification errors.
* **Kernel Coefficient ($\gamma$):** `[1, 0.1, 0.01, 0.001, 0.0001]` — Defines the structural radius of influence for individual support vectors mapping the non-linear decision spaces.

### 4. Classification Assessment & Evaluation
Quantifying classification margins and structural errors on hidden test matrices through dual-performance scoring blocks:
* **Confusion Matrix:** Map out exact True Positive, True Negative, False Positive, and False Negative counts to identify target cross-misclassifications.
* **Classification Reports:** Documenting exact performance ratios across precision, recall, and comprehensive $F_1$-score dimensions.
* **Final Results:** The fully optimized Support Vector Machine configuration successfully eliminated misclassification penalties, securing a superior cross-validated **predictive accuracy of ~98%**.

---

## Core Technical Stack
* **Language:** Python
* **Data Manipulation & Processing:** Pandas, NumPy
* **Data Visualization & Plotting:** Matplotlib, Seaborn
* **Predictive Machine Learning Engine:** Scikit-learn (`svm.SVC`, `model_selection.GridSearchCV`, `model_selection.train_test_split`, `metrics`)
