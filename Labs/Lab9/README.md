# ARTI308 – Lab 9: Credit Risk Classification Using Decision Trees & Random Forests

This repository contains an end-to-end predictive machine learning workflow implementing supervised ensemble and non-ensemble tree-based models to evaluate loan default probabilities.

## Project Objective
The primary goal of this project is to build, optimize, and compare a standalone **Decision Tree Classifier** against a **Random Forest Ensemble** model. The system analyzes historical consumer credit profiles to classify whether a borrower will default on their financial obligations (`not.fully.paid`), providing vital risk assessment strategies for banking institutions.

---

## Dataset Profile (`loan_data.csv`)
The dataset evaluates historical lending parameters, tracking key financial indicators across borrowers:
* **Credit Metrics**: Includes variables such as `credit.score` (FICO ratings), `int.rate` (loan interest rates), `installment` (monthly payment commitments), and `log.annual.inc` (the natural log of the borrower's self-reported annual income).
* **Borrower Profile**: Incorporates metrics tracking financial stability, such as `dti` (debt-to-income ratio), `revol.bal` (revolving balance), and `revol.util` (revolving line utilization rate).
* **`not.fully.paid` (Target Categorical Variable)**: The binary ground-truth label indicating loan repayment outcomes (1 = Account defaulted / Not fully paid, 0 = Account successfully paid in full).

---

## Technical Methodology & Pipeline Steps

### 1. Exploratory Data Auditing & Preprocessing
* Utilizing Seaborn histograms and countplots to inspect FICO credit score distributions segmented by loan repayment outcomes.
* Developing trend visualizations to explore relationships between interest rates and credit profiles.
* Encoding categorical features (such as `purpose`) into binary dummy variables via Pandas to prepare structural inputs for the mathematical algorithms.

### 2. Predictive Modeling Engine
* **Standalone Decision Tree Classifier**: Fitting an optimized, non-parametric single decision tree to establish a clear baseline for interpretability and directional splitting rules.
* **Random Forest Ensemble Classifier**: Deploying an ensemble bagging model consisting of multiple bootstrapped decision trees to reduce overall model variance and combat overfitting.

### 3. Structural Model Evaluation & Comparative Metrics
Evaluating performance using classification matrices, classification reports, and weighted precision/recall benchmarks to highlight the operational strengths of each approach:

| Model Machine Learning Architecture | Overall Accuracy | Operational Character & Behavioral Trade-offs |
| :--- | :---: | :--- |
| **Single Decision Tree** | **73%** | Lower overall accuracy due to higher variance, but displays significantly higher sensitivity (Recall) toward detecting high-risk borrowers who default. |
| **Random Forest Ensemble** | **85%** | Delivers superior overall predictive accuracy, but suffers from data-imbalance biases, ultimately missing a larger portion of actual financial defaulters. |

---

## Core Technical Stack
* **Language**: Python
* **Data Processing & Engineering**: Pandas, NumPy
* **Data Visualization & Plotting**: Matplotlib, Seaborn
* **Supervised Machine Learning Framework**: Scikit-learn (`tree.DecisionTreeClassifier`, `ensemble.RandomForestClassifier`, `model_selection.train_test_split`, `metrics`)
