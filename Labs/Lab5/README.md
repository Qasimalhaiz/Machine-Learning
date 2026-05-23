# ARTI308 – Lab 5: Feature Engineering & Dimensionality Reduction (Classification)

This repository features a comprehensive data preparation, feature engineering, and supervised classification pipeline utilizing the enhanced logistical delivery dataset (`talabat_enhanced_orders.csv`).

## Project Objective
The primary goal is to design and evaluate advanced feature engineering strategies, thresholding heuristics, and supervised feature selection methods. The pipeline optimizes a machine learning framework to accurately classify final fulfillment statuses (`Order_Status`), maximizing predictive metrics while minimizing computational complexity.

---

## Technical Methodology & Executed Tasks

### 1. Advanced Feature Derivation (Logistical Metrics)
* **Task 1: Logistical Velocity Modeling** Engineered the compound interaction variable `distance_per_item` to establish a quantitative baseline for delivery routing efficiency. This feature maps spatial constraints directly against order complexity vectors, uncovering hidden structural boundaries affecting order fulfillment rates.

### 2. Heuristic Temporal Engineering & Boundary Optimization
* **Task 2: High-Demand Temporal Thresholding** Designed, implemented, and validated an alternative peak-hour operational rule to capture broader variations in restaurant demand cycles. This customized temporal window accurately isolates high-stress dispatch periods to measure their explicit impact on delivery latency and status changes.

### 3. Dimensionality Management & Categorical Consolidation
* **Task 4: Cardinality Constraint Evaluation (`Top-K` Reduction)** Conducted a rigorous analysis of high-cardinality categorical variables. By compressing sparse categories into a localized `Top-K` subset and grouping low-frequency residuals into a unified background category, the pipeline balances data representation without inflating matrix dimensionality.

### 4. Automated Feature Selection & Sparse Regularization
* **Task 5: Model-Driven Feature Pruning (`SelectFromModel`)** Deployed an embedded feature selection wrapper leveraging the `SelectFromModel` architecture. By auditing underlying feature importance coefficients, the pipeline successfully eliminated redundant covariates—**reducing the total feature space by 50%** while preserving an outstanding **85.2% classification accuracy**.

---

## Core Technical Stack
* **Language:** Python
* **Data Engineering Frameworks:** Pandas, NumPy
* **Visualization & Plotting Engines:** Matplotlib, Seaborn
* **Predictive Machine Learning Engine:** Scikit-learn (`feature_selection.SelectFromModel`, `ensemble.RandomForestClassifier`, `model_selection.train_test_split`, `metrics`)
