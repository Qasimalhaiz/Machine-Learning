# ARTI308 – Lab 4: Data Quality Assessment & Preprocessing

This repository contains a structured, end-to-end data engineering and preprocessing pipeline designed to optimize features within the **Buy Now Pay Later (BNPL) Transaction Dataset**.

## Project Objective
The primary goal is to establish a rigorous data cleaning framework that mitigates the impact of financial distribution anomalies, handles missing numerical parameters, and transforms highly skewed customer attributes. This ensures maximum data quality and structure prior to deploying predictive supervised or unsupervised machine learning models.

---

## Technical Pipeline & Methodology

### 1. Exploratory Data Inspection & Schema Validation
* Auditing baseline data structures, feature data types, and row dimension counts.
* Validating column constraints across core quantitative transactional attributes and customer demographic profiles (e.g., verifying `Purchase_Amount` and customer age boundaries).

### 2. Synthetic Sparsity & Missing Value Imputation
* Intentionally introducing missingness (NaN values) into key financial tracking arrays to simulate real-world transmission gaps or unrecorded interactions.
* Deploying and evaluating standard univariate statistical imputation methods, specifically comparing the operational variance of **Mean Imputation** against **Median Imputation** on continuous numerical distributions.

### 3. Outlier Mitigation & Robust Feature Capping
* Using the non-parametric **Interquartile Range (IQR)** filtering method to systematically identify purchasing anomalies and extreme transactional behavior.
* Applying defensive percentile-based capping boundaries (Winsorization limits) to constrain severe outliers, neutralizing their leveraging effects without truncating essential row indexes from the dataset.

### 4. Feature Rescaling & Covariate Transformation
Standardizing numerical distributions to eliminate scale-based bias during downstream optimization tasks using two distinct techniques:
* **Min-Max Scaling (Normalization):** Bounding feature limits strictly within a continuous range of $[0, 1]$ to preserve relative distance ratios.
* **Z-Score Standardization:** Rescaling variables to match a standard normal distribution with a mean ($\mu$) of $0$ and a standard deviation ($\sigma$) of $1$, ensuring equal weight for high-variance parameters like `Annual_Income`.

### 5. Multi-Variable Covariance & Correlation Analysis
* Constructing a Pearson product-moment correlation matrix and accompanying heatmaps to audit underlying relationships.
* Assessing structural linear redundancies and multicollinearity between independent financial dimensions to determine data reduction necessity.

### 6. Dimensionality Reduction via Principal Component Analysis (PCA)
* Projecting high-dimensional coordinate workspaces into an orthogonal, lower-dimensional geometric subspace.
* Maximizing the captured variance across the leading principal components (PC1 and PC2) to simplify the feature space while minimizing overall data information loss.

---

## Core Technical Stack
* **Language:** Python
* **Data Engineering Frameworks:** Pandas, NumPy
* **Visualization & Plotting Engines:** Matplotlib, Seaborn
* **Mathematical Optimization Engine:** Scikit-learn (`preprocessing.MinMaxScaler`, `preprocessing.StandardScaler`, `decomposition.PCA`)
