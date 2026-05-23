# ARTI308 – Lab 11: Customer Behavior Segmentation Using K-Means Clustering

This repository contains a complete unsupervised machine learning pipeline implementing **K-Means Clustering** and **Principal Component Analysis (PCA)** to uncover hidden customer cohorts within credit card usage records.

## Project Objective
The primary focus of this project is to profile and partition credit card users based on historical financial behaviors (balances, cash advances, and installment purchasing frequencies). By identifying distinct customer archetypes without relying on pre-labeled target metrics, the framework generates data-driven segmentations to optimize personalized marketing and credit risk allocation strategies.

---

## Technical Pipeline & Architecture

### 1. Missing Data Handling & Profiling
* Auditing structural shapes across the 18 transactional dimensions of `CC_GENERAL.csv`.
* Isolating target columns with missing observations (`MINIMUM_PAYMENTS` and `CREDIT_LIMIT`) and applying robust median imputation to protect underlying feature distributions from skewness.

### 2. Feature Scale Normalization
Because variance magnitudes differ widely between features (e.g., account `BALANCE` values scaling into thousands vs. `PURCHASES_FREQUENCY` bounding between 0 and 1), distance-based clusterers are highly sensitive to raw scale units.
* Applying `StandardScaler` to transform features to a uniform distribution centering around a mean of 0 and a standard deviation of 1.

### 3. Hyperparameter Cluster Optimization ($K$-Selection)
Pinpointing the mathematical "sweet spot" for customer clusters using two complementary verification techniques:
* **The Elbow Method:** Testing clusters across a range of $K=1$ to $K=10$ and plotting the Within-Cluster Sum of Squares (WCSS) to locate the inflection point where additional centroids yield diminishing returns.
* **Silhouette Analysis:** Computing silhouette coefficients to evaluate cluster tightness, ensuring high intra-cluster similarity and distinct inter-cluster separation.

### 4. Dimensionality Reduction & Visual Projection
* Deploying **Principal Component Analysis (PCA)** to capture maximum global variance and project the multi-dimensional scaled data down to two prominent principal components (PC1 and PC2).
* Generating a 2D scatter plot colored by K-Means cluster assignments to visually evaluate the geometric boundary separation of the identified consumer segments.

### 5. Final Segmentation Interpretations
The optimized pipeline successfully maps consumers into clear behavioral clusters:
* **High-Value Spenders:** Users showing elevated purchase volumes, high credit limits, and substantial payment activities.
* **Cash-Advance Reliant Users:** Consumers characterized by minimal purchase records combined with high cash-advance draws and lingering account balances.
* **Conservative / Balanced Users:** Average cardholders exhibiting moderate balances and balanced payment-to-purchase ratios.

---

## Core Technical Stack
* **Language:** Python
* **Data Processing & Cleansing:** Pandas, NumPy
* **Visual Analytics & Plots:** Matplotlib, Seaborn
* **Unsupervised Learning Framework:** Scikit-learn (`cluster.KMeans`, `preprocessing.StandardScaler`, `decomposition.PCA`, `metrics.silhouette_score`)
