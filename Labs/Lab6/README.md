# ARTI308 – Lab 6: Linear Regression

This project implements a **Linear Regression** framework to analyze customer transactional behavior and digital channel engagement metrics using the **Ecommerce Customers dataset**.

## Objective
Build and validate a predictive machine learning regression model to estimate a user's **Yearly Amount Spent** based on session durations, mobile application usage, and website engagement patterns to provide actionable retail business insights.

---

## Dataset Description (`Ecommerce Customers.csv`)
The dataset consists of continuous numerical tracking metrics across consumer accounts:

* **`Avg. Session Length`**: Average duration (in minutes) of in-store style advice and support sessions.
* **`Time on App`**: Average time spent by the customer interacting with the brand's mobile application.
* **`Time on Website`**: Average time spent by the customer exploring the desktop website interface.
* **`Length of Membership`**: Total number of consecutive years the customer has maintained an active account with the company.
* **`Yearly Amount Spent` (Target Variable)**: The total annual capital spent by the customer (continuous numerical target vector to be predicted).

---

## Core Tasks & Methodology

### 1. Exploratory Data Analysis (EDA) & Visualization
* **Jointplots:** Leveraging Seaborn jointplots to structurally isolate dual-variable visual pairings, scatter density trends, and regression alignments between separate engagement platforms and total expenditure.
* **Pairplots:** Constructing data-wide pairplots to evaluate multi-variable distribution densities and uncover linear correlations across the entire workspace grid.

### 2. Correlation Analysis
* Generating a Pearson product-moment correlation matrix and heatmap to isolate and identify the absolute strongest linear predictors of spending behavior (e.g., evaluating the predictive weight of account longevity versus standalone platform engagement).

### 3. Model Training & Data Partitioning
* Isolating feature parameters ($X$) from target outcome criteria ($y$).
* Splitting data structures into independent **Training** and **Testing** partitions using `scikit-learn` to prevent analytical data leakage and rigorously measure model generalization on unseen vectors.
* Fitting an Ordinary Least Squares (OLS) Linear Regression model to establish baseline mathematical intercepts and predictive feature coefficients.

### 4. Model Prediction & Evaluation
Quantifying the predictive exactness and residual error spread of the model on the test data split using standard regression metrics:
* **Mean Absolute Error (MAE):** Quantifies the absolute average magnitude of prediction errors.
* **Mean Squared Error (MSE):** Accents and penalizes larger, catastrophic residual errors heavily by squaring the distance vectors.
* **Root Mean Squared Error (RMSE):** Translates total error variance back into original currency units ($) for straightforward business risk interpretation.

---

## Technical Stack
* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn (LinearRegression, train_test_split, metrics)
