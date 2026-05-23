# ARTI308 – Lab 11:Credit Card Customer Segmentation with K-Means

---

## Overview
This lab applies **K-Means Clustering** to segment credit card customers based on their financial behavior and spending patterns.  
The project focuses on discovering hidden customer groups using unsupervised learning techniques.

---

## Dataset
**CC_GENERAL.csv**

The dataset contains customer information such as:
- Balance
- Purchases
- Credit limit
- Cash advance usage
- Payments

---

## Objectives
- Perform customer segmentation using K-Means
- Handle missing values
- Scale numerical features
- Determine the optimal number of clusters
- Evaluate clustering performance
- Visualize clusters using PCA

---

## Steps Performed
1. Data exploration and preprocessing  
2. Handling missing values  
3. Feature scaling using `StandardScaler`  
4. Applying the Elbow Method  
5. Evaluating clusters using Silhouette Score  
6. Training the K-Means model  
7. Visualizing clusters with PCA  

---

## Libraries Used
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Results
The model successfully grouped customers into different segments based on:
- Spending behavior
- Credit usage
- Cash advance activity
- Balance patterns

These segments can help businesses improve customer targeting and marketing strategies.
