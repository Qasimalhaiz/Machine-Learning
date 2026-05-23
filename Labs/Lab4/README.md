ARTI308 – Lab 4: Data Quality Assessment & Preprocessing

This project performs data cleaning and preprocessing for the Buy Now Pay Later (BNPL) Transaction Dataset.

Objective : 

- Prepare the transaction and customer credit dataset for predictive modeling by improving data quality, mitigating the impact of financial outliers, and transforming features appropriately.

Key Steps :

- Data Inspection & Validation: Verifying column schemas such as transaction values and customer age.

- Missing Value Handling: Simulating missing data in transaction metrics and applying mean/median imputation techniques.

- Outlier Detection & Management: Identifying extreme purchasing anomalies using the Interquartile Range (IQR) method and applying percentile capping limits.

- Normalization & Standardization: Rescaling numerical features like Purchase_Amount and Annual_Income using Min-Max and Z-Score scaling.

- Correlation Analysis: Checking the linear relationships between financial attributes to evaluate data redundancy.

- Principal Component Analysis (PCA): Applying dimensionality reduction techniques to capture variance and consolidate numeric features.