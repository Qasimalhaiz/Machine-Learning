# IAU Machine Learning - Lab 3: Exploratory Data Analysis (EDA)

## Project Overview
This project focuses on performing a comprehensive **Exploratory Data Analysis (EDA)** on a Buy Now Pay Later (BNPL) dataset. This analysis is a critical step in my graduation project, which aims to build a predictive model for credit risk assessment in the BNPL industry.

The goal of this lab is to understand the underlying patterns, detect anomalies, and identify relationships between customer demographics, financial status, and their repayment behavior.

## Dataset Description
The dataset used in this analysis, `bnpl_dataset.csv`, contains 50,000 transaction records with the following features:

* **Customer_Age**: The age of the customer.
* **Annual_Income**: Total yearly income.
* **Credit_Score**: Financial creditworthiness (typically 300-850).
* **Purchase_Amount**: The total cost of the transaction.
* **Purchase_Category**: Category of the item (Electronics, Fashion, Travel, etc.).
* **BNPL_Provider**: The service used (Affirm, Klarna, Afterpay, etc.).
* **Repayment_Status (Target)**: Whether the customer "Paid On Time" or "Defaulted".

## Key EDA Steps Performed

### 1. Data Profiling
* Verified data integrity using `.info()` and `.isnull()`.
* Generated statistical summaries with `.describe()` to identify the range and distribution of financial metrics.

### 2. Univariate Analysis
* Analyzed the **Age Distribution** of customers using histograms to identify target demographics.
* Examined the balance of the target variable (`Repayment_Status`) to prepare for future classification tasks.

### 3. Bivariate Analysis
* **Income vs. Repayment**: Utilized Boxplots to compare the income levels of customers who pay on time versus those who default.
* **Category vs. Repayment**: Analyzed default rates across different purchase sectors (e.g., higher risk in Travel vs. Groceries).
* **Credit Score vs. Purchase Amount**: Used Scatter Plots to identify clusters of risk based on creditworthiness and transaction size.

### 4. Correlation Analysis
* Developed a **Correlation Heatmap** to identify strong linear relationships between numerical features, aiding in future feature selection.

## Technologies Used
* **Python 3.x**
* **Pandas**: Data manipulation and cleaning.
* **Matplotlib & Seaborn**: Statistical data visualization.
* **Jupyter Notebook**: For interactive development and documentation.

## How to Run
1. Clone this repository.
2. Ensure you have the required libraries installed: `pip install pandas seaborn matplotlib`.
3. Open `CCSIT_ARTI308_Lab3.ipynb` in Jupyter Notebook or VS Code.
4. Ensure `bnpl_dataset.csv` is in the same directory.
5. Run all cells to generate the visualizations.

---
**Author:** Qasem Alhayaz  
**Institution:** Imam Abdulrahman Bin Faisal University (IAU)  
**Course:** ARTI308 - Machine Learning
