ARTI308 – Lab 10: Support Vector Machines (SVM)
Overview

This lab focused on applying Support Vector Machines (SVM) to classify the well-known Iris flower dataset. The project involved data exploration, model training, performance evaluation, and hyperparameter optimization using Grid Search.

Dataset

The Iris dataset contains 150 flower samples divided into three species:

Iris Setosa
Iris Virginica
Iris Versicolor

Each sample includes four measured features:

Sepal length
Sepal width
Petal length
Petal width

All measurements are recorded in centimeters.

Project Steps
1. Exploratory Data Analysis (EDA)

Seaborn and Matplotlib were used to explore and visualize the dataset.

Pairplots were created to observe relationships between features and species classifications.
KDE plots were used to analyze the distribution of sepal length and sepal width.
The visualizations showed that the Setosa species was clearly distinguishable from the other two species.
2. Model Training

The dataset was divided into training and testing sets using a 70/30 split ratio.

The SVC classifier from Scikit-learn was implemented.
The model was trained using the training data.
Performance was evaluated using a confusion matrix and classification report.
3. Hyperparameter Optimization

To improve model accuracy, GridSearchCV was applied to determine the best values for the parameters C and gamma.

The following parameter values were tested:

C: [0.1, 1, 10, 100, 1000]
gamma: [1, 0.1, 0.01, 0.001, 0.0001]
4. Final Results

After optimization, the SVM model achieved an accuracy of approximately 98%, successfully classifying nearly all test samples correctly.

Technologies Used

Programming Language:

Python

Libraries:

pandas and numpy for data handling and analysis
seaborn and matplotlib for data visualization
scikit-learn for machine learning implementation and evaluation