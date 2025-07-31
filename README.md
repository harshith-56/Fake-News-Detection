# Fake-News-Detection

A machine learning project to classify news articles as FAKE or REAL using Natural Language Processing (NLP) techniques and the Passive Aggressive Classifier.

# Overview
This project aims to identify fake news articles using a supervised learning model trained on labeled news data. The goal is to develop a system that can detect misleading content and help promote reliable information.

# Features
Classifies news articles into FAKE or REAL.

Preprocesses news text for cleaner model input.

Extracts custom features like article length and title length.

Converts text to numerical form using TF-IDF Vectorization.

Trains multiple models including:

Logistic Regression

Support Vector Machine (SVM)

Random Forest

Naive Bayes

Passive Aggressive Classifier (Final Model)

Applies GridSearchCV for hyperparameter tuning.

Achieves 93.9% accuracy on validation data.

# Dataset
Contains ~6,000 labeled news articles

Columns: title, text, label (FAKE or REAL)

# Technologies Used
Python

Jupyter Notebook

NumPy, Pandas

scikit-learn

matplotlib, seaborn (for EDA)

# Evaluation Metrics
Accuracy

Precision

Recall

F1-Score

# Challenges Faced
Data Imbalance: Handled via stratified splitting

Feature Selection: Multiple iterations to find effective features

Overfitting: Managed using regularization and model simplification

# Key Insights
Fake news articles tend to be longer than real ones.

Text-based features like length and word importance help improve classification.

A high-performing ML model can be a powerful tool in combating misinformation.


