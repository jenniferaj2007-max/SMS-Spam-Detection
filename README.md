SMS Spam Detection Using Machine Learning

Project Overview

This project detects whether an SMS message is Spam or Ham (Normal) using Machine Learning classification algorithms.

Dataset

- Dataset: SMS Spam Collection
- Source: UCI Machine Learning Repository
- Original dataset size: 5,574 messages (5,572 records loaded in this project)
- Features: Label and Message
- Target classes: Ham and Spam

Machine Learning Algorithms

1. Logistic Regression
2. Support Vector Machine (SVM)
3. Decision Tree

Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Matplotlib
- Joblib
- SciPy

Project Workflow

1. Data Loading
2. Data Understanding
3. Data Cleaning
4. Text Preprocessing using TF-IDF
5. Training Machine Learning Models
6. Model Evaluation
7. Model Comparison
8. Performance Visualization
9. New SMS Spam Prediction

Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score

Project Structure

SMS_Spam_Detection/
├── data/
├── models/
├── results/
├── data_loading.py
├── data_understanding.py
├── data_cleaning.py
├── data_preprocessing.py
├── logistic_regression_model.py
├── svm_model.py
├── decision_tree_model.py
├── model_comparison.py
├── model_visualization.py
├── spam_prediction.py
└── README.md

Conclusion

This project demonstrates how machine learning algorithms can classify SMS messages into Spam and Ham categories using text preprocessing and TF-IDF feature extraction. The three algorithms are evaluated and compared to identify their performance.