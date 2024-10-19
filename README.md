# Credit Risk Analysis
## Project Overview

This project focuses on predicting the credit risk of customers using machine learning models. Credit risk refers to the likelihood that a borrower will default on their financial obligations. Analyzing and predicting this risk is crucial for financial institutions, as it helps them make informed lending decisions.

Key aspects of this problem include:
- Identifying potential high-risk customers
- Building predictive models to accurately assess the credit risk
- Evaluating model performance using metrics such as accuracy, precision, recall, and F1-score

## Methods Used

In this project, we implemented three machine learning models to predict credit risk:

1. **Support Vector Classifier (SVC)**: A powerful supervised learning algorithm used for classification tasks. It works by finding the hyperplane that best divides a dataset into classes.
  
2. **Decision Tree Classifier**: A simple yet effective classification algorithm that splits the data into subsets based on the most important features. It builds a tree-like structure of decisions to classify data points.

3. **Random Forest Classifier**: An ensemble learning method that builds multiple decision trees and combines their outputs to improve accuracy and reduce the risk of overfitting. It aggregates the results from individual trees to determine the final class.

## Data Collection

The dataset for this project was sourced from Kaggle. It contains information on customer demographics, credit history, and financial behaviors, which are used to predict whether a customer is likely to default.

## Model Training

### 1. Decision Tree Classifier
The Decision Tree Classifier is a simple yet powerful tool for classification tasks. It works by recursively splitting the dataset based on feature values, forming a tree structure of decisions. Each node in the tree represents a decision rule, and each leaf node represents a class label (credit risk or no risk). This method is interpretable and useful for understanding how individual features contribute to the classification decision.

### 2. Random Forest Classifier
Random Forest is an ensemble method that constructs multiple decision trees during training. Each tree is built from a random subset of the data, and the final prediction is determined by aggregating the results of all trees. This technique helps to reduce overfitting, improves accuracy, and provides a robust model for predicting credit risk. Random Forest can handle a large number of features and is generally more stable than individual decision trees.

### 3. Support Vector Classifier (SVC)
Support Vector Classifier is a machine learning model that tries to separate the classes by finding the hyperplane that maximizes the margin between data points. It’s effective in high-dimensional spaces and can work well with both linear and non-linear data. In this case, we use SVC to differentiate between risky and non-risky customers based on their financial history and demographics.

## Results

The performance of each model was evaluated using key metrics such as accuracy, precision, recall, and F1-score. Below is a summary of the results:

- **Decision Tree Classifier**: Achieved decent accuracy but was prone to overfitting due to the nature of the algorithm.
- **Random Forest Classifier**: Performed better than Decision Tree, with improved accuracy and reduced overfitting. The ensemble approach helped in creating a more generalizable model.
- **SVC**: Provided moderate results, but after applying random oversampling techniques to balance the data, it showed significant improvements in performance, particularly in recall and F1-score.

## Conclusion

In conclusion, the Random Forest Classifier proved to be the most effective model for this credit risk analysis, offering the best trade-off between accuracy and model robustness. However, the use of random oversampling techniques yielded the best results for the Support Vector Classifier, significantly improving its performance on the imbalanced dataset. For the Decision Tree and Random Forest models, no significant improvements were observed with oversampling, but the Random Forest still outperformed the other models in terms of accuracy and stability.

---

This README file provides a concise and structured overview of your Credit Risk Analysis project, from problem description to model performance and conclusions.
