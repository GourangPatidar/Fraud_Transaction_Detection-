
---

# Fraud Detection Model

## Introduction

This project involves building and evaluating a fraud detection model to identify fraudulent transactions within a financial dataset. The dataset, which contains 6,362,620 rows and 10 columns, is sourced from Kaggle. The model aims to distinguish between legitimate and fraudulent transactions using a variety of machine learning techniques.

## Table of Contents

- [Data Collection](#data-collection)
- [Data Cleaning](#data-cleaning)
- [Data Analysis & Visualization](#data-analysis--visualization)
- [Data Pre-processing](#data-pre-processing)
- [Model Selection](#model-selection)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Variable Selection](#variable-selection)
- [Performance Demonstration](#performance-demonstration)
- [Key Fraud Indicators](#key-fraud-indicators)
- [Infrastructure Prevention Measures](#infrastructure-prevention-measures)
- [Effectiveness Assessment](#effectiveness-assessment)
- [Implementation](#implementation)
- [Additional Resources](#additional-resources)

## Data Collection

The dataset for this project is obtained from Kaggle and is in CSV format. It includes various features related to financial transactions, with the goal of identifying fraudulent activities.

- **Dataset Link**: [Fraudulent Transactions Data](https://www.kaggle.com/datasets/chitwanmanchanda/fraudulent-transactions-data)

## Data Cleaning

Data cleaning involves preprocessing the dataset to handle missing values, outliers, and inconsistencies. Key steps include:

1. **Handling Missing Values**: Filling or removing missing data.
2. **Outlier Detection**: Identifying and addressing outliers to prevent skewed results.
3. **Data Consistency**: Ensuring uniformity in data formats and values.

## Data Analysis & Visualization

Data analysis is performed to extract meaningful insights and understand the patterns within the data. Key observations include:

1. **Distribution of Transaction Types**: Identifying the frequency of each transaction type.
2. **Fraudulent vs. Non-Fraudulent Transactions**: Analyzing the proportion of fraudulent transactions.
3. **Amount Analysis**: Comparing transaction amounts for different types.
4. **Balance Changes**: Evaluating how balances change before and after transactions.
5. **Geographical Patterns**: Identifying any geographical trends in fraudulent transactions.

## Data Pre-processing

Data pre-processing prepares the dataset for modeling by performing the following tasks:

1. **Data Splitting**: Dividing the dataset into training and testing sets.
2. **Encoding Categorical Variables**: Converting categorical features into numerical format.
3. **Scaling Numerical Features**: Normalizing features to ensure equal weight.
4. **Feature Engineering**: Creating new features that may improve model performance.

## Model Selection

Various machine learning algorithms are evaluated for fraud detection:

1. **Logistic Regression**: A statistical model for binary classification.
2. **Decision Trees**: A model that splits data based on feature values.
3. **Support Vector Machines**: A model that finds the optimal hyperplane for classification.
4. **Gradient Boosting Machines**: An ensemble method that builds models sequentially.

**Chosen Algorithm**: Decision Tree, due to its interpretability and high accuracy in this context.

## Model Training

The Decision Tree model is trained on the preprocessed data. Training involves:

1. **Fitting the Model**: Learning patterns and relationships in the data.
2. **Parameter Tuning**: Adjusting model parameters to minimize the loss function.
3. **Achieved Accuracy**: 0.9997

## Model Evaluation

The model is evaluated on a separate test dataset using various metrics:

1. **Accuracy**: Measures the proportion of correctly classified transactions.
2. **Precision**: The ratio of true positives to the sum of true and false positives.
3. **Recall**: The ratio of true positives to the sum of true positives and false negatives.
4. **F1-score**: The harmonic mean of precision and recall.
5. **ROC-AUC**: The area under the receiver operating characteristic curve.
6. **Confusion Matrix**: A table that outlines the performance of the classification model.

## Variable Selection

The dataset includes the following variables:

1. **step**
2. **type**
3. **amount**
4. **nameOrig**
5. **oldbalanceOrg**
6. **newbalanceOrig**
7. **nameDest**
8. **oldbalanceDest**
9. **newbalanceDest**
10. **isFraud**
11. **isFlaggedFraud**

Variables with minimal impact, such as `step`, `nameOrig`, and `isFlaggedFraud`, are dropped based on their low correlation with the target variable and their lack of contribution to the model's accuracy.

## Performance Demonstration

### Model Training and Evaluation

1. **Training and Testing Split**: The dataset is divided into training and testing sets using `train_test_split` from scikit-learn.
2. **Algorithm Choice**: Decision Tree is chosen for its performance and interpretability.
3. **Evaluation Metrics**: Accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix are used to assess model performance.

### Data Visualization

1. **Confusion Matrix**: Visualizes the true vs. predicted classifications.
2. **ROC Curve**: Shows the trade-off between true positive rate and false positive rate.
3. **Precision-Recall Curve**: Illustrates the trade-off between precision and recall.
4. **Calibration Curve**: Evaluates how well the predicted probabilities match the actual outcomes.
5. **Feature Importances**: Displays the relative importance of each feature in the model.

### Hyperparameter Tuning

Hyperparameter tuning is performed using `GridSearchCV` or `RandomizedSearchCV` to find the best model parameters and enhance performance.

## Key Fraud Indicators

Key factors predicting fraudulent transactions include:

1. **Transaction Amount**: Unusually large amounts may indicate fraud.
2. **Balance Changes**: Significant changes in account balances before and after transactions.
3. **Transaction Type**: Certain types of transactions may be more likely to be fraudulent.
4. **Account Activity**: Unusual patterns in account activity or transactions.

## Infrastructure Prevention Measures

To enhance security and prevent fraud:

1. **Use Certified Applications**: Ensure that system updates come from reputable sources.
2. **Secure Websites**: Browse only secure websites with HTTPS protocols.
3. **Secure Internet Connections**: Use VPNs to protect data transmission.
4. **Regular Updates**: Keep security software up-to-date.
5. **Caution with Communications**: Avoid responding to unsolicited communications.
6. **Prompt Reporting**: Contact financial institutions immediately if fraud is suspected.

## Effectiveness Assessment

To determine if prevention measures are effective:

1. **Monitor Statements**: Track the frequency and content of electronic statements.
2. **Review Account Activity**: Encourage customers to check for discrepancies.
3. **Maintain Payment Logs**: Keep records of all payments for verification.
4. **Request Source Security**: Verify the security of request sources and legitimacy of requesting organizations.
5. **Vendor Transaction History**: Analyze transaction histories of vendors.

## Implementation

To run the project on your local machine:

1. **Download the Dataset**: [Fraudulent Transactions Data](https://www.kaggle.com/datasets/chitwanmanchanda/fraudulent-transactions-data)
2. **Run the Kaggle Notebook**: [Kaggle Notebook](https://www.kaggle.com/code/sauhardsaini/fraud-detection-dtrf/notebook)

Follow the steps outlined in the notebook for data processing, model training, and evaluation.

---

