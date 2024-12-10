# Fraud Detection Machine Learning Project

## Project Overview

This project aims to develop a machine learning system for detecting fraudulent transactions in a Point of Sale (PoS) application.

## Approach

### 0. Getting dataset
I swear this was the most fucked up thing of this whole process. I tried so fucking hard not to use this dataset. [This shit of anonymised](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

### 1. Visualising data and Data preprocessing

- Get a histogram of every feature in the dataset
- Fill in missing data with mean of the columns
- Create a graph comparing fraud and legitimate transactions
- BoxPlot of amount
- Correlation heatmap

### 2. Data Cleaning

- Removed outliers using `IQR` method while preserving fraud transactions because they are already scarce.
- Average out `Time` values and use `RobustScaler` for `Amount` so we can weed out the outliers.
- Get a statiscal description of both `new_df` and `df`

- Using `StandardScaler` to scale the data.
- Split data into `70:30` for training and testing purposes.
- Utilized SMOTE (Synthetic Minority Over-sampling Technique) to address the significant class imbalance
- Experimented with different sampling strategies (0.1, 0.2, 0.3, 0.4, 0.5) to optimize model performance
- Carefully balanced the dataset to prevent bias towards the majority class (legitimate transactions). This also forced us to choose `F1-Score` to rank our models.

### 3. Model Development

Implemented and compared multiple classification algorithms:
- Logistic Regression
- Random Forest
- LightGBM
- XGBoost
- Neural Network

**PS**: I dropped SVM because of its extremely long runtime.

### 4. Model Evaluation

Evaluation Metrics:
- AUC-ROC Score
- **F1 Score**
- Accuracy
- Precision
- Recall
- Confusion Matrix
- Classification Report

The F1 score is often preferred over accuracy when data is unbalanced, such as when one class has many more examples than the other. Thus, it can be useful in fraud detection, where traditional accuracy metrics can be misleading. 

### 5. Model Interpretability with SHAP
To better understand our model's decisions, we applied SHAP (SHapley Additive exPlanations). SHAP helps interpret the output of complex models by explaining how each feature contributes to individual predictions. By using SHAP, we were able to:

- Visualize feature importance
- Identify key features driving fraudulent transaction predictions
- Ensure model transparency and trustworthiness
- Example SHAP values visualizations and explanations were created for all the models to show which features were most influential in detecting fraud.

### 6. Real-time Fraud Detection API with FastAPI
We also created a real-time fraud detection API using FastAPI. This API serves as a deployment solution for the fraud detection model, allowing users to:

- Input transaction data.
- Get predictions on whether a transaction is fraudulent or legitimate.
- Get health status of the API.


## Challenges

1. **Severe Class Imbalance**: The dataset contained significantly more legitimate transactions compared to fraudulent ones, potentially introducing bias in model training.

2. **Model Complexity**: Some models like Random Forests and Support Vector Machines:
   - Demonstrated ineffective evaluation
   - Exhibited excessively long runtime
   - Could not be comprehensively evaluated due to computational constraints

## Results

Model Performance Comparison (F1 Scores):
- Random Forest: 0.9714
- LightGBM: 0.9534
- XGBoost: 0.9432
- Neural Network: 0.9280
- Logistic Regression: 0.38 (I do not know the reason behind this)

**Key Achievements**:
- Successfully identified 99.97% of fraudulent transactions
- Maintained an extremely low false positive rate
- `Random Forest` emerged as the most accurate model for this dataset
- Visualised `SHAP` values for all the models.
- Implemented a cURL API to get the prediction values for given data.

## Conclusion

The project demonstrated the effectiveness of machine learning techniques in fraud detection, particularly `Random Forest`, as well as the powerful use of `F1 Score`. By addressing class imbalance and carefully selecting and tuning models, we developed a robust system capable of identifying fraudulent transactions with high accuracy.