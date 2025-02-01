### PREDICTING-CANCER-USING-CLASSIFICATION-MACHINE-LEARNING-MODELS
A comprehensive project I have done involving data preparation, balancing, feature selection and classification models building.

## Table of Contents
- [Introduction](#introduction)
- [Data Preprocessing](#data-preprocessing)
- [Data Balancing](#data-balancing)
- [Class Distribution After Balancing](#class-distribution-after-balancing)
- [Feature Selection](#feature-selection)
- [Classification Models Building](#classification-models-building)
- [Performance Metrics](#performance-metrics)
- [Hyperparameter Tuning Efforts](#hyperparameter-tuning-efforts)
- [Ensemble Method](#ensemble-method)
- [Conclusion and Future Improvements](#conclusion-and-future-improvements)

## Introduction
**Goal**: Predict cancer diagnosis based on 2019 Behavioral Risk Factor Surveillance System (BRFSS) survey data from CDC.

**Dataset**: [2019 Behavioral Risk Factor Surveillance System (BRFSS)](https://www.cdc.gov/brfss/annual_data/annual_2019.html)

This report documents the process and results of a data mining project aimed at predicting whether a person has ever been told they had cancer based on survey data. The dataset contains various health-related attributes, and the target variable indicates whether a person was ever told they had cancer (Y for yes, N for no). To speed up the process, parallel processing was used throughout the code, leveraging multiple CPU cores for optimal running time.

## Key Results Summary
- **Best Model**: Naive Bayes on the ros_chi_sq dataset (Random Over Sampling on Chi-Sqaure feature selected data set)
- **Class0 TPR**: 67%
- **Class1 TPR**: 64%
- **Weighted AUC**: 0.72

## Data Preprocessing
Data preprocessing involved several steps:
- Loading Data: The dataset was loaded into the R environment using appropriate libraries. This step involves reading the dataset from a CSV file and storing it in a data frame for further processing.
- Removing Unnecessary Columns: Columns deemed irrelevant to the prediction task were removed.
- Handling Special Codes and Missing Values: Special codes representing missing or invalid values were replaced with NA.
- Imputation: Missing values were imputed using appropriate methods (KNN for numeric, mode for categorical, and median for ordinal).
- Transformation and Standardization: Numeric attributes like weight and height were transformed to standard units, and numeric columns were scaled using min-max scaling.
- Outlier Treatment: Outliers were capped using the interquartile range (IQR).
- Target Variable Encoding: The target variable Class was encoded as 1 for Y (cancer diagnosis) and 0 for N (no cancer diagnosis).
- Data Splitting: The dataset was split into training and testing sets using a 80-20 split ratio. This allows us to train the model on one portion of the data and evaluate it on the unseen data to assess its generalization performance.
- Data validation: The dataset was examined right after splitting to ensure that there are no missing values, duplicated values as well as ensuring the structure of the data.

## Data Balancing 
Data balancing was performed to address class imbalance in the target variable:
- Random Over-Sampling (ROS): The minority class was oversampled to match the size of the majority class.
- Random Under-Sampling (RUS): The majority class was under-sampled to match the size of the minority class.
Although, data balancing can be somewhat miss leading to the audiances, because it could potentially make the model bias towards the synthtic data. However, an unbalanced data can affect the ML models to bias towards the majority class which leads to poor performance when we try to apply our trained model to other real world data. To mitigate this situation, I will evaluate the model's performance thoroughly using unseen test data set, and apply feature engineering to ensure synthetic samples are as close as possible to real patterns. 

## Class Distribution After Balancing:
- ROS: Class 0: 3315; Class 1: 3315
- RUS: Class 0: 685; Class 1: 685

## Feature Selection
I utilized several feature selection methods to enhance the performance of our classification models. These methods are crucial for identifying the most significant features in our datasets, thereby improving model accuracy and reducing computational complexity. The methods employed are:
- Recursive Feature Elimination (RFE):
RFE works by recursively removing the least significant features and building a model on the remaining features. It helps in ranking the features by importance and selecting the optimal subset.
- Random Forest Importance:
This method uses the Random Forest algorithm to evaluate the importance of each feature. Features are ranked based on their contribution to the model's predictive power.
- Information Gain:
Information Gain measures the reduction in entropy or uncertainty when a feature is used to split the data. It is a common technique used in decision tree algorithms.
- Boruta:
Boruta is a wrapper algorithm around the Random Forest classifier. It iteratively removes features that are deemed less important than random probes, ensuring the selection of all relevant features.
- Chi-Squared Test:
The Chi-Squared Test assesses the independence of features from the target variable. It is used for categorical data to determine if there is a significant association between the feature and the target.
Dimensions of all feature selected data: (ROS as random over sampling, RUS as random under sampling)

## Class Distribution after feature selection methods (Class0: ; Class1:):
- ros_boruta_features: 6630 : 140
- ros_rfe_features: 6630: 16
- ros_rf_features: 6630 : 26
- ros_ig_features: 6630 : 31
- ros_chi_sq_features: 6630 : 51
- rus_rfe_features: 1370 : 158
- rus_rf_features: 1370 : 26
- rus_ig_features: 1370 : 31
- rus_boruta_features: 1370 : 25
- rus_chi_sq_features: 1370 : 51

## Classification Models Building

I employed a variety of classification models to analyze the performance across different datasets generated using various feature selection methods. A custom performance table function was created to calculate associated values. The models used in our analysis are:
Logistic Regression (GLM):
- GLM is a flexible generalization of ordinary linear regression that allows for response variables to have error distribution models other than a normal distribution. It is widely used for binary classification problems.
Recursive Partitioning (rpart):
- Recursive Partitioning is a decision tree-based method that recursively splits the data into partitions to build a predictive model. It is known for its simplicity and interpretability.
Support Vector Machine (SVM):
- SVM is a powerful classifier that works by finding a hyperplane that best divides a dataset into classes. It is effective in high-dimensional spaces and particularly in cases where the number of dimensions exceeds the number of samples.
Random Forest (rf):
- Random Forest is an ensemble learning method that constructs multiple decision trees and merges them to get a more accurate and stable prediction. It is effective in handling large datasets and reducing overfitting.
Naive Bayes (nb):
- Naive Bayes is a probabilistic classifier based on Bayes' Theorem with the assumption of independence between features. It is particularly effective for large datasets and text classification tasks.
k-Nearest Neighbors (knn):
- k-NN is a non-parametric method used for classification. It predicts the class of a sample based on the majority class of its k-nearest neighbors in the feature space.
Performance Metrics
- The performance of each model was evaluated using several metrics including accuracy, precision, recall, F1-score. Confusion matrices also created in the process. Both performance tables and confusion matrices are stored into CSV files for future calculation. The Area Under the Curve (AUC) of the Receiver Operating Characteristic (ROC) curve, MCC and Kappa statistics with their weighted average for different classes will be calculated manually using the stored confusion matrices. I have included the top 5 performing models’ tables in the following for each model and feature selection method combination.

## Performance Metrics

1.Naive Bayes model with ros_chi_sq feature selected dataset

|             | TPR      | FPR      | Precision | Recall   | F_measure |  ROC     |  MCC     |  Kappa   |
|-------------|----------|----------|-----------|----------|-----------|----------|----------|----------|
|      Class0 | 0.613475 | 0.287582 | 0.921847  | 0.613475 | 0.736693  | 0.717139 | 0.236651 | 0.185422 |
|      Class1 | 0.712418 | 0.386525 | 0.25      | 0.712418 | 0.370119  | 0.717139 | 0.236651 | 0.185422 |
| Wt. Average | 0.628629 | 0.302735 | 0.818952  | 0.628629 | 0.680551  | 0.717139 | 0.236651 | 0.185422 |

## Among the models evaluated so far, the Naive Bayes model on the ros_chi_sq dataset emerged as the top performer. With Class1 TPR = 71.2% and Class0 TPR = 61.3%. Additionally the weighted ROC of 0.72 is also a good indicator of the model's ability to distinguish between Class0 and Class1. 

## Confusion matrix for Naive Bayes model on the ros_chi_sq dataset:

|               | Predicted Class0| Predicted Class1|
|---------------|-----------------|-----------------|
| Actual Class0 | 519             | 44              |
| Actual Class1 | 327             | 109             |

## ROC for Naive Bayes model on the ros_chi_sq dataset:
![ROC Curve for nb on ros_chi_sq](./ROC_NB_Chisq.png)

## Hyperparameter Tuning Efforts
To further improve the performance of the model, hyperparameter tuning was performed. The “tuneLength” parameter was set to 10, which means that the algorithm tries 10 different combinations of hyperparameters to find the optimal settings. The goal was to find the optimal parameters that could potentially enhance the model's performance.
## Hyperparameter Grids:
For each model, a grid of hyperparameters was defined. These grids specify the range of values for each hyperparameter to be tested during tuning.
- Logistic Regression (GLM): No hyperparameter tuning required.
- Decision Tree (RPART): Complexity parameter (cp) was varied from 0 to 0.1.
- Random Forest (RF): Number of variables randomly sampled as candidates at each split (mtry) was varied.
- Naive Bayes (NB): Parameters fL, usekernel, and adjust were varied.
- k-Nearest Neighbors (KNN): Number of neighbors (kmax), distance metric, and kernel type were varied.
- Support Vector Machine (SVM): Cost (C) and sigma were varied.

Naive Bayes on ros_chi_sq Dataset - Hyperparameter Tuning Performance Metrics after Tuning:

|               | TPR      | FPR      | Precision | Recall   | F_measure | ROC     | MCC       | Kappa    |
|---------------|----------|----------|-----------|----------|-----------|---------|-----------|----------|
| Class0        | 0.619385 | 0.287582 | 0.922535  | 0.619385 | 0.74116   | 0.717038 | 0.241268 | 0.190233 |
| Class1        | 0.712418 | 0.380615 | 0.2529    | 0.712418 | 0.373288  | 0.717038 | 0.241268 | 0.190233 |
| Wt. Average   | 0.633634 | 0.30183  | 0.819979  | 0.633634 | 0.684819  | 0.717038 | 0.241268 | 0.190233 |

Naive Bayes on ros_chi_sq Dataset - Confusion Matrix after Hyperparameter Tuning:

|               | Predicted Class0 | Predicted Class1 |
|---------------|-------------------|------------------|
| Actual Class0 | 524               | 44              |
| Actual Class1 | 322               | 109             |

## Despite the hyperparameter tuning efforts, the Naive Bayes model on the ros_chi_sq dataset along with other 6 top performing models did not show any significant improvement in terms of TPR. One of the possible reasons is that Naïve Bayes models are often less sensitive to hyperparameter tuning compared to other models because they make strong assumptions about the data, such as the independence assumption between features. The same tuning process was applied to all other models. However, no significant improvements were observed in the performance metrics for these models after tuning. Only very minor ups and downs between the model performances. This demonstrates that the initial model configuration was already optimized for the given dataset and classification task.

## Ensemble Method
Ensemble method was employed to see if we could lift the performance of the models. The ensemble approach (stacking) aims to leverage the strengths of multiple individual models to achieve better predictive performance. The following models were included in the ensemble:
- Naive Bayes (NB)
- Random Forest (RF)
- Logistic Regression (GLM)
The ensemble method works by generating predictions from each individual model and then combining these predictions using a meta-model. I used 3 top performing models for this task. For this analysis as well as a logistic regression model with the meta-model to aggregate the predictions from the base models.

## Steps to Conduct the Ensemble Method:
- Training Base Models:
Each base model (NB, RF, GLM) was trained using all the features datasets with 10-fold cross-validation to ensure robustness and avoid overfitting.

- Generating Predictions:
Predictions were generated from each base model for the training dataset. The predicted probabilities of the positive class (Class1) were used as features for the meta-model.
The training predictions from each base model were stored in a new data frame along with the actual class labels.

 - Train the Meta-Model:
A logistic regression model was trained using the predictions from the base models as input features. This meta-model learns to combine the base model predictions to improve overall classification performance.

- Evaluatr the Ensemble Model:
The ensemble model was then evaluated on the test dataset. Predictions from each base model on the test data were combined and fed into the trained logistic regression meta-model to produce final predictions.

Performance Metrics for the Meta-Model:

|               | TPR      | FPR      | Precision | Recall   | F_measure | AUC     | MCC      | Kappa    |
|---------------|----------|----------|-----------|----------|-----------|---------|----------|----------|
| Class0        | 0.666667 | 0.359477 | 0.911147  | 0.666667 | 0.769966  | 0.705226 | 0.227877 | 0.191078 |
| Class1        | 0.640523 | 0.359477 | 0.257895  | 0.640523 | 0.36773   | 0.705226 | 0.227877 | 0.191078 |
| Weighted Avg  | 0.662663 | 0.359477 | 0.811099  | 0.662663 | 0.708362  | 0.705226 | 0.227877 | 0.191078 |

Confusion Matrix for the meta-model:

|               | Predicted Class0 | Predicted Class1 |
|---------------|-------------------|------------------|
| Actual Class0 | 564               | 55              |
| Actual Class1 | 282               | 98              |

## Observations from the performance metrics of the meta model

Class0 Performance: The ensemble model achieved a TPR of 0.67, indicating that it correctly identified 67.7% of Class0 instances. The precision for Class0 was high at 0.91, showing that the model was relatively accurate in its positive predictions for Class0.
Class1 Performance: The model's performance for Class1 was lower, with a TPR of 0.64, meaning it correctly identified 64.0% of Class1 instances. The precision for Class1 was 0.26, which is relatively low.
Weighted Average Performance: The weighted average metrics provide a balanced view considering the distribution of both classes in the dataset. The ensemble method achieved a weighted average TPR of 0.66, precision of 0.81, and F_measure of 0.71.
Overall Performance: The AUC of 0.70 indicates moderate discriminative ability of the ensemble model. The Kappa statistic of 0.19 reflects slight agreement between the predictions and the actual class labels, suggesting that the ensemble method did not provide a substantial improvement over individual models.
These results indicate that the ensemble method, despite combining multiple models, did not significantly enhance the performance. The Naive Bayes model on the ros_chi_sq dataset remains the best-performing model.

## Conclusion and future improvements
This project aimed to enhance the performance of classification models through comprehensive data preparation, data balancing, feature selection, classification model building and training, hyperparameter tunning, ensemble method, feature engineering, and model evaluation. By leveraging advanced techniques in feature creation and data balancing, I sought to improve the predictive accuracy and reliability of various machine learning models. The use of various method and try to improve the performance of the models were crucial in understanding the ability of different techniques can do to classification models. Total 60 models were train and tuned in the process. However, the performance for all models were on par with each other. Even with intensive hyperparameter tunning and implementing the ensemble method, the results remained similar. 
However, this project demonstrates the complexity of data mining, there are challenges throughout the whole process from data cleaning to fine tunning the model parameters. Finding ways to tackle the problems and searching for better solutions is very difficult but at the same time very rewarding. Especially when I found a model combined with different techniques that can perform well than others. It is like climbing a mountain, you don’t know which way is going to lead you to the peak, so you have to keep trying until you find the right path with the right tools.
I will continue to contribute to this project by trying different models as well as feature engineering mehtods (maybe aggregating the features). I encourage anyone who is interested in data science projects to play around with this data set and see if you could bring the TPR for Class0 and Class1 up tp around 80%. 

