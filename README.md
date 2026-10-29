# ML Assignment 2 - Classification Models and Streamlit Deployment

## a. Problem Statement

Build a classification system using 5 different machine learning models on a single dataset, evaluate each model using 6 metrics (Accuracy, AUC, Precision, Recall, F1, MCC), and deploy an interactive Streamlit web application that loads pre-trained models and evaluates them on user-uploaded test data.

## b. Dataset Description

- **Name:** Telco Customer Churn
- **Source:** Kaggle (https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Features:** 20 columns (customerID + 18 features + Churn target), cleaned and one-hot encoded to 23 features after dropping customerID
- **Instances:** 7043 (5174 stayed, 1869 churned)
- **Target:** Binary classification (0 = No Churn, 1 = Churn)
- **Feature types:** Numeric (tenure, MonthlyCharges, TotalCharges) and categorical (Contract, PaymentMethod, InternetService, etc.)

## c. GitHub Repository Link

[Insert your GitHub repository link here]

## d. Models Used

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.7353 | 0.8464 | 0.5008 | 0.7941 | 0.6143 | 0.4545 |
| Decision Tree | 0.7140 | 0.8305 | 0.4770 | 0.8048 | 0.5990 | 0.4315 |
| KNN | 0.7842 | 0.8289 | 0.6000 | 0.5615 | 0.5801 | 0.4356 |
| Naive Bayes | 0.7488 | 0.8302 | 0.5182 | 0.7594 | 0.6161 | 0.4568 |
| Support Vector Machine | 0.7509 | 0.8274 | 0.5205 | 0.7807 | 0.6246 | 0.4698 |

### Observations

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Acc=0.7353, MCC=0.4545, F1=0.6143, Recall=0.7941 (MCC rank 3/5). Tuned C=10, penalty=l1, balanced class weights. Simple linear model with a small train-test gap - a strong, explainable baseline. |
| Decision Tree | Acc=0.7140, MCC=0.4315, F1=0.5990, Recall=0.8048 (MCC rank 5/5). Tuned max_depth=5, min_samples_leaf=20, balanced class weights. Depth and leaf limits stop it from memorizing the training data. |
| KNN | Acc=0.7842, MCC=0.4356, F1=0.5801, Recall=0.5615 (MCC rank 4/5). Tuned n_neighbors=31, metric=manhattan, weights=uniform. Gained from removing duplicate columns, but sklearn KNN has no class weights so recall stays lower. |
| Naive Bayes | Acc=0.7488, MCC=0.4568, F1=0.6161, Recall=0.7594 (MCC rank 2/5). Tuned var_smoothing=0.1. Biggest gain from dropping the perfectly correlated dummies and log-transforming TotalCharges, since NB assumes independent, bell-shaped features. |
| Support Vector Machine | Acc=0.7509, MCC=0.4698, F1=0.6246, Recall=0.7807 (MCC rank 1/5). Tuned C=1, gamma=scale, balanced class weights. RBF kernel captures non-linear patterns, and class weighting lifts recall on the churn class. |
| Overall, Winner for your dataset? | SVM (highest MCC = 0.4698) |

## e. Methodology Notes

### Preprocessing

- Dropped customerID (identifier), converted TotalCharges to numeric (11 blank values filled with 0)
- One-hot encoded 15 categorical columns with drop_first=True, giving 30 columns
- Removed 7 duplicate dummies: the 6 `*_No internet service` columns are identical to `InternetService_No`, and `MultipleLines_No phone service` equals 1 - `PhoneService_Yes`. This leaves 23 features
- Log-transformed TotalCharges (log1p) because it is heavily right-skewed
- 80/20 stratified train-test split (5634 train, 1409 test), StandardScaler fitted on training data only

### Hyperparameter Tuning

All 5 models were tuned with GridSearchCV using 5-fold StratifiedKFold on the training set. Stratified folds keep the 73.5 / 26.5 churn ratio in every fold. MCC was used as the scoring metric, the same metric used to pick the winner, because accuracy is misleading on imbalanced data. For Logistic Regression, Decision Tree and SVM the grid also chose between no class weighting and `class_weight='balanced'`, which gives the minority churn class more weight. Balanced weighting trades some accuracy and precision for much higher recall, which is the right trade for churn: a missed churner is lost revenue, a false alarm only costs a retention offer.

| ML Model Name | Search Space | Best Params | Best CV MCC |
| --- | --- | --- | --- |
| Logistic Regression | C: [0.01, 0.1, 1, 10, 100], penalty: [l1, l2], class_weight: [None, balanced] | C=10, class_weight=balanced, penalty=l1 | 0.4877 |
| Decision Tree | max_depth: [3, 5, 7, 10, None], min_samples_split: [2, 5, 10], min_samples_leaf: [1, 5, 10, 20], criterion: [gini, entropy], class_weight: [None, balanced] | class_weight=balanced, criterion=entropy, max_depth=5, min_samples_leaf=20, min_samples_split=2 | 0.4411 |
| KNN | n_neighbors: [11, 15, 21, 25, 31, 41], weights: [uniform, distance], metric: [euclidean, manhattan] | metric=manhattan, n_neighbors=31, weights=uniform | 0.4634 |
| Naive Bayes | var_smoothing: [1e-9, 1e-7, 1e-5, 1e-3, 1e-2, 1e-1] | var_smoothing=0.1 | 0.4767 |
| Support Vector Machine | C: [0.5, 1, 2, 5], gamma: [scale, auto], class_weight: [None, balanced] (RBF kernel) | C=1, class_weight=balanced, gamma=scale | 0.4741 |

### Overfitting Check (Learning Curves)

Learning curves (training F1 vs 3-fold CV F1 over increasing training sizes) were plotted for every tuned model. A gap above 0.15 is treated as overfitting, and a CV F1 below 0.45 as underfitting.

| ML Model Name | Train F1 | CV F1 | Gap | Verdict |
| --- | --- | --- | --- | --- |
| Logistic Regression | 0.6391 | 0.6334 | 0.0057 | Good fit |
| Decision Tree | 0.6275 | 0.6078 | 0.0197 | Good fit |
| KNN | 0.6196 | 0.5948 | 0.0248 | Good fit |
| Naive Bayes | 0.6261 | 0.6276 | -0.0015 | Good fit |
| Support Vector Machine | 0.6895 | 0.6274 | 0.0621 | Good fit |

### Metric Used to Pick the Winner

The winner is picked on MCC (Matthews Correlation Coefficient). With 73.5% non-churners, a model that always predicts No Churn gets 73.5% accuracy but MCC = 0. MCC uses all four cells of the confusion matrix (TP, TN, FP, FN), so it only rewards models that actually separate churners from non-churners. F1 is used as a secondary check.
