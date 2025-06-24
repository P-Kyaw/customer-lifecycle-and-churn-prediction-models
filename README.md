# customer-lifecycle-and-churn-prediction-models #
## Overview 
Predicting customer churn using machine learning with Kaplan-Meier survival analysis and RFM features. Includes SMOTE and 10-fold CV to compare models and identify the most reliable for future use.

## Business Context
Understanding when and why customers churn helps businesses design targeted retention strategies. This project combines traditional RFM (Recency, Frequency, Monetary) analysis with Kaplan-Meier survival analysis and predictive machine learning models (Regression, Randon Forest, Gradient Boost and XGBoost) to offer a comprehensive view of customer lifecycle.

## Methodology
- **Data Processing:** Cleaned and aggregated transactional data to compute key features such as recency, frequency, monetary value, tenure, and interpurchase time.
- **Survival Analysis:** Applied Kaplan-Meier estimator to model customer retention over time and segmented customers based on RFM quantiles.
- **Imbalanced Data Handling:** Used SMOTE to balance the churn classes in the training data.
- **Model Validation:** Employed 10-fold cross-validation to evaluate model performance using accuracy, precision, recall, F1-score, and specificity.
- **Predictive Modelling:** Built and compared multiple classification models (Logistic Regression, Decision Tree, Random Forest, XGBoost) to predict churn.
# Model Evaluation
To compare the effectiveness of different churn prediction models, we used 10-fold cross-validation on SMOTE-balanced data. Each model was evaluated based on:
- **Accuracy**: Overall correct predictions
- **Precision**: Correct churn predictions vs all churn predictions
- **Recall**: Ability to identify actual churners
- **F1 Score**: Balance between precision and recall
- **Specificity**: Ability to correctly identify non-churners
  
## Results Summary
- Customer Lifetime Value (CLV) is estimated by combining survival analysis with average order value and purchase frequency.
- Kaplan-Meier curves provide insights into customer retention patterns across segments.
- Comparative model evaluation identifies the most reliable predictive model for churn.
### Results
| Model               | Accuracy | Precision | Recall | F1 Score | Specificity |
|---------------------|----------|-----------|--------|----------|-------------|
| Logistic Regression | 99.09%   | 70.83%    | 98.14% | 78.61%   | 97.17%      |
| Decision Tree       | 99.99%   | 99.99%    | 98.25% | 99.02%   | 96.50%      |
| Random Forest       | 99.99%   | 99.99%    | 99.08% | 99.52%   | 98.17%      |
| XGBoost             | 99.94%   | 96.90%    | 98.65% | 97.63%   | 97.33%      |


## How to Run
1. This notebook was originally developed and tested in **Google Colab**, which supports packages like `lifelines`, `imbalanced-learn`, and `xgboost` without additional setup.  
If you're running it locally (e.g. on a Mac using Jupyter), you may need to install these dependencies manually:
```bash
pip install lifelines imbalanced-learn xgboostClone this repository:
   `
