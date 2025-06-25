# customer-lifecycle-and-churn-prediction-models #
## Overview 
Predicting customer churn using machine learning with Kaplan-Meier survival analysis and RFM features. Includes SMOTE and 10-fold CV to compare models and identify the most reliable for future use.

## Business Context
Understanding when and why customers churn helps businesses design targeted retention strategies. This project combines traditional RFM (Recency, Frequency, Monetary) analysis with Kaplan-Meier survival analysis and predictive machine learning models (Regression, Randon Forest, Gradient Boost and XGBoost) to offer a comprehensive view of customer lifecycle.

## Methodology
- **Data Processing:**
- Cleaned and aggregated transactional data.
- Computed features like recency, frequency, monetary value, tenure, and interpurchase time.

### Behavioural Profiling
- **RFM (Recency, Frequency, Monetary)** features used to segment customers.
- **Survival Analysis:** Applied Kaplan-Meier estimator to model customer retention over time and segmented customers based on RFM quantiles.
  
- **Predictive Modelling:** Built and compared multiple classification models (Logistic Regression, Decision Tree, Random Forest, XGBoost) to predict churn.
- **Imbalanced Data Handling:** Used SMOTE to balance the churn classes in the training data.
- **Model Validation:** Employed 10-fold cross-validation to evaluate model performance using accuracy, precision, recall, F1-score, and specificity.

### Model Evaluation Metric
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
#### Interpretation
While all models achieved very high accuracy, XGBoost offers the most reliable and balanced performance across all key metrics:
- **High Precision (96.90%)**: It produces very few false positives, which means fewer loyal customers are mistakenly flagged as churn risks.
- **High Recall (98.65%)**: It successfully identifies nearly all customers who are likely to churn — ideal for retention strategies.
- **Strong F1 Score (97.63%)**: It balances both precision and recall, making it well-suited for customer churn prediction where both false alarms and missed churners have business costs.
- **High Specificity (97.33%)**: It accurately identifies customers who are not at risk, helping target interventions more efficiently.
Compared to other models, XGBoost consistently performs well across all KPIs, making it a strong candidate for deployment in churn prediction systems.

## Data Source
The dataset used in this project is originally sourced from(https://www.kaggle.com/datasets/carrie1/ecommerce-data/data) . It contains historical e-commerce transactions and has been adapted for RFM segmentation, churn prediction, and CLV analysis.
Note: The raw dataset is not included in this repository due to size and licensing restrictions. You may download it directly from Kaggle and place it in the working directory as `rfm_ecommerce_data.csv`.

##  How to Run
This notebook was developed and tested in **Google Colab**, where packages like `lifelines`, `imbalanced-learn`, and `xgboost` are pre-installed.

###  Run in Google Colab  
Simply open the `.ipynb` notebook in Colab and run all cells.

### Run Locally (Jupyter on Mac or Windows)  
If using Jupyter locally, install dependencies:

```bash
pip install lifelines imbalanced-learn xgboost scikit-learn matplotlib seaborn
