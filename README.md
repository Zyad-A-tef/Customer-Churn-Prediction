# Customer Churn Prediction — Machine Learning vs Deep Learning (ANN)

A DEPI / MCIT / eYouth capstone project that benchmarks classical Machine
Learning models against a Deep Learning (ANN) model on the **IBM Telco Customer
Churn** dataset, and justifies when each approach is appropriate for tabular
classification problems.

## Introduction

Customer churn is one of the most expensive problems in the telecom industry:
acquiring a new customer costs several times more than retaining an existing
one. A reliable churn classifier turns a reactive retention process into a
proactive one — at-risk customers can be surfaced early and targeted with
personalized offers before they leave.

## Problem Statement

Given customer demographics, account information, service usage, contract and
billing details, predict whether a customer will churn.

- **Task type**: Binary classification
- **Target**: `Churn` (1 = churn, 0 = retain)
- **Primary metric**: **F1-score** on the churn class (imbalanced target)
- **Secondary metrics**: ROC-AUC, PR-AUC, training / inference time

## Dataset Description

- **Source**: [Kaggle — yeanzc/telco-customer-churn-ibm-dataset](https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset)
- **Local file**: `data/Telco_customer_churn.xlsx`
- **Size**: 7,043 customers × 33 raw columns
- **Target prevalence**: ~26.5% churn (class imbalance)

### Feature groups

| Group | Examples |
| --- | --- |
| Demographics | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| Account | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod` |
| Services | `InternetService`, `PhoneService`, `OnlineSecurity`, `TechSupport`, `StreamingTV`, … |
| Financial | `MonthlyCharges`, `TotalCharges`, `CLTV` |

High-cardinality / non-predictive geographic columns (`City`, `Zip Code`,
`Lat Long`, `Latitude`, `Longitude`, `Country`, `State`) and leakage columns
(`Churn Label`, `Churn Score`, `Churn Reason`, `Count`) are dropped during
cleaning.

## Methodology

1. **Problem definition** — binary classification, F1 as primary metric.
2. **Data reading** — load from local Excel, fall back to KaggleHub.
3. **Data cleaning** — schema normalization, `TotalCharges` coercion, duplicate
   removal, drop of leakage + non-predictive geographic columns, target
   harmonization to a single `Churn` column.
4. **EDA** — target distribution, numeric distributions with skew/IQR outlier
   table, categorical churn-rate bars, correlation heatmap, tenure-vs-price
   scatter.
5. **Preprocessing** — stratified train/test split first, then a single
   `ColumnTransformer`:
   - `StandardScaler` on continuous features only
     (`tenure`, `MonthlyCharges`, `TotalCharges`, `CLTV`)
   - `OneHotEncoder(drop='if_binary', handle_unknown='ignore')` on categoricals
   - Passthrough for already-numeric 0/1 fields
   - Fitted on training fold only → no leakage
6. **Model building** — three ML baselines + one ANN:
   - Logistic Regression (`class_weight='balanced'`)
   - Random Forest (`class_weight='balanced'`)
   - XGBoost (`scale_pos_weight` for imbalance)
   - ANN — 128 → 64 → 32 dense, BatchNorm + Dropout, Adam, early stopping
7. **Evaluation** — identical metrics for every model:
   Accuracy, F1, ROC-AUC, PR-AUC, classification report, confusion matrix,
   ROC + PR curves, **stratified 5-fold CV** on the training fold.
8. **Optimization** — `RandomizedSearchCV` on Random Forest and XGBoost
   (20–25 iterations, F1 as scoring); feature-selection experiment on XGBoost
   with its own re-tune to keep the comparison fair.
9. **Final selection** — pick the model with the best F1; secondary tie-breakers
   are PR-AUC and training / inference time.

## Project structure

```
Customer-Churn-Prediction/
├── data/
│   └── Telco_customer_churn.xlsx
├── notebooks/
│   └── analysis.ipynb          ← end-to-end pipeline + narrative
├── src/
│   ├── data_cleaning.py
│   ├── preprocessing.py
│   ├── model.py
│   └── evaluation.py
├── img/                        ← logos used in the notebook
├── report.md                   ← results-focused write-up
├── requirements.txt
└── README.md
```

## Results

Metrics are computed on the same held-out 20% stratified test split and logged
in the final comparison table inside the notebook. See
[`report.md`](report.md) for the full discussion and the ML-vs-DL justification.

## Conclusion

- On medium-sized tabular data with engineered features, well-tuned gradient
  boosting (**XGBoost**) typically matches or beats a modest ANN on F1 while
  training **orders of magnitude faster** and remaining easier to explain.
- The ANN benefits from class weighting and threshold tuning (Youden’s J);
  without those it is noticeably dominated by tree-based models.
- **Recommended production model**: tuned XGBoost (or its feature-selection
  variant), with probability outputs fed into a retention-cost-based threshold.

### Future work

- Probability calibration (Platt / isotonic) so the outputs match observed
  churn rates.
- SHAP explanations for per-customer retention outreach.
- Drift monitoring and periodic retraining.

## How to reproduce

```bash
pip install -r requirements.txt
jupyter lab Notebooks/analysis.ipynb
```

Then run all cells top to bottom. The notebook prefers the local
`data/Telco_customer_churn.xlsx` file and falls back to KaggleHub if missing.

## Team

- Abdallah Mohamed Fahmy
- Abdelrahman Mahmoud Ahmed
- Abdlrhman Hisham Ismail
- Amgad Mohamed Mohamed
- Omar Tarek Emam
- Zyad Atef
