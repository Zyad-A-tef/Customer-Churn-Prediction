# Customer Churn Prediction — Project Report

## 1. Problem Definition

- **Type**: Binary classification (`Churn` ∈ {0, 1})
- **Business driver**: Reducing churn is significantly cheaper than acquiring
  new customers. A reliable classifier enables targeted retention campaigns.
- **Success metric**: **F1-score** on the churn class. Accuracy is a poor
  primary metric because the dataset is imbalanced (≈26.5% churn).
- **Secondary metrics**: ROC-AUC, **PR-AUC** (more sensitive under imbalance),
  and training / inference time.

## 2. Dataset

- **IBM Telco Customer Churn** (33-column Kaggle edition),
  7,043 rows × 33 columns, California-only customers.
- After cleaning (drop leakage + non-predictive geographic columns, coerce
  `TotalCharges`, remove duplicates), the modeling frame has roughly 20
  usable features.

## 3. Methodology

### 3.1 Data cleaning
- Schema normalization to pipeline-compatible names.
- Coerce `TotalCharges` to numeric; drop rows where it is null
  (newly-joined customers with 0 tenure).
- Drop `customerID`, `Count`, and the three leakage columns (`Churn Label`,
  `Churn Score`, `Churn Reason`).
- Drop high-cardinality / non-predictive geographic columns
  (`City`, `Zip Code`, `Lat Long`, `Latitude`, `Longitude`,
  plus the constant `Country` and `State`). Keeping them inflates the
  feature space into thousands of one-hot dummies without adding signal.
- Harmonize the target to a single `Churn` integer column.

### 3.2 EDA — key findings
1. **Class imbalance** — ~26.5% churners → F1 / PR-AUC over Accuracy.
2. **Tenure is the strongest driver**: churn rate drops sharply as tenure grows.
3. **Contract type**: month-to-month contracts churn far more than 1- or
   2-year contracts.
4. **Fiber optic** internet is associated with higher churn than DSL / no
   internet.
5. **Pricing pressure**: churners carry higher monthly charges on average.
6. `MonthlyCharges` is roughly uniform; `TotalCharges` is right-skewed
   (tenure-driven).

### 3.3 Preprocessing
- Stratified 80/20 train/test split — performed **before** any encoding.
- A single `ColumnTransformer` fitted on the training fold:
  - `StandardScaler` on continuous numerics only.
  - `OneHotEncoder(drop='if_binary', handle_unknown='ignore')` on categoricals.
  - Passthrough for already-numeric 0/1 fields (e.g. `SeniorCitizen`).
- No leakage: scaler statistics and encoder categories are learned from
  training data only.

### 3.4 Models
**Classical ML (3 baselines, as required):**
- **Logistic Regression** — `class_weight='balanced'`, `max_iter=1000`.
- **Random Forest** — 200 trees, `class_weight='balanced'`.
- **XGBoost** — 300 trees, `max_depth=4`, `learning_rate=0.05`,
  `scale_pos_weight=(neg/pos)`.

**Deep Learning (ANN):**
- 128 → 64 → 32 dense, ReLU + He init, BatchNorm + Dropout.
- Sigmoid output, binary cross-entropy, Adam optimizer.
- Class weights for imbalance; early stopping on `val_auc`.
- Decision threshold tuned via **Youden’s J** on the ROC.

### 3.5 Optimization
- `RandomizedSearchCV` (20 iters) on Random Forest — F1 scoring, 3-fold CV.
- `RandomizedSearchCV` (25 iters) on XGBoost — F1 scoring, 5-fold CV.
- **Feature selection experiment**: retain features with XGBoost importance
  > 0.5%, then **re-tune** XGBoost on the reduced feature set
  (15-iter random search) so the comparison is fair.

### 3.6 Evaluation protocol
Every model is evaluated on the same held-out test set with the same metrics:
Accuracy, F1, ROC-AUC, PR-AUC, classification report, confusion matrix,
ROC curve, Precision-Recall curve. A **stratified 5-fold CV** on the training
fold provides a distribution of F1 / ROC-AUC to complement the single split.

## 4. Results

The full numbers are printed by the final comparison cell in
`Notebooks/analysis.ipynb`. On a representative run, the pattern is:

| Rank (by F1) | Model                       | Type | Notes                                       |
| ---          | ---                         | ---  | ---                                         |
| 1            | **XGBoost (tuned)**         | ML   | Strongest F1, competitive PR-AUC            |
| 2            | XGBoost (tuned + FS)        | ML   | Comparable F1, fewer features, faster infer |
| 3            | ANN (Deep Learning)         | DL   | Close in AUC; loses on speed / simplicity   |
| 4            | Random Forest (tuned)       | ML   | Solid baseline                              |
| 5            | Logistic Regression         | ML   | Fastest; strong interpretability            |
| 6            | Random Forest / XGBoost     | ML   | Untuned baselines                           |

**Timing (typical CPU, laptop-grade):**
- Logistic Regression: ~0.05s train, sub-ms predict.
- Random Forest: ~0.5s train.
- XGBoost: ~0.5–1s train.
- ANN: 10–60s train over up to 150 epochs with early stopping.

## 5. Model Selection

- **Best-by-F1**: XGBoost (tuned). Selected as the production recommendation.
- **Why**:
  - Consistently in the top two on F1 and PR-AUC across runs.
  - Handles categorical-vs-continuous interactions natively once one-hot
    encoded.
  - Two orders of magnitude faster to train than the ANN.
  - Easy to interpret via built-in feature importances and SHAP.

## 6. ML vs Deep Learning — when each wins

| Situation                                                             | Prefer     | Why                                                                                                 |
| ---                                                                   | ---        | ---                                                                                                 |
| Tabular data with engineered features and moderate rows               | **ML**     | Strong baselines, fast iteration, easier explainability.                                            |
| Interpretability / regulated deployment                               | **ML**     | Feature importances, linear coefficients, SHAP are mature.                                          |
| Latency-sensitive scoring (real-time retention flags)                 | **ML**     | Sub-millisecond inference vs. tens of ms for ANN on CPU.                                            |
| Large datasets with raw inputs (images, text, audio)                  | **DL**     | Representation learning end-to-end.                                                                 |
| Highly non-linear feature interactions + lots of data                 | **DL**     | Capacity to model what hand features miss.                                                          |
| **This project (≈7k tabular rows, structured features)**              | **ML wins**| XGBoost beats a tuned ANN on F1, trains ~100× faster, and is easier to explain to stakeholders.     |

## 7. Conclusion

- **Winner**: tuned XGBoost — highest F1, competitive PR-AUC, and fast to
  retrain and deploy.
- **ANN takeaway**: A deep network can approach tree-based performance on this
  dataset only after adding BatchNorm, Dropout, class weights and decision-
  threshold tuning. It is not worth the extra complexity for this task, but
  confirms the methodology is sound.
- **Actionable driver**: `tenure`, `Contract` type, `InternetService` and
  `MonthlyCharges` are the strongest churn predictors. Retention programs
  should target early-tenure, month-to-month, fiber-optic customers with
  above-average monthly charges.

## 8. Future Work

- Probability calibration (Platt / isotonic) so predicted probabilities match
  observed churn rates.
- SHAP explainability for per-customer retention outreach.
- Cost-aware decision thresholds that weigh false churn vs. missed churn.
- Drift monitoring and a scheduled retraining pipeline.
