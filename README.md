# Credit Risk Underwriting Engine

An end-to-end machine-learning system for predicting loan default risk from credit and borrower information.

The project combines financial feature engineering, classification models, hyperparameter optimization, model explainability, and API serving to create a credit-risk underwriting pipeline.

---

# Project Objective

The objective is to estimate the probability that a borrower will default on a loan.

The system takes borrower/application information and historical credit-bureau information as input and produces a risk prediction that can be consumed by an underwriting API.

```text id="1y7c5f"
Borrower Data
     +
Credit Bureau History
          ↓
    Data Processing
          ↓
  Feature Engineering
          ↓
   Machine Learning
          ↓
 Default Probability
          ↓
   Risk Decision
          ↓
     FastAPI
```

---

# Key Features

* Credit-risk classification
* Historical bureau-data aggregation
* Financial feature engineering
* Missing-value and anomaly handling
* Categorical feature processing
* Class-imbalance handling
* Logistic Regression baseline
* Random Forest experimentation
* XGBoost final model
* Optuna hyperparameter optimization
* Precision-Recall based evaluation
* SHAP explainability
* Probability-based predictions
* FastAPI inference service
* Model persistence

---

# Overall Architecture

```text id="g8e2lz"
                 Borrower Application
                         │
                         ▼
              ┌────────────────────┐
              │ Data Preprocessing │
              └─────────┬──────────┘
                        │
                        ▼
              ┌────────────────────┐
              │ Feature Engineering│
              └─────────┬──────────┘
                        │
                        ▼
              ┌────────────────────┐
              │ Train / Validation │
              │      Split         │
              └─────────┬──────────┘
                        │
             ┌──────────┴───────────┐
             │                      │
             ▼                      ▼
      Logistic Regression     Random Forest
             │                      │
             └──────────┬───────────┘
                        ▼
                    XGBoost
                        │
                        ▼
                 Optuna Tuning
                        │
                        ▼
                Final ML Model
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
           SHAP                  API
       Explainability          FastAPI
                                  │
                                  ▼
                         Underwriting Decision
```

---

# 1. Data Sources

The project works with borrower/application information together with historical credit-bureau records.

The bureau data can contain multiple records for the same borrower.

For example:

```text id="y8j3m4"
Borrower A
 ├── Previous Loan 1
 ├── Previous Loan 2
 ├── Previous Loan 3
 └── Previous Loan 4
```

However, the ML model needs a borrower-level feature representation.

Therefore, the historical records are aggregated into borrower-level features.

---

# 2. Bureau Data Aggregation

The relationship can be viewed as:

```text id="j0k6g2"
One Borrower
     │
     ├── Historical Credit Record
     ├── Historical Credit Record
     ├── Historical Credit Record
     └── Historical Credit Record
```

These records are aggregated to produce features describing the borrower's historical credit behaviour.

This prevents the model from treating each historical loan as an independent borrower.

The resulting dataset becomes approximately:

```text id="9m3y9f"
One Row = One Borrower
```

with features representing their credit history.

---

# 3. Data Cleaning

Real-world financial datasets contain missing values, inconsistent values and anomalies.

One important example is the `DAYS_EMPLOYED` field.

An anomalous value such as:

```text id="u6t9lq"
365243
```

is treated as an invalid/missing representation rather than a genuine employment duration.

The general pipeline is:

```text id="9d0xak"
Raw Data
   ↓
Detect anomalies
   ↓
Handle missing values
   ↓
Encode categorical variables
   ↓
Model-ready dataset
```

---

# 4. Feature Engineering

Financial ratios are created to provide the model with more meaningful representations of borrower behaviour.

Examples include:

### Credit-to-Income Ratio

```text id="4o8t0e"
CREDIT_INCOME_RATIO
=
Credit Amount / Income
```

This provides information about the size of the requested credit relative to the borrower's income.

### Annuity-to-Income Ratio

```text id="7g0m9m"
ANNUITY_INCOME_RATIO
=
Annuity / Income
```

### Credit Term

Represents the relationship between credit amount and repayment/annuity characteristics.

### Employment Percentage

A feature derived from employment duration relative to the relevant age/time information.

### High Leverage Flag

A binary feature identifying borrowers with comparatively high leverage.

---

# 5. Train/Test Split

The dataset is divided into training and testing data.

```text id="5m7x6p"
Dataset
   │
   ├── 80% → Training
   │
   └── 20% → Testing
```

Because the target is highly imbalanced, stratification is used to preserve the class distribution between the splits.

---

# 6. Class Imbalance

The dataset has approximately:

```text id="6z5e9p"
92% → Non-default
8%  → Default
```

Therefore, accuracy alone is not an appropriate metric.

For example, a model predicting:

```text id="l3o6pn"
"Non-default"
```

for every borrower could achieve around 92% accuracy while completely failing to identify defaulters.

This makes metrics such as:

* Precision
* Recall
* PR-AUC / Average Precision

more useful.

---

# 7. Model Development

The models were developed progressively rather than directly choosing the final model.

## Logistic Regression

Used as the initial baseline.

Why?

* Simple
* Interpretable
* Fast
* Provides a useful benchmark

```text id="d3n5q0"
Credit Features
      ↓
Logistic Regression
      ↓
Baseline Probability
```

---

## Random Forest

The next experiment was Random Forest.

Why?

Random Forest can capture:

* nonlinear relationships
* feature interactions

and does not require the same linear assumptions as Logistic Regression.

```text id="o8u6s3"
Features
   ↓
Multiple Decision Trees
   ↓
Random Forest
   ↓
Risk Prediction
```

---

## XGBoost

XGBoost was then used as the main model because the problem is structured/tabular data and boosting methods are particularly effective for this type of problem.

Conceptually:

```text id="w0f4nz"
Data
 ↓
Tree 1
 ↓
Errors
 ↓
Tree 2
 ↓
Errors
 ↓
Tree 3
 ↓
...
 ↓
Final Prediction
```

Each successive tree attempts to improve the overall prediction by learning from previous errors.

---

# 8. Handling Class Imbalance with `scale_pos_weight`

XGBoost provides `scale_pos_weight` to increase the importance of the minority class during training.

In this project, the default class is the minority class.

Conceptually:

```text id="3glj0f"
Normal examples → normal weight

Default examples
       ↓
Higher weight
       ↓
Model pays more attention
```

This helps the model focus more on correctly identifying borrowers who may default.

---

# 9. Hyperparameter Optimization

Instead of manually selecting XGBoost parameters, Optuna was used for hyperparameter optimization.

Parameters can include:

```text id="4w2l1h"
learning_rate
max_depth
n_estimators
subsample
colsample_bytree
min_child_weight
```

The process is:

```text id="5k3w8m"
Optuna
  ↓
Suggest parameters
  ↓
Train model
  ↓
Evaluate
  ↓
Try another configuration
  ↓
Find strong configuration
```

The model was tuned toward **Precision-Recall AUC**, which is more informative than accuracy for this imbalanced problem.

---

# 10. Model Evaluation

The main evaluation metrics include:

### Precision

Of the borrowers predicted as default:

```text id="p4g8n2"
How many actually defaulted?
```

### Recall

Of all actual defaulters:

```text id="e7x1s9"
How many did the model identify?
```

### PR-AUC / Average Precision

Measures the quality of precision-recall performance across different classification thresholds.

The tuned model achieved approximately:

```text id="4q8s1m"
PR-AUC ≈ 0.2633
```

The important point is that the model should be evaluated against an appropriate baseline and business objective rather than interpreting this number in isolation.

---

# 11. Probability Prediction

Instead of only producing:

```text id="y5b1v6"
Default
Non-default
```

the model can produce a probability.

For example:

```text id="6v0f8p"
Borrower A → 0.12
Borrower B → 0.73
Borrower C → 0.31
```

This is useful for an underwriting system because the probability can be used as an input to a decision policy.

---

# 12. Classification Threshold

The probability does not automatically determine the final business decision.

For example:

```text id="p8d2h5"
Probability > threshold
        ↓
Classify as higher risk
```

Changing the threshold changes the trade-off between precision and recall.

Therefore, the threshold should ideally be selected according to the business objective and cost of false positives vs false negatives.

---

# 13. SHAP Explainability

A major component of the project is model explainability.

SHAP helps explain how individual features contributed to a model prediction.

Example:

```text id="1j5x8k"
Prediction: High Risk

Factors increasing risk:
  ↓ High credit/income ratio
  ↓ High leverage

Factors reducing risk:
  ↑ Stable employment history
```

This is useful because a financial institution may need to understand **why** a model produced a particular prediction.

---

# 14. Global vs Local Explainability

### Global

Answers:

> "Which features are generally important to the model?"

Example:

```text id="f5k0q8"
Income
Credit Amount
Employment
Previous Credit Behaviour
```

### Local

Answers:

> "Why did the model make this prediction for this particular borrower?"

This distinction is important when discussing explainable ML.

---

# 15. FastAPI Inference Service

The trained model is exposed through a FastAPI service.

Conceptually:

```text id="b8r2s7"
Client
  │
  │ Borrower JSON
  ▼
FastAPI
  │
  ▼
Preprocessing
  │
  ▼
Trained Model
  │
  ▼
Probability
  │
  ▼
Underwriting Response
```

This converts the ML model from an offline notebook/model into something that can be consumed by another application.

---

# 16. Model Persistence

The trained model is saved using model serialization so that it does not need to be retrained every time the API starts.

```text id="f9n3x2"
Training
   ↓
Trained Model
   ↓
Save
   ↓
Model File
   ↓
FastAPI Startup
   ↓
Load Model
   ↓
Inference
```

---

# 17. Production-Oriented Flow

The complete system can therefore be viewed as:

```text id="u7k4z1"
             Borrower Request
                    │
                    ▼
             FastAPI Endpoint
                    │
                    ▼
             Input Validation
                    │
                    ▼
           Feature Processing
                    │
                    ▼
             XGBoost Model
                    │
                    ▼
           Default Probability
                    │
             ┌──────┴──────┐
             │             │
             ▼             ▼
        Risk Decision    SHAP
                         Explanation
             │             │
             └──────┬──────┘
                    ▼
             API Response
```

---

# Technology Stack

| Technology   | Purpose                     |
| ------------ | --------------------------- |
| Python       | ML pipeline and backend     |
| Pandas       | Data processing             |
| NumPy        | Numerical operations        |
| Scikit-learn | Preprocessing/evaluation    |
| XGBoost      | Final classification model  |
| Optuna       | Hyperparameter optimization |
| SHAP         | Model explainability        |
| FastAPI      | Model-serving API           |
| Joblib       | Model persistence           |
| Git/GitHub   | Version control             |

---

# Project Structure

A simplified logical structure is:

```text id="s5n8c2"
CreditRiskPrediction/
│
├── data/
│
├── notebooks/
│
├── src/
│   ├── preprocessing
│   ├── feature_engineering
│   ├── model_training
│   └── evaluation
│
├── model/
│
├── api/
│
├── requirements.txt
└── README.md
```

The exact organization may vary depending on the implementation.

---

# Key Challenges

## 1. Class Imbalance

Only around 8% of the observations belong to the default class.

Solution:

```text id="q6t1y7"
Class-aware training
+
Precision/Recall
+
PR-AUC
```

---

## 2. One-to-Many Bureau Data

A borrower can have multiple historical credit records.

Solution:

```text id="d0v4k9"
Multiple Historical Records
          ↓
      Aggregation
          ↓
One Borrower-Level Representation
```

---

## 3. Feature Engineering

Raw financial variables don't always directly express borrower risk.

Solution:

Create meaningful ratios and behavioural features such as:

```text id="z8j3c6"
Credit / Income
Annuity / Income
Employment-related features
Leverage indicators
```

---

## 4. Model Selection

Instead of assuming one model is best:

```text id="r7c5x4"
Logistic Regression
       ↓
Random Forest
       ↓
XGBoost
```

Models were progressively evaluated and the stronger approach was further optimized.

---

## 5. Explainability

A prediction alone is not sufficient in a financial-risk setting.

SHAP was used to understand feature-level contributions to predictions.

---

# Future Improvements

The current system establishes the core credit-risk ML pipeline. The next development stages I would explore are:

### Neural Network Experiment

Build an MLP for the same tabular credit-risk problem and compare it with:

```text id="x4m9s2"
Logistic Regression
Random Forest
XGBoost
Neural Network
```

The objective would not simply be to replace XGBoost, but to understand the performance and trade-offs of neural networks on this dataset.

---

### Probability Calibration

Because the model outputs default probabilities, calibration can be explored to ensure predicted probabilities better correspond to observed default frequencies.

---

### Model Monitoring

Monitor:

* prediction distributions
* input feature distributions
* data drift
* model performance over time

---

### Model Versioning

Maintain different versions of:

```text id="j4s6k8"
Training Data
Feature Pipeline
Model
Hyperparameters
Evaluation Results
```

This improves reproducibility and makes production ML systems easier to maintain.

---

### Improved API Productionization

Future improvements could include:

* stronger request validation
* structured logging
* monitoring
* API authentication
* model versioning
* containerized deployment
* automated testing

---

# What I Learned

This project helped me understand the complete lifecycle of a machine-learning system:

```text id="n6w3p0"
Raw Financial Data
       ↓
Data Cleaning
       ↓
Feature Engineering
       ↓
Class Imbalance
       ↓
Baseline Model
       ↓
Model Comparison
       ↓
XGBoost
       ↓
Hyperparameter Optimization
       ↓
Evaluation
       ↓
Explainability
       ↓
Model Persistence
       ↓
FastAPI
       ↓
Real-Time Inference
```

The main learning was that building an ML project is not just about training a model. It also involves **data quality, feature design, appropriate evaluation, explainability and serving the model as a usable system.**

---

# Future Direction

The project can evolve from an offline credit-risk model into a more complete production ML system:

```text id="c9w2r4"
Current
Credit Risk Model
      ↓
Neural Network Experiment
      ↓
Probability Calibration
      ↓
Model Versioning
      ↓
Data / Model Drift Monitoring
      ↓
Automated Retraining
      ↓
Production ML Pipeline
```

The goal is to continuously improve the system while understanding the trade-offs between model performance, interpretability, reliability and business requirements.

---

## Disclaimer

This project is an educational machine-learning implementation and is not intended to make real-world lending decisions without appropriate validation, governance, regulatory review and human oversight.
