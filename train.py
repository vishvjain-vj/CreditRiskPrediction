import pandas as pd
import optuna
import numpy as np
import xgboost as xgb
import shap
import gc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score

# Data only application_train.csv and bureau.csv are used in this script. Ensure they are present in the 'data/' directory.


# 1. Load the data
print("Loading data...")
df = pd.read_csv('data/application_train.csv')
# ==========================================
# 1.5 INTEGRATE BUREAU HISTORY (Alternative Data)
# ==========================================
print("Loading and aggregating bureau history...")
bureau = pd.read_csv('data/bureau.csv')

# Compress multiple historical rows into one summary row per borrower
bureau_agg = bureau.groupby('SK_ID_CURR').agg({
    'SK_ID_BUREAU': 'count',           # Total number of past loans
    'AMT_CREDIT_SUM': ['sum', 'mean'], # Total and average historical credit
    'AMT_CREDIT_SUM_DEBT': 'sum',      # Total historical unpaid debt
    'CREDIT_DAY_OVERDUE': 'max'        # Worst historical late payment
})

# Flatten the multi-index columns
bureau_agg.columns = ['BUREAU_LOAN_COUNT', 'BUREAU_DEBT_SUM', 'BUREAU_DEBT_MEAN', 'BUREAU_UNPAID_DEBT', 'BUREAU_MAX_OVERDUE']
bureau_agg = bureau_agg.reset_index()

# Merge safely using a left join
print("Merging bureau data with main application...")
df = df.merge(bureau_agg, on='SK_ID_CURR', how='left')

# Delete variables and force garbage collection to save RAM
del bureau, bureau_agg
gc.collect()




# ==========================================
# 2. FEATURE ENGINEERING (The New Additions)
# ==========================================
print("Engineering financial features...")

# Fix the known anomaly in DAYS_EMPLOYED (365243 means unemployed/error)
df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)

new_features = pd.DataFrame({
    'CREDIT_INCOME_RATIO': df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL'],
    'ANNUITY_INCOME_RATIO': df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL'],
    'CREDIT_TERM': df['AMT_CREDIT'] / df['AMT_ANNUITY'],
    'DAYS_EMPLOYED_PERC': df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
})

# Calculate the flag based on the newly created column
new_features['HIGH_LEVERAGE_FLAG'] = (new_features['CREDIT_INCOME_RATIO'] > 10).astype(int)

# Join them all to the main DataFrame at once
df = pd.concat([df, new_features], axis=1)

# 3. Separate Features (X) and Target (y)
X = df.drop(columns=['TARGET', 'SK_ID_CURR']) 
y = df['TARGET']

# 4. Handle Categorical Columns
print("Encoding categorical features...")
categorical_cols = X.select_dtypes(include=['object', 'str', 'category']).columns

le = LabelEncoder()
for col in categorical_cols:
    X[col] = X[col].astype(str)
    X[col] = le.fit_transform(X[col])

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 6. Calculate the imbalance weight dynamically
scale_weight = (y_train == 0).sum() / (y_train == 1).sum()

# # 7. Initialize and Train XGBoost
# print("Training XGBoost model...")
# model = xgb.XGBClassifier(
#     scale_pos_weight=scale_weight,
#     eval_metric='aucpr', 
#     random_state=42,
#     tree_method='hist' 
# )
# model.fit(X_train, y_train)

# 7. Bayesian Hyperparameter Tuning with Optuna
print("Setting up Bayesian Optimization...")

def objective(trial):
    # 1. Suggest smart parameters to test
    param = {
        'scale_pos_weight': scale_weight,
        'eval_metric': 'aucpr',
        'random_state': 42,
        'tree_method': 'hist',
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 100, 300)
    }
    
    # 2. Create a quick validation split inside the training data to test the parameters
    X_tune_train, X_tune_val, y_tune_train, y_tune_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # 3. Train and score the temporary model
    temp_model = xgb.XGBClassifier(**param)
    temp_model.fit(X_tune_train, y_tune_train)
    
    preds = temp_model.predict_proba(X_tune_val)[:, 1]
    return average_precision_score(y_tune_val, preds)

# Run 10 smart Bayesian iterations (n_trials)
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10) 

print(f"\nBest Bayesian Parameters Found: {study.best_params}")

# Rebuild the final parameters dictionary
best_params = study.best_params
best_params['scale_pos_weight'] = scale_weight
best_params['eval_metric'] = 'aucpr'
best_params['random_state'] = 42
best_params['tree_method'] = 'hist'

# Train the absolute best model on the FULL training set
print("Training final model with best Bayesian parameters...")
model = xgb.XGBClassifier(**best_params)
model.fit(X_train, y_train)


# 8. Evaluate Performance
print("Generating predictions...")
y_pred = model.predict(X_test)         
y_prob = model.predict_proba(X_test)[:, 1] 

print("\n--- Model Evaluation ---")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

pr_auc = average_precision_score(y_test, y_prob)
print(f"PR-AUC Score: {pr_auc:.4f}")

# ==========================================
# 9. THRESHOLD OPTIMIZATION
# ==========================================
print("\n--- Custom Threshold Testing ---")

# Test a range of probability cutoffs
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

print(f"{'Threshold':<12} | {'Recall (Caught %)':<18} | {'Precision':<10} | {'False Positives (Lost Deals)'}")
print("-" * 75)

for thresh in thresholds:
    # Manually create predictions based on the custom threshold
    custom_preds = (y_prob >= thresh).astype(int)
    
    # Extract confusion matrix metrics
    tn, fp, fn, tp = confusion_matrix(y_test, custom_preds).ravel()
    
    # Calculate custom metrics
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    
    print(f"{thresh:<12.2f} | {recall:<18.2f} | {precision:<10.2f} | {fp}")

    # ==========================================
# 10. EXPLAINABLE AI (SHAP)
# ==========================================
print("\n--- SHAP Explainability ---")
print("Initializing SHAP TreeExplainer...")

# 1. Initialize the Explainer
explainer = shap.TreeExplainer(model)

# 2. Calculate SHAP values (Using a 1,000 row sample to save time)
X_test_sample = X_test.sample(1000, random_state=42)
shap_values = explainer.shap_values(X_test_sample)

# 3. Save a Summary Plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_test_sample, show=False)
plt.savefig('shap_summary.png', bbox_inches='tight')
print("Saved portfolio risk summary plot to 'shap_summary.png'")

# 4. Explain a Single Prediction (Regulatory Compliance)
# Find the first customer in the test set that we aggressively rejected
rejected_indices = np.where(y_prob > 0.70)[0] 

if len(rejected_indices) > 0:
    idx = rejected_indices[0] # Grab the first high-risk customer
    customer_data = X_test.iloc[[idx]]
    
    print(f"\nExplaining Customer at index {idx}:")
    print(f"Predicted Probability of Default: {y_prob[idx]:.2f}")
    
    # Get SHAP values for this specific customer
    customer_shap = explainer.shap_values(customer_data)
    
    # Map features to their numerical impact
    feature_impacts = pd.DataFrame({
        'Feature': X_test.columns,
        'Value': customer_data.iloc[0].values,
        'Impact': customer_shap[0]
    })
    
    # Sort by absolute impact to find the biggest drivers (positive or negative)
    feature_impacts['Abs_Impact'] = feature_impacts['Impact'].abs()
    feature_impacts = feature_impacts.sort_values(by='Abs_Impact', ascending=False)
    
    print("\nTop 5 Risk Drivers for this Borrower:")
    print(feature_impacts[['Feature', 'Value', 'Impact']].head(5).to_string(index=False))

import joblib
joblib.dump(model, 'xgboost_credit_model.pkl')
print("Model saved to xgboost_credit_model.pkl")

import json
test_json = X_test.iloc[[idx]].to_dict(orient='records')[0]
print(json.dumps({"features": test_json}))