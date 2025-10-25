import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from database
print("Loading data from database...")
conn = sqlite3.connect('fraud_detection.db')
df = pd.read_sql_query("SELECT * FROM transactions", conn)
conn.close()

# Convert transaction_date to datetime
df['transaction_date'] = pd.to_datetime(df['transaction_date'])

print(f"Loaded {len(df)} transactions")
print(f"Fraud rate: {df['is_fraud'].mean()*100:.2f}%")

# ==========================================
# FEATURE ENGINEERING
# ==========================================

print("\nEngineering features...")

# Encode categorical variables
le_merchant = LabelEncoder()
le_location = LabelEncoder()
le_device = LabelEncoder()

df['merchant_category_encoded'] = le_merchant.fit_transform(df['merchant_category'])
df['location_encoded'] = le_location.fit_transform(df['location'])
df['device_type_encoded'] = le_device.fit_transform(df['device_type'])

# Create additional time-based features
df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)

# Amount-based features
df['amount_zscore'] = (df['amount'] - df['avg_amount']) / (df['std_amount'] + 1e-5)
df['is_high_amount'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)

# Select features for modeling
feature_columns = [
    'amount', 'hour', 'day_of_week', 'is_weekend', 'is_night', 'is_business_hours',
    'merchant_category_encoded', 'location_encoded', 'device_type_encoded',
    'avg_amount', 'std_amount', 'transaction_count', 'amount_zscore', 
    'is_high_amount', 'risk_score'
]

X = df[feature_columns]
y = df['is_fraud']

print(f"Using {len(feature_columns)} features")

# ==========================================
# TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")
print(f"Fraud in training: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
print(f"Fraud in test: {y_test.sum()} ({y_test.mean()*100:.2f}%)")

# ==========================================
# SCALE FEATURES
# ==========================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# TRAIN RANDOM FOREST MODEL
# ==========================================

print("\nTraining Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test_scaled)
y_pred_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

# ==========================================
# MODEL EVALUATION
# ==========================================

print("\n" + "="*50)
print("RANDOM FOREST MODEL PERFORMANCE")
print("="*50)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf, target_names=['Legitimate', 'Fraud']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_rf)
print(cm)
print(f"\nTrue Negatives: {cm[0,0]}")
print(f"False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}")
print(f"True Positives: {cm[1,1]}")

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_pred_proba_rf)
print(f"\nROC-AUC Score: {roc_auc:.4f}")

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

print("\n" + "="*50)
print("FEATURE IMPORTANCE")
print("="*50)

feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# ==========================================
# SAVE PREDICTIONS TO DATABASE
# ==========================================

print("\nSaving predictions to database...")

# Add predictions to test data
df_test = df.loc[X_test.index].copy()
df_test['fraud_prediction'] = y_pred_rf
df_test['fraud_probability'] = y_pred_proba_rf
df_test['prediction_correct'] = (df_test['is_fraud'] == df_test['fraud_prediction']).astype(int)

# Save to database
conn = sqlite3.connect('fraud_detection.db')
df_test[['transaction_id', 'fraud_prediction', 'fraud_probability', 
         'prediction_correct']].to_sql('ml_predictions', conn, if_exists='replace', index=False)

# Create summary statistics table
summary_stats = pd.DataFrame({
    'metric': ['Total Transactions', 'Fraudulent Transactions', 'Fraud Rate %', 
               'Model Accuracy', 'ROC-AUC Score', 'True Positives', 'False Positives',
               'True Negatives', 'False Negatives'],
    'value': [
        len(df),
        df['is_fraud'].sum(),
        round(df['is_fraud'].mean() * 100, 2),
        round((y_pred_rf == y_test).mean() * 100, 2),
        round(roc_auc, 4),
        cm[1,1],
        cm[0,1],
        cm[0,0],
        cm[1,0]
    ]
})
summary_stats.to_sql('model_metrics', conn, if_exists='replace', index=False)

conn.close()

# ==========================================
# EXPORT FOR POWER BI
# ==========================================

print("\nExporting data for Power BI...")

# Export test predictions
df_test.to_csv('test_predictions.csv', index=False)

# Export feature importance
feature_importance.to_csv('feature_importance.csv', index=False)

# Export model metrics
summary_stats.to_csv('model_metrics.csv', index=False)

print("\n" + "="*50)
print("ANALYSIS COMPLETE!")
print("="*50)
print("\nFiles exported for Power BI:")
print("  - test_predictions.csv")
print("  - feature_importance.csv")
print("  - model_metrics.csv")
print("\nDatabase tables created:")
print("  - ml_predictions")
print("  - model_metrics")