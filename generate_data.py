import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import sqlite3

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Generate synthetic transaction data
def generate_transactions(n_transactions=10000):
    """Generate synthetic credit card transactions with fraud cases"""
    
    # Time range: last 90 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    data = {
        'transaction_id': range(1, n_transactions + 1),
        'customer_id': np.random.randint(1000, 5000, n_transactions),
        'transaction_date': [start_date + timedelta(seconds=random.randint(0, 90*24*3600)) 
                            for _ in range(n_transactions)],
        'amount': np.random.exponential(50, n_transactions),
        'merchant_category': np.random.choice(['Retail', 'Online', 'Restaurant', 
                                              'Gas Station', 'Travel', 'Grocery'], 
                                             n_transactions, p=[0.25, 0.20, 0.15, 0.15, 0.15, 0.10]),
        'location': np.random.choice(['Domestic', 'International'], n_transactions, p=[0.85, 0.15]),
        'device_type': np.random.choice(['POS', 'Online', 'Mobile'], n_transactions, p=[0.5, 0.3, 0.2])
    }
    
    df = pd.DataFrame(data)
    
    # Add derived features
    df['hour'] = df['transaction_date'].dt.hour
    df['day_of_week'] = df['transaction_date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Calculate customer statistics
    customer_stats = df.groupby('customer_id').agg({
        'amount': ['mean', 'std', 'count']
    }).reset_index()
    customer_stats.columns = ['customer_id', 'avg_amount', 'std_amount', 'transaction_count']
    
    df = df.merge(customer_stats, on='customer_id', how='left')
    
    # Initialize fraud flag
    df['is_fraud'] = 0
    
    # Generate fraud patterns (3-5% fraud rate)
    n_frauds = int(n_transactions * 0.04)
    
    # Pattern 1: High amount transactions (40% of frauds)
    high_amount_frauds = int(n_frauds * 0.4)
    high_amount_idx = df.nlargest(high_amount_frauds * 3, 'amount').sample(high_amount_frauds).index
    df.loc[high_amount_idx, 'is_fraud'] = 1
    
    # Pattern 2: Multiple transactions in short time (30% of frauds)
    time_based_frauds = int(n_frauds * 0.3)
    df_sorted = df.sort_values(['customer_id', 'transaction_date'])
    df_sorted['time_diff'] = df_sorted.groupby('customer_id')['transaction_date'].diff().dt.total_seconds() / 60
    
    rapid_tx = df_sorted[(df_sorted['time_diff'] < 10) & (df_sorted['time_diff'].notna())]
    if len(rapid_tx) > 0:
        rapid_fraud_idx = rapid_tx.sample(min(time_based_frauds, len(rapid_tx))).index
        df.loc[rapid_fraud_idx, 'is_fraud'] = 1
    
    # Pattern 3: International transactions at odd hours (20% of frauds)
    intl_frauds = int(n_frauds * 0.2)
    odd_hour_intl = df[(df['location'] == 'International') & 
                       ((df['hour'] < 6) | (df['hour'] > 23))]
    if len(odd_hour_intl) > 0:
        intl_fraud_idx = odd_hour_intl.sample(min(intl_frauds, len(odd_hour_intl))).index
        df.loc[intl_fraud_idx, 'is_fraud'] = 1
    
    # Pattern 4: Unusual merchant for customer (10% of frauds)
    remaining_frauds = n_frauds - df['is_fraud'].sum()
    if remaining_frauds > 0:
        remaining_idx = df[df['is_fraud'] == 0].sample(remaining_frauds).index
        df.loc[remaining_idx, 'is_fraud'] = 1
    
    # Add fraud risk score (feature for analysis)
    df['risk_score'] = (
        (df['amount'] > df['avg_amount'] + 2 * df['std_amount']).astype(int) * 30 +
        (df['location'] == 'International').astype(int) * 20 +
        ((df['hour'] < 6) | (df['hour'] > 23)).astype(int) * 15 +
        (df['amount'] > 500).astype(int) * 25 +
        (df['is_weekend'] == 1).astype(int) * 10
    )
    
    return df

# Generate data
print("Generating transaction data...")
df_transactions = generate_transactions(10000)

# Save to SQLite database
print("Saving to database...")
conn = sqlite3.connect('fraud_detection.db')

df_transactions.to_sql('transactions', conn, if_exists='replace', index=False)

# Create additional tables for the system

# Customer table
customers = df_transactions[['customer_id']].drop_duplicates()
customers['customer_name'] = ['Customer_' + str(i) for i in customers['customer_id']]
customers['email'] = [f'customer{i}@email.com' for i in customers['customer_id']]
customers['signup_date'] = [datetime.now() - timedelta(days=random.randint(180, 1095)) 
                            for _ in range(len(customers))]
customers.to_sql('customers', conn, if_exists='replace', index=False)

# Fraud alerts table (for flagged transactions)
fraud_alerts = df_transactions[df_transactions['is_fraud'] == 1].copy()
fraud_alerts['alert_date'] = fraud_alerts['transaction_date']
fraud_alerts['status'] = np.random.choice(['Pending', 'Confirmed', 'False Positive'], 
                                          len(fraud_alerts), p=[0.3, 0.5, 0.2])
fraud_alerts['reviewed_by'] = np.random.choice(['System', 'Analyst_1', 'Analyst_2'], 
                                               len(fraud_alerts))
fraud_alerts[['transaction_id', 'customer_id', 'amount', 'alert_date', 
              'status', 'reviewed_by']].to_sql('fraud_alerts', conn, if_exists='replace', index=False)

conn.close()

# Export to CSV for Excel analysis
print("Exporting to CSV...")
df_transactions.to_csv('transactions.csv', index=False)
customers.to_csv('customers.csv', index=False)
fraud_alerts.to_csv('fraud_alerts.csv', index=False)

# Print summary statistics
print("\n" + "="*50)
print("DATA GENERATION COMPLETE")
print("="*50)
print(f"\nTotal Transactions: {len(df_transactions):,}")
print(f"Fraudulent Transactions: {df_transactions['is_fraud'].sum():,}")
print(f"Fraud Rate: {df_transactions['is_fraud'].mean()*100:.2f}%")
print(f"\nTotal Customers: {df_transactions['customer_id'].nunique():,}")
print(f"Date Range: {df_transactions['transaction_date'].min()} to {df_transactions['transaction_date'].max()}")
print("\nFraud by Category:")
print(df_transactions[df_transactions['is_fraud']==1]['merchant_category'].value_counts())
print("\nFiles created:")
print("  - fraud_detection.db (SQLite database)")
print("  - transactions.csv")
print("  - customers.csv")
print("  - fraud_alerts.csv")