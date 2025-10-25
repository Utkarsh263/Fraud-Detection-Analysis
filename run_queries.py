import sqlite3
import pandas as pd

# Connect to database
conn = sqlite3.connect('fraud_detection.db')

# =====================================================
# Define all fraud detection analytical queries
# =====================================================
queries = {

    # 1️⃣ FRAUD OVERVIEW DASHBOARD
    "fraud_overview": """
        SELECT 
            COUNT(*) as total_transactions,
            SUM(is_fraud) as fraud_count,
            ROUND(AVG(is_fraud) * 100, 2) as fraud_rate_pct,
            ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) as fraud_amount,
            ROUND(AVG(amount), 2) as avg_transaction_amount,
            ROUND(AVG(CASE WHEN is_fraud = 1 THEN amount END), 2) as avg_fraud_amount
        FROM transactions;
    """,

    # 2️⃣ FRAUD BY MERCHANT CATEGORY
    "fraud_by_merchant_category": """
        SELECT 
            merchant_category,
            COUNT(*) as total_transactions,
            SUM(is_fraud) as fraud_count,
            ROUND(AVG(is_fraud) * 100, 2) as fraud_rate_pct,
            ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) as total_fraud_amount
        FROM transactions
        GROUP BY merchant_category
        ORDER BY fraud_rate_pct DESC;
    """,

    # 3️⃣ FRAUD BY LOCATION
    "fraud_by_location": """
        SELECT 
            location,
            COUNT(*) as total_transactions,
            SUM(is_fraud) as fraud_count,
            ROUND(AVG(is_fraud) * 100, 2) as fraud_rate_pct,
            ROUND(AVG(amount), 2) as avg_amount
        FROM transactions
        GROUP BY location
        ORDER BY fraud_rate_pct DESC;
    """,

    # 4️⃣ TIME-BASED FRAUD ANALYSIS
    "time_based_fraud_analysis": """
        SELECT 
            hour,
            COUNT(*) as total_transactions,
            SUM(is_fraud) as fraud_count,
            ROUND(AVG(is_fraud) * 100, 2) as fraud_rate_pct
        FROM transactions
        GROUP BY hour
        ORDER BY hour;
    """,

    # 5️⃣ HIGH-RISK CUSTOMERS
    "high_risk_customers": """
        SELECT 
            t.customer_id,
            c.customer_name,
            COUNT(*) as total_transactions,
            SUM(t.is_fraud) as fraud_count,
            ROUND(SUM(t.amount), 2) as total_spent,
            ROUND(SUM(CASE WHEN t.is_fraud = 1 THEN t.amount ELSE 0 END), 2) as fraud_amount,
            ROUND(AVG(t.risk_score), 2) as avg_risk_score
        FROM transactions t
        JOIN customers c ON t.customer_id = c.customer_id
        GROUP BY t.customer_id, c.customer_name
        HAVING fraud_count >= 2
        ORDER BY fraud_count DESC, fraud_amount DESC;
    """,

    # 6️⃣ DAILY FRAUD TREND
    "daily_fraud_trend": """
        SELECT 
            DATE(transaction_date) as transaction_day,
            COUNT(*) as total_transactions,
            SUM(is_fraud) as fraud_count,
            ROUND(AVG(is_fraud) * 100, 2) as fraud_rate_pct,
            ROUND(SUM(amount), 2) as total_amount
        FROM transactions
        GROUP BY DATE(transaction_date)
        ORDER BY transaction_day;
    """,

    # 7️⃣ HIGH-VALUE SUSPICIOUS TRANSACTIONS
    "high_value_suspicious_transactions": """
        SELECT 
            t.transaction_id,
            t.customer_id,
            c.customer_name,
            t.transaction_date,
            t.amount,
            t.merchant_category,
            t.location,
            t.risk_score,
            t.is_fraud
        FROM transactions t
        JOIN customers c ON t.customer_id = c.customer_id
        WHERE t.amount > 300 
          AND t.risk_score > 50
        ORDER BY t.risk_score DESC, t.amount DESC
        LIMIT 50;
    """,

    # 8️⃣ RAPID TRANSACTION DETECTION
    "rapid_transaction_detection": """
        WITH transaction_intervals AS (
            SELECT 
                customer_id,
                transaction_id,
                transaction_date,
                amount,
                is_fraud,
                LAG(transaction_date) OVER (PARTITION BY customer_id ORDER BY transaction_date) as prev_transaction_date
            FROM transactions
        )
        SELECT 
            customer_id,
            transaction_id,
            transaction_date,
            amount,
            is_fraud,
            ROUND((JULIANDAY(transaction_date) - JULIANDAY(prev_transaction_date)) * 1440, 2) as minutes_since_last
        FROM transaction_intervals
        WHERE prev_transaction_date IS NOT NULL
          AND (JULIANDAY(transaction_date) - JULIANDAY(prev_transaction_date)) * 1440 < 30
        ORDER BY customer_id, transaction_date;
    """,

    # 9️⃣ FRAUD ALERT STATUS SUMMARY
    "fraud_alert_status_summary": """
        SELECT 
            status,
            COUNT(*) as alert_count,
            ROUND(SUM(amount), 2) as total_amount,
            reviewed_by,
            COUNT(DISTINCT customer_id) as affected_customers
        FROM fraud_alerts
        GROUP BY status, reviewed_by
        ORDER BY alert_count DESC;
    """,

    # 🔟 WEEKEND VS WEEKDAY FRAUD
    "weekend_vs_weekday_fraud": """
        SELECT 
            CASE WHEN is_weekend = 1 THEN 'Weekend' ELSE 'Weekday' END as day_type,
            COUNT(*) as total_transactions,
            SUM(is_fraud) as fraud_count,
            ROUND(AVG(is_fraud) * 100, 2) as fraud_rate_pct,
            ROUND(AVG(amount), 2) as avg_amount
        FROM transactions
        GROUP BY is_weekend;
    """,

    # 11️⃣ DEVICE TYPE FRAUD ANALYSIS
    "device_type_fraud_analysis": """
        SELECT 
            device_type,
            COUNT(*) as total_transactions,
            SUM(is_fraud) as fraud_count,
            ROUND(AVG(is_fraud) * 100, 2) as fraud_rate_pct,
            ROUND(AVG(amount), 2) as avg_amount
        FROM transactions
        GROUP BY device_type
        ORDER BY fraud_rate_pct DESC;
    """,

    # 12️⃣ CUSTOMER RISK PROFILE
    "customer_risk_profile": """
        SELECT 
            t.customer_id,
            c.customer_name,
            COUNT(*) as transaction_count,
            ROUND(AVG(t.amount), 2) as avg_transaction,
            ROUND(MAX(t.amount), 2) as max_transaction,
            SUM(CASE WHEN t.location = 'International' THEN 1 ELSE 0 END) as intl_transactions,
            SUM(t.is_fraud) as fraud_count,
            ROUND(AVG(t.risk_score), 2) as avg_risk_score,
            CASE 
                WHEN AVG(t.risk_score) > 60 THEN 'High Risk'
                WHEN AVG(t.risk_score) > 30 THEN 'Medium Risk'
                ELSE 'Low Risk'
            END as risk_category
        FROM transactions t
        JOIN customers c ON t.customer_id = c.customer_id
        GROUP BY t.customer_id, c.customer_name
        ORDER BY avg_risk_score DESC;
    """
}

# =====================================================
# Execute each query and store results
# =====================================================
results = {}

for name, q in queries.items():
    try:
        df = pd.read_sql_query(q, conn)
        results[name] = df
        print(f"\n=== {name.upper()} ===")
        print(df.head(), "\n")
    except Exception as e:
        print(f"\n[ERROR] Query '{name}' failed: {e}\n")

# Close connection
conn.close()
