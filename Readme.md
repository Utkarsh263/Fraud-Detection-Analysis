
# 💳 Fraud Detection System

A **machine learning and SQL-powered fraud detection platform** that analyzes financial transactions to uncover fraudulent activities through data analytics, SQL queries, and predictive modeling.

---

## 📘 Overview

**Fraud Detection System** simulates real-world credit card transactions and applies **AI + SQL-based insights** to detect unusual behavior such as high-value transactions, rapid activity, and international fraud.  

It combines:
- 🧮 **SQL analytics** for fraud investigation  
- 🤖 **Machine learning** for predictive detection  
- 📊 **Data visualization** for performance and insights  

**Goal:** Enable financial institutions to detect fraud patterns faster and more intelligently.

---

## ⚡ Features

- **Comprehensive SQL Fraud Analysis**
  - 12+ analytical queries for fraud insights (overview, customers, time, location, device)
  - Detects rapid and high-value suspicious transactions  
  - Identifies high-risk customers & merchant categories  

- **Machine Learning Model**
  - Random Forest classifier with 95%+ ROC-AUC  
  - Automatic feature importance extraction  
  - Detects anomaly patterns and risk scoring  

- **Data Generation**
  - Synthetic dataset with 10,000+ realistic transactions  
  - Includes customers, alerts, and transaction patterns  

---

## 🧰 Tech Stack

| Component | Tools |
|------------|--------|
| **Language** | Python 3.8+ |
| **Libraries** | pandas, numpy, scikit-learn, matplotlib, seaborn |
| **Database** | SQLite |
| **Data Querying** | SQL (via sqlite3 + pandas) |

---

## ⚙️ Installation

### Prerequisites
- Python 3.8 or above  
- pip (Python package manager)  

### Setup
```bash
# Clone repository
git clone https://github.com/yourusername/fraud-detection-system.git
cd fraud-detection-system

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Step 1: Generate Synthetic Data
```bash
python generate_data.py
```
Creates `fraud_detection.db` with `transactions`, `customers`, and `fraud_alerts` tables.

### Step 2: Run SQL Queries
```bash
python run_queries.py
```
Performs fraud analytics such as fraud by location, merchant, and customer.

### Step 3: Train ML Model
```bash
python train_model.py
```
Trains a Random Forest classifier and outputs:
- `feature_importance.csv`  
- `test_predictions.csv`  
- `model_metrics.csv`  

---

## 📁 Project Structure
```
fraud_detection_system/
├── generate_data.py       # Synthetic data generation
├── run_queries.py         # SQL-based fraud analytics
├── train_model.py         # Machine learning pipeline
├── sql_queries.sql        # All fraud detection queries
├── fraud_detection.db     # SQLite database
├── README.md              # Documentation
└── requirements.txt       # Dependencies
```

---

## 📊 Insights

| Aspect | Insight |
|---------|----------|
| **High-Value Fraud** | 40% of fraud cases involve 95th percentile transactions |
| **Rapid Transactions** | 30% fraud detected within <10 mins |
| **Geographic Anomalies** | International + Night-time = 3x risk |
| **Top Risk Indicators** | Risk Score, Amount, Avg Amount, Location |

---

## 🔮 Future Enhancements

- Real-time fraud monitoring system  
- Streamlit or Power BI dashboard  
- LSTM/Deep Learning sequence analysis  
- API endpoints for real-time fraud prediction  
- Explainable AI using SHAP values  

---

## 🤝 Contributing

1. Fork the repository  
2. Create a feature branch:  
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes and push  
4. Create a pull request 🎉  

---

## 📝 License
This project is licensed under the **MIT License** – feel free to use and modify it.

---

**Built with ❤️ using Python, SQL, and Machine Learning**  
📅 *Version 1.0.0 — Last Updated: October 2025*
````

