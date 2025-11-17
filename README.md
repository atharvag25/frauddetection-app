# 🛡️ Fraud Detection System

A machine learning-powered fraud detection system built with Streamlit.

## Features
- Real-time fraud detection
- Automatic feature engineering
- User-friendly interface
- Probability scoring

## How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Dataset
Requires a dataset with the following columns:
- transaction_id, customer_id, merchant_id
- amount, transaction_date
- card_type, location, purchase_category
- customer_age, fraud_type, is_fraud
