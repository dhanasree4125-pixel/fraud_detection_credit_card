# Fraud Detection in Credit Card Transactions

## Objective

Identify fraudulent transactions using anomaly detection and supervised classification.

## Tools

Python, Scikit-learn, XGBoost, Pandas, Streamlit, Joblib

## What I Did

1. Loaded and explored the credit card transactions dataset, confirming extreme class imbalance (~0.17% fraud cases)
2. Scaled `Amount` and `Time` features using StandardScaler
3. Split data into train/test sets using stratified sampling to preserve fraud ratio
4. Applied unsupervised anomaly detection using Isolation Forest and Local Outlier Factor
5. Trained a supervised XGBoost classifier on the same data
6. Evaluated models using precision, recall, F1-score, and ROC-AUC (accuracy alone is misleading on imbalanced data)
7. Saved the trained model and built a Streamlit dashboard for uploading transaction CSVs and viewing live predictions

## Dataset

Credit Card Fraud Detection dataset (Kaggle, mlg-ulb) — ~284,807 transactions with PCA-anonymized features (V1–V28), Amount, Time, and Class (0 = normal, 1 = fraud)

## Results

- Isolation Forest — Precision / Recall / F1:
- Local Outlier Factor — Precision / Recall / F1:
- XGBoost — Precision / Recall / F1:
- XGBoost ROC-AUC:
- Fraud cases detected (dashboard test run):

## Observations

- Accuracy alone is misleading on this dataset due to the ~0.17% fraud rate — recall on the fraud class is the metric that actually matters.
- XGBoost (supervised) outperformed the unsupervised anomaly detection methods, since it can directly learn from labeled fraud examples.
- Isolation Forest and LOF are useful when labeled fraud data isn't available, but had higher false positive/negative rates in this comparison.

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook fraud_detection.ipynb
```

To launch the dashboard:

```bash
streamlit run app.py
```

## Project Structure

```
fraud-detection-credit-card/
├── README.md
├── requirements.txt
├── fraud_detection.ipynb   # Data prep, model training, evaluation
├── fraud_model.pkl         # Saved trained XGBoost model
└── app.py                  # Streamlit dashboard for live predictions
```
