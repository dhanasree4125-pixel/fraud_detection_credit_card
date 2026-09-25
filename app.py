import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

st.title("Credit Card Fraud Detector")

model = joblib.load('fraud_model.pkl')

uploaded = st.file_uploader("Upload transaction CSV", type="csv")
if uploaded:
    data = pd.read_csv(uploaded)

    # Drop Class if present (it's the label, not a feature)
    if 'Class' in data.columns:
        data = data.drop('Class', axis=1)

    # Apply the same scaling used during training
    scaler = StandardScaler()
    data['Amount_scaled'] = scaler.fit_transform(data[['Amount']])
    data['Time_scaled'] = scaler.fit_transform(data[['Time']])
    data = data.drop(['Amount', 'Time'], axis=1)

    preds = model.predict(data)
    st.write("Predictions (0 = Normal, 1 = Fraud):")
    st.write(preds)