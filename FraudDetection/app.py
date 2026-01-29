import streamlit as st
import pandas as pd
import numpy as np
import joblib

from config import MODEL_PATH, FRAUD_THRESHOLD
from data_loader import ensure_dataset_exists
from data_validation import prepare_processed_data
from train import main as train_model

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🚨",
    layout="centered"
)

st.title("🚨 Fraud Detection System")
st.write("Live ML-based fraud detection demo")

# -------------------------
# Ensure data & model exist
# -------------------------
ensure_dataset_exists()
prepare_processed_data()

if not MODEL_PATH.exists():
    st.info("Training model for the first time. This may take a minute...")
    train_model()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()


# -------------------------
# User Inputs
# -------------------------
st.subheader("Transaction Details")

amount = st.number_input("Transaction Amount", min_value=0.01, value=500.0)
time = st.number_input("Transaction Time (seconds)", min_value=0, value=100000)

# -------------------------
# Build input features
# -------------------------
data = {
    "Time": time,
    "Amount": amount,
}

for i in range(1, 29):
    data[f"V{i}"] = 0.0

data["amount_log"] = np.log1p(amount)
data["is_high_amount"] = int(amount > 1000)
data["transaction_hour"] = (time // 3600) % 24
data["amount_to_mean_ratio"] = amount / 88.0

input_df = pd.DataFrame([data])
input_df = input_df[model.feature_names_in_]

# -------------------------
# Prediction
# -------------------------
if st.button("Predict Fraud"):
    prob = model.predict_proba(input_df)[0][1]

    st.markdown("---")
    st.write(f"**Fraud Probability:** `{prob:.4f}`")

    if prob >= FRAUD_THRESHOLD:
        st.error("🚨 Fraudulent Transaction Detected")
    else:
        st.success("✅ Legitimate Transaction")