import joblib
import numpy as np
from config import MODEL_PATH, FRAUD_THRESHOLD

def predict_fraud(input_df):
    model = joblib.load(MODEL_PATH)
    prob = model.predict_proba(input_df)[:, 1]
    pred = (prob >= FRAUD_THRESHOLD).astype(int)
    return pred, prob