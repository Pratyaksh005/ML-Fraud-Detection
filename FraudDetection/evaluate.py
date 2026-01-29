import joblib
import numpy as np
from feature_engineering import create_features

def evaluate_stability():
    model = joblib.load("models/fraud_model.pkl")
    df = create_features()

    sample = df.sample(1000, random_state=42)
    X = sample.drop("Class", axis=1)

    preds1 = model.predict_proba(X)[:, 1]
    preds2 = model.predict_proba(X)[:, 1]

    diff = np.mean(np.abs(preds1 - preds2))
    print(f"Mean prediction difference: {diff:.6f}")

if __name__ == "__main__":
    evaluate_stability()