import pandas as pd
import numpy as np
from config import PROCESSED_DATA_PATH

def load_features():
    df = pd.read_csv(PROCESSED_DATA_PATH)

    df["amount_log"] = np.log1p(df["Amount"])
    df["is_high_amount"] = (df["Amount"] > df["Amount"].quantile(0.95)).astype(int)
    df["transaction_hour"] = (df["Time"] // 3600) % 24
    df["amount_to_mean_ratio"] = df["Amount"] / df["Amount"].mean()

    feature_cols = (
        ["Time", "Amount"]
        + [f"V{i}" for i in range(1, 29)]
        + [
            "amount_log",
            "is_high_amount",
            "transaction_hour",
            "amount_to_mean_ratio",
        ]
    )

    return df[feature_cols], df["Class"]
