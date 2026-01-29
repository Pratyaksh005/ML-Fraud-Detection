import pandas as pd
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH
from data_loader import ensure_dataset_exists

def main():
    ensure_dataset_exists()

    df = pd.read_csv(RAW_DATA_PATH)
    df = df[df["Amount"] > 0]
    df.drop_duplicates(inplace=True)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("✅ Data validated & saved")

if __name__ == "__main__":
    main()
