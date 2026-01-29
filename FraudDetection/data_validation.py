import pandas as pd
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH
from data_loader import ensure_dataset_exists

def prepare_processed_data():
    ensure_dataset_exists()

    if PROCESSED_DATA_PATH.exists():
        print("✅ Processed data already exists")
        return

    print("⚙️ Preparing processed data...")

    df = pd.read_csv(RAW_DATA_PATH)
    df = df[df["Amount"] > 0]
    df.drop_duplicates(inplace=True)

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("✅ Processed data created")

if __name__ == "__main__":
    prepare_processed_data()
