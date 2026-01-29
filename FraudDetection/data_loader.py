import requests
from config import RAW_DATA_PATH

DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

def ensure_dataset_exists():
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    if RAW_DATA_PATH.exists():
        print("✅ Dataset already exists")
        return

    print("📥 Downloading dataset...")

    response = requests.get(DATASET_URL, stream=True)
    response.raise_for_status()

    with open(RAW_DATA_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print("✅ Dataset downloaded successfully")
