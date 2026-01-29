from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "creditcard.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "processed_data.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "fraud_model.pkl"

RANDOM_STATE = 42
TEST_SIZE = 0.2
FRAUD_THRESHOLD = 0.35