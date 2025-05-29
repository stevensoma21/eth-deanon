# config.py
# === config.py ===
import os

# Path where all output files (plots, CSVs, etc.) will be saved
OUTPUT_DIR = "/content/drive/MyDrive/blockchain_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Default BigQuery config
BQ_DATASET = "eth_data"
BQ_TABLE = "eth_2024"
BQ_CREDS_PATH = "/content/drive/MyDrive/Colab Notebooks/crypto-argon-351622-5587062ee54e.json"

# Default query window
DEFAULT_START_DATE = "2024-01-01"
DEFAULT_END_DATE = "2024-01-31"
DEFAULT_TX_LIMIT = 500000
