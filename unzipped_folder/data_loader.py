# data_loader.py
# === data_loader.py ===
import os
from google.cloud import bigquery
import pandas as pd

def setup_google_cloud(creds_path):
    """
    Authenticate and return BigQuery client.
    """
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
    client = bigquery.Client()
    client.query("SELECT 1").to_dataframe()  # test query
    return client

def load_transaction_data(client, start_date, end_date, limit, dataset, table):
    """
    Load Ethereum transaction data from BigQuery.
    """
    query = f"""
    SELECT
      from_address,
      to_address,
      value,
      receipt_gas_used AS gas_used,
      gas_price,
      block_timestamp AS timestamp
    FROM `{client.project}.{dataset}.{table}`
    WHERE value > 1e15
      AND from_address IS NOT NULL
      AND to_address IS NOT NULL 
      AND from_address != to_address
      AND DATE(block_timestamp) BETWEEN '{start_date}' AND '{end_date}'
    LIMIT {limit}
    """

    df = client.query(query).to_dataframe()
    df['value_eth'] = df['value'].astype(float) / 1e18
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[df['from_address'] != df['to_address']]

    return df
# data_loader.py
