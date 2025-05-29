"""
Data loading and preprocessing functions for Ethereum transaction analysis
"""

import logging
import pandas as pd
from google.cloud import bigquery
from ..config.settings import START_DATE, END_DATE, TRANSACTION_LIMIT

logger = logging.getLogger(__name__)

def setup_google_cloud(creds_path):
    """
    Setup Google Cloud credentials and return BigQuery client

    Args:
        creds_path: Path to the Google Cloud credentials JSON file

    Returns:
        BigQuery client
    """
    import os
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
    client = bigquery.Client()
    # Verify connection
    client.query("SELECT 1").to_dataframe()
    return client

def load_transaction_data(client, start_date=START_DATE, end_date=END_DATE, limit=TRANSACTION_LIMIT):
    """
    Load Ethereum transaction data from BigQuery

    Args:
        client: BigQuery client
        start_date: Start date for data range
        end_date: End date for data range
        limit: Maximum number of transactions to load

    Returns:
        DataFrame with transaction data
    """
    query = f'''
    WITH sanctioned AS (
        SELECT *
        FROM (
            SELECT *
            FROM `crypto-argon-351622.eth_data.eth_with_ofac_flag`
            WHERE value > 1e15
                AND from_address IS NOT NULL
                AND to_address IS NOT NULL
                AND to_address != from_address
                AND ofac_sanction_flag = TRUE
                AND DATE(timestamp) BETWEEN '{start_date}' AND '{end_date}'
        )
        ORDER BY RAND()
        LIMIT {int(limit * 1)}
    ),
    unsanctioned AS (
        SELECT *
        FROM (
            SELECT *
            FROM `crypto-argon-351622.eth_data.eth_with_ofac_flag`
            WHERE value > 1e15
                AND from_address IS NOT NULL
                AND to_address IS NOT NULL
                AND to_address != from_address
                AND (ofac_sanction_flag IS NULL OR ofac_sanction_flag = FALSE)
                AND DATE(timestamp) BETWEEN '{start_date}' AND '{end_date}'
        )
        ORDER BY RAND()
        LIMIT {int(limit * .5)}
    )

    SELECT
        from_address,
        to_address,
        value,
        gas_used,
        gas_price,
        timestamp AS timestamp,
        ofac_sanction_flag
    FROM sanctioned

    UNION ALL

    SELECT
        from_address,
        to_address,
        value,
        gas_used AS gas_used,
        gas_price,
        timestamp AS timestamp,
        ofac_sanction_flag
    FROM unsanctioned;
    '''

    df = client.query(query).to_dataframe()
    df['value_eth'] = df['value'].astype(float) / 1e18
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[df['from_address'] != df['to_address']]

    logger.info(f"Dataset loaded: {len(df)} transactions")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    logger.info(f"Unique addresses: {len(set(df['from_address'].tolist() + df['to_address'].tolist()))}")
    
    return df

def normalize_addresses(df, address_cols):
    """
    Consistently normalize all addresses

    Args:
        df: DataFrame containing address columns
        address_cols: List of column names containing addresses

    Returns:
        DataFrame with normalized addresses
    """
    for col in address_cols:
        df[col] = df[col].astype(str).str.lower().str.strip()
    return df
