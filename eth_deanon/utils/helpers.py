"""
Utility functions for Ethereum transaction analysis
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def setup_logging(log_level='INFO'):
    """
    Set up logging configuration

    Args:
        log_level: Logging level (default: 'INFO')
    """
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def format_timestamp(timestamp):
    """
    Format timestamp to datetime string

    Args:
        timestamp: Unix timestamp

    Returns:
        Formatted datetime string
    """
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

def calculate_time_delta(timestamps):
    """
    Calculate time differences between consecutive timestamps

    Args:
        timestamps: List of timestamps

    Returns:
        List of time differences in seconds
    """
    return np.diff(timestamps)

def normalize_address(address):
    """
    Normalize Ethereum address format

    Args:
        address: Ethereum address string

    Returns:
        Normalized address string
    """
    return address.lower()

def calculate_percentiles(data, percentiles=[25, 50, 75]):
    """
    Calculate percentiles for a dataset

    Args:
        data: List or array of numerical data
        percentiles: List of percentiles to calculate

    Returns:
        Dictionary mapping percentiles to values
    """
    return {p: np.percentile(data, p) for p in percentiles}

def filter_outliers(data, lower_percentile=1, upper_percentile=99):
    """
    Filter outliers from a dataset using percentiles

    Args:
        data: List or array of numerical data
        lower_percentile: Lower percentile threshold
        upper_percentile: Upper percentile threshold

    Returns:
        Filtered data array
    """
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)
    return data[(data >= lower_bound) & (data <= upper_bound)]

def create_time_windows(start_date, end_date, window_size_days):
    """
    Create time windows for temporal analysis

    Args:
        start_date: Start date as datetime
        end_date: End date as datetime
        window_size_days: Size of each window in days

    Returns:
        List of (window_start, window_end) tuples
    """
    windows = []
    current_start = start_date
    
    while current_start < end_date:
        current_end = min(current_start + timedelta(days=window_size_days), end_date)
        windows.append((current_start, current_end))
        current_start = current_end
    
    return windows

def calculate_rolling_metrics(df, window_size, metrics=['mean', 'std']):
    """
    Calculate rolling metrics for a DataFrame

    Args:
        df: DataFrame with time series data
        window_size: Size of rolling window
        metrics: List of metrics to calculate

    Returns:
        DataFrame with rolling metrics
    """
    result = pd.DataFrame()
    
    for metric in metrics:
        if metric == 'mean':
            result[f'rolling_{metric}'] = df.rolling(window_size).mean()
        elif metric == 'std':
            result[f'rolling_{metric}'] = df.rolling(window_size).std()
        elif metric == 'median':
            result[f'rolling_{metric}'] = df.rolling(window_size).median()
    
    return result

def save_results(results, filename):
    """
    Save analysis results to a file

    Args:
        results: Dictionary or DataFrame containing results
        filename: Output filename
    """
    if isinstance(results, pd.DataFrame):
        results.to_csv(filename, index=False)
    else:
        pd.DataFrame(results).to_csv(filename, index=False)
    
    logger.info(f"Results saved to {filename}")

def load_results(filename):
    """
    Load analysis results from a file

    Args:
        filename: Input filename

    Returns:
        DataFrame containing results
    """
    return pd.read_csv(filename)

def format_currency(value):
    """
    Format currency value with appropriate units

    Args:
        value: Numerical value

    Returns:
        Formatted string with units
    """
    if value >= 1e9:
        return f"${value/1e9:.2f}B"
    elif value >= 1e6:
        return f"${value/1e6:.2f}M"
    elif value >= 1e3:
        return f"${value/1e3:.2f}K"
    else:
        return f"${value:.2f}"

def calculate_growth_rate(initial_value, final_value, time_period):
    """
    Calculate compound annual growth rate (CAGR)

    Args:
        initial_value: Initial value
        final_value: Final value
        time_period: Time period in years

    Returns:
        CAGR as a percentage
    """
    if initial_value <= 0 or time_period <= 0:
        return 0
    
    return ((final_value / initial_value) ** (1 / time_period) - 1) * 100
