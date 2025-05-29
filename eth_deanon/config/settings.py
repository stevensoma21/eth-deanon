"""
Configuration settings for the Ethereum deanonymization analysis
"""

# Data loading settings
START_DATE = "2024-01-01"
END_DATE = "2024-07-01"
TRANSACTION_LIMIT = 6000000

# Graph construction settings
MIN_DEGREE_FILTER = 2

# Clustering settings
DBSCAN_PARAMS = {
    'eps': 1.0,
    'min_samples': 5
}

HDBSCAN_PARAMS = {
    'min_cluster_size': 100,
    'min_samples': 20
}

# Temporal analysis settings
WINDOW_SIZE_DAYS = 14

# Cohort analysis settings
STABILITY_THRESHOLDS = {
    'high_stability': 0.3,
    'low_stability': 0.7,
    'min_windows': 2,
    'frequent_windows': 4
}

# Visualization settings
PLOT_SETTINGS = {
    'figsize': (12, 8),
    'dpi': 300,
    'style': 'seaborn'
}

# Logging settings
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'
