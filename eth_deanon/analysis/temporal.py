"""
Temporal analysis functions for Ethereum transaction analysis
"""

import logging
import pandas as pd
import networkx as nx
from datetime import datetime, timedelta
from ..config.settings import WINDOW_SIZE_DAYS
from .entropy import calculate_entropy

logger = logging.getLogger(__name__)

def create_time_windows(df, window_size_days=WINDOW_SIZE_DAYS):
    """
    Create time windows for temporal analysis

    Args:
        df: DataFrame with transaction data
        window_size_days: Size of each time window in days

    Returns:
        List of (start_time, end_time) tuples
    """
    time_windows = []
    start_date = df['timestamp'].min()
    end_date = df['timestamp'].max()
    window_size = timedelta(days=window_size_days)
    current_date = start_date

    while current_date < end_date:
        next_date = current_date + window_size
        time_windows.append((current_date, next_date))
        current_date = next_date

    logger.info(f"Created {len(time_windows)} time windows of {window_size} each")
    return time_windows

def perform_temporal_clustering(df, time_windows):
    """
    Perform temporal clustering on transaction data

    Args:
        df: DataFrame with transaction data
        time_windows: List of (start_time, end_time) tuples

    Returns:
        Dictionary of results including temporal clusters and transitions
    """
    logger.info("\n=== BEGINNING TEMPORAL CLUSTER ANALYSIS ===\n")

    # Track cluster assignments over time
    address_cluster_history = {}
    cluster_evolution = []

    for i, (start_time, end_time) in enumerate(time_windows):
        logger.info(f"Processing window {i+1}/{len(time_windows)}: {start_time} to {end_time}")

        # Filter transactions for current time window
        window_df = df[(df['timestamp'] >= start_time) & (df['timestamp'] < end_time)]

        if len(window_df) < 10:  # Skip windows with too few transactions
            logger.info(f"  Skipping window - only {len(window_df)} transactions found")
            continue

        # Build graph for current window
        G_window = nx.DiGraph()
        for _, row in window_df.iterrows():
            G_window.add_edge(row['from_address'], row['to_address'])

        # Filter nodes with minimal activity
        filtered_nodes = [n for n in G_window.nodes() if G_window.out_degree(n) > 0]
        G_window = G_window.subgraph(filtered_nodes).copy()

        # Calculate features for clustering
        entropy_window = {node: calculate_entropy(G_window, node) for node in G_window.nodes()}
        degrees_window = dict(G_window.degree())
        
        # Filter out nodes with total degree <= 1
        filtered_nodes = {addr: deg for addr, deg in degrees_window.items() if deg > 1}
        window_summary = pd.DataFrame({
            'address': list(filtered_nodes.keys()),
            'degree': list(filtered_nodes.values()),
            'entropy': [entropy_window.get(a, 0) for a in filtered_nodes.keys()],
            'window_start': start_time,
            'window_end': end_time
        })

        # Convert all addresses to strings to ensure consistent typing
        addresses = [str(addr) for addr in degrees_window.keys()]

        window_summary = pd.DataFrame({
            'address': addresses,
            'degree': list(degrees_window.values()),
            'entropy': [entropy_window.get(a, 0) for a in degrees_window.keys()],
            'window_start': start_time,
            'window_end': end_time
        })

        # Explicitly set address as string type
        window_summary['address'] = window_summary['address'].astype(str)

        if len(window_summary) < 5:  # Skip if too few addresses
            logger.info(f"  Skipping window - only {len(window_summary)} addresses found")
            continue

        # Extract features for clustering
        features = window_summary[['degree', 'entropy']].values

        # Apply clustering
        try:
            import hdbscan
            hdb = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=10)
            window_summary['cluster'] = hdb.fit_predict(features)
        except Exception as e:
            logger.error(f"  Clustering error: {e}")
            continue

        # Track cluster assignments
        for _, row in window_summary.iterrows():
            addr = str(row['address'])
            cluster = row['cluster']
            if addr not in address_cluster_history:
                address_cluster_history[addr] = []
            address_cluster_history[addr].append((start_time, cluster))

        # Store window results
        cluster_evolution.append(window_summary)

    if not cluster_evolution:
        logger.error("No valid time windows could be processed. Try adjusting parameters.")
        return None

    # Combine all window results
    temporal_clusters = pd.concat(cluster_evolution, ignore_index=True)

    logger.info("\nTemporal clustering complete!")
    logger.info(f"- Processed addresses across {len(cluster_evolution)} time windows")
    logger.info(f"- Tracking {len(address_cluster_history)} unique addresses")

    # Find addresses that appear in multiple time windows
    multi_window_addresses = [addr for addr, history in address_cluster_history.items() if len(history) > 1]
    logger.info(f"- Found {len(multi_window_addresses)} addresses appearing in multiple time windows")

    # Track cluster transitions
    transitions = []
    for addr in multi_window_addresses:
        history = address_cluster_history[addr]
        for i in range(len(history) - 1):
            time1, cluster1 = history[i]
            time2, cluster2 = history[i + 1]
            if cluster1 != cluster2:  # Only record when cluster changes
                transitions.append({
                    'address': str(addr),
                    'from_time': time1,
                    'to_time': time2,
                    'from_cluster': cluster1,
                    'to_cluster': cluster2
                })

    transitions_df = pd.DataFrame(transitions)
    logger.info(f"- Detected {len(transitions_df)} cluster transitions")

    # Return results dictionary
    results = {
        'temporal_clusters': temporal_clusters,
        'address_cluster_history': address_cluster_history,
        'multi_window_addresses': multi_window_addresses,
        'transitions_df': transitions_df,
        'cluster_evolution': cluster_evolution
    }

    return results
