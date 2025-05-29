"""
Cohort analysis functions for Ethereum transaction analysis
"""

import logging
import pandas as pd
import numpy as np
from ..config.settings import STABILITY_THRESHOLDS

logger = logging.getLogger(__name__)

def calculate_stability_metrics(address_cluster_history):
    """
    Calculate stability metrics for addresses based on cluster history

    Args:
        address_cluster_history: Dictionary mapping addresses to their cluster history

    Returns:
        DataFrame with stability metrics
    """
    logger.info("Creating address stability metrics...")

    address_stability = {}
    for addr, history in address_cluster_history.items():
        if len(history) <= 1:
            continue
        clusters = [c for _, c in history]
        unique_clusters = set(clusters)
        stability_score = len(unique_clusters) / len(history)  # Lower is more stable
        address_stability[addr] = stability_score

    stability_df = pd.DataFrame({
        'address': [str(addr) for addr in address_stability.keys()],
        'stability_score': list(address_stability.values()),
        'num_windows': [len(address_cluster_history[addr]) for addr in address_stability.keys()]
    })

    stability_df['address'] = stability_df['address'].astype(str)

    return stability_df

def define_cohorts(stability_df, temporal_clusters, address_cluster_history, transitions_df=None):
    """
    Define cohorts of addresses based on stability metrics and cluster transitions

    Args:
        stability_df: DataFrame with stability metrics
        temporal_clusters: DataFrame with temporal clustering results
        address_cluster_history: Dictionary mapping addresses to their cluster history
        transitions_df: DataFrame with cluster transitions

    Returns:
        Dictionary with cohort DataFrames and information
    """
    logger.info("Segmenting addresses into cohorts...")

    # Define cohorts based on thresholds
    high_stability = stability_df[
        (stability_df['stability_score'] < STABILITY_THRESHOLDS['high_stability']) & 
        (stability_df['num_windows'] > STABILITY_THRESHOLDS['min_windows'])
    ]
    low_stability = stability_df[
        (stability_df['stability_score'] > STABILITY_THRESHOLDS['low_stability']) & 
        (stability_df['num_windows'] > STABILITY_THRESHOLDS['min_windows'])
    ]
    frequent_appearance = stability_df[
        (stability_df['num_windows'] > STABILITY_THRESHOLDS['frequent_windows'])
    ]

    # Calculate entropy transition
    entropy_delta = temporal_clusters.groupby('address')['entropy'].apply(
        lambda x: x.max() - x.min()
    ).reset_index()
    entropy_delta.columns = ['address', 'entropy_delta']
    entropy_transition_threshold = entropy_delta['entropy_delta'].quantile(0.9)
    entropy_transition_addresses = entropy_delta[
        entropy_delta['entropy_delta'] > entropy_transition_threshold
    ]['address'].tolist()

    # Assign cohort to temporal clusters
    address_history_array = []
    for address, history in address_cluster_history.items():
        for window_start, cluster in history:
            address_history_array.append([address, window_start, cluster])

    address_history_df = pd.DataFrame(
        address_history_array, 
        columns=['address', 'window_start', 'cluster']
    )
    cohort_assignments = address_history_df.groupby('address')['cluster'].first().reset_index()
    cohort_assignments.rename(columns={'cluster': 'cohort'}, inplace=True)
    temporal_clusters = pd.merge(temporal_clusters, cohort_assignments, on='address', how='left')

    # Save cohorts to dataframes
    high_stability_df = stability_df[stability_df['address'].isin(high_stability['address'])]
    low_stability_df = stability_df[stability_df['address'].isin(low_stability['address'])]
    frequent_appearance_df = stability_df[stability_df['address'].isin(frequent_appearance['address'])]
    entropy_transition_df = stability_df[stability_df['address'].isin(entropy_transition_addresses)]

    logger.info(f"- High stability cohort: {len(high_stability)} addresses")
    logger.info(f"- Low stability cohort: {len(low_stability)} addresses")
    logger.info(f"- Frequent appearance cohort: {len(frequent_appearance)} addresses")
    logger.info(f"- Entropy transition cohort: {len(entropy_transition_addresses)} addresses")

    # Return cohort information
    cohorts = {
        'high_stability': high_stability,
        'low_stability': low_stability,
        'frequent_appearance': frequent_appearance,
        'entropy_transition_addresses': entropy_transition_addresses,
        'high_stability_df': high_stability_df,
        'low_stability_df': low_stability_df,
        'frequent_appearance_df': frequent_appearance_df,
        'entropy_transition_df': entropy_transition_df,
        'temporal_clusters': temporal_clusters
    }

    return cohorts

def extract_address_evolution(address, temporal_clusters):
    """
    Extract evolution data for a specific address

    Args:
        address: Address to extract evolution data for
        temporal_clusters: DataFrame with temporal clustering results

    Returns:
        Dictionary with evolution data or None if insufficient data
    """
    addr_data = temporal_clusters[temporal_clusters['address'] == address].sort_values('window_start')
    if len(addr_data) > 1:
        evolution = {
            'address': address,
            'windows': len(addr_data),
            'start_degree': addr_data['degree'].iloc[0],
            'end_degree': addr_data['degree'].iloc[-1],
            'degree_change': addr_data['degree'].iloc[-1] - addr_data['degree'].iloc[0],
            'start_entropy': addr_data['entropy'].iloc[0],
            'end_entropy': addr_data['entropy'].iloc[-1],
            'entropy_change': addr_data['entropy'].iloc[-1] - addr_data['entropy'].iloc[0],
            'cluster_changes': len(set(addr_data['cluster'])) - 1
        }
        return evolution
    return None

def analyze_cohort_evolution(cohorts, temporal_clusters, sample_size=1000):
    """
    Analyze evolution of addresses in each cohort

    Args:
        cohorts: Dictionary with cohort information
        temporal_clusters: DataFrame with temporal clustering results
        sample_size: Number of addresses to sample from each cohort

    Returns:
        DataFrame with evolution data
    """
    logger.info("\nAnalyzing behavior evolution for cohorts...")

    # Get cohort dataframes
    high_stability_df = cohorts['high_stability_df']
    low_stability_df = cohorts['low_stability_df']
    frequent_appearance_df = cohorts['frequent_appearance_df']
    entropy_transition_df = cohorts['entropy_transition_df']

    logger.info("Cohort counts (before sampling):")
    logger.info(f"High: {len(high_stability_df)}")
    logger.info(f"Low: {len(low_stability_df)}")
    logger.info(f"Frequent: {len(frequent_appearance_df)}")
    logger.info(f"Entropy: {len(entropy_transition_df)}")

    # Sample addresses from each cohort
    high_stability_sample = high_stability_df['address'].sample(
        min(sample_size, len(high_stability_df)), 
        replace=False
    ).tolist()
    low_stability_sample = low_stability_df['address'].sample(
        min(sample_size, len(low_stability_df))
    ).tolist()
    frequent_sample = frequent_appearance_df['address'].sample(
        min(sample_size, len(frequent_appearance_df))
    ).tolist()
    entropy_transition_sample = entropy_transition_df['address'].sample(
        min(sample_size, len(entropy_transition_df))
    ).tolist()

    logger.info("Sample sizes:")
    logger.info(f"  High: {len(high_stability_sample)}")
    logger.info(f"  Low: {len(low_stability_sample)}")
    logger.info(f"  Frequent: {len(frequent_sample)}")
    logger.info(f"  Entropy: {len(entropy_transition_sample)}")

    # Extract evolution data
    evolution_data = []

    for addr in high_stability_sample:
        data = extract_address_evolution(addr, temporal_clusters)
        if data:
            data['cohort'] = 'High Stability'
            evolution_data.append(data)

    for addr in low_stability_sample:
        data = extract_address_evolution(addr, temporal_clusters)
        if data:
            data['cohort'] = 'Low Stability'
            evolution_data.append(data)

    for addr in frequent_sample:
        data = extract_address_evolution(addr, temporal_clusters)
        if data:
            data['cohort'] = 'Frequent Appearance'
            evolution_data.append(data)

    for addr in entropy_transition_sample:
        data = extract_address_evolution(addr, temporal_clusters)
        if data:
            data['cohort'] = 'Entropy Transition'
            evolution_data.append(data)

    evolution_df = pd.DataFrame(evolution_data)
    evolution_df.to_csv('evolution_df.csv', index=False)

    # Return evolution data and samples
    result = {
        'evolution_df': evolution_df,
        'high_stability_sample': high_stability_sample,
        'low_stability_sample': low_stability_sample,
        'frequent_sample': frequent_sample,
        'entropy_transition_sample': entropy_transition_sample
    }

    return result

def filter_static_entropy_addresses(temporal_clusters, address_list, min_std=0.001):
    """
    Drop addresses with nearly constant entropy over time.
    """
    filtered = []
    for addr in address_list:
        series = temporal_clusters[temporal_clusters['address'] == addr]['entropy']
        if series.std() > min_std:
            filtered.append(addr)
    return filtered
