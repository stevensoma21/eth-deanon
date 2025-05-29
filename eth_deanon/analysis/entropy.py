"""
Entropy analysis functions for Ethereum transaction analysis
"""

import logging
import numpy as np
from collections import Counter
import networkx as nx
import pandas as pd
from ..config.settings import STABILITY_THRESHOLDS

logger = logging.getLogger(__name__)

def calculate_entropy(graph, address):
    """
    Calculate the entropy of an address in a graph

    Args:
        graph: NetworkX directed graph
        address: Address to calculate entropy for

    Returns:
        Entropy value (float)
    """
    if graph.out_degree(address) == 0:
        return 0.0
    targets = [t for _, t in graph.out_edges(address)]
    freq = Counter(targets)
    total = sum(freq.values())
    probs = [count / total for count in freq.values()]
    return -sum(p * np.log2(p) for p in probs)

def compute_address_metrics(graph):
    """
    Compute metrics for addresses in a graph, filtering out single-use addresses.

    Args:
        graph: NetworkX directed graph

    Returns:
        DataFrame with address metrics
    """
    logger.info("Calculating entropy and degree metrics...")

    # Only keep nodes with degree > 1 (i.e., more than one interaction)
    degrees = dict(graph.degree())
    filtered_nodes = [node for node, deg in degrees.items() if deg > 1]

    entropy = {node: calculate_entropy(graph, node) for node in filtered_nodes}
    filtered_degrees = {node: degrees[node] for node in filtered_nodes}

    summary_df = pd.DataFrame({
        'address': list(filtered_degrees.keys()),
        'degree': list(filtered_degrees.values()),
        'entropy': [entropy.get(a, 0) for a in filtered_degrees.keys()]
    })

    logger.info(f"Filtered to {len(summary_df)} addresses with degree > 1")
    return summary_df

def analyze_time_deltas_by_cohort(df, cohort_name):
    """
    Compute and analyze inter-transaction times for a given cohort.

    Args:
        df: DataFrame containing at least ['address', 'timestamp']
        cohort_name: Name of the cohort (for plot titles)

    Returns:
        dict with:
            - interarrival_times: list of ∆t in seconds
            - powerlaw_fit: fitted powerlaw.Fit object
            - lognorm_params: (shape, loc, scale) for lognormal fit
    """
    logger.info(f"Analyzing {cohort_name} cohort...")

    # Ensure sorting
    df = df.sort_values(by=['address', 'timestamp'])
    df = df[df['cohort']==cohort_name]
    df['timestamp'] = pd.to_datetime(df['timestamp'])  # in case it's string
    df['time_diff'] = df.groupby('address')['timestamp'].diff().dt.total_seconds()
    delta_series = df['time_diff'].dropna()
    delta_series = delta_series[delta_series > 0]

    # Plot histogram
    import matplotlib.pyplot as plt
    import powerlaw
    from scipy.stats import lognorm

    plt.figure(figsize=(14, 10))
    plt.hist(delta_series, bins=100, density=True, alpha=0.4, label='Empirical', color='gray', log=True)

    # Fit power-law
    fit = powerlaw.Fit(delta_series, xmin=1)  # You can tune xmin
    fit.power_law.plot_pdf(label='Power-law', color='red')

    # Fit log-normal for comparison
    shape, loc, scale = lognorm.fit(delta_series, floc=0)  # Fix loc=0 for stability
    x_vals = np.linspace(min(delta_series), max(delta_series), 500)
    plt.plot(x_vals, lognorm.pdf(x_vals, shape, loc, scale), 'b--', label='Log-normal')

    plt.title(f'⏱ Inter-Transaction Time Distribution — {cohort_name}')
    plt.xlabel('Time Between Transactions (s)')
    plt.ylabel('Density (log scale)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return {
        'interarrival_times': delta_series.values,
        'powerlaw_fit': fit,
        'lognorm_params': (shape, loc, scale)
    }
