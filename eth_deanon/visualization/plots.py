"""
Plotting functions for Ethereum transaction analysis
"""

import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from ..config.settings import PLOT_SETTINGS

logger = logging.getLogger(__name__)

def plot_clusters(df, cluster_col='dbscan_cluster', save_path=None):
    """
    Plot clusters in 2D space (degree vs. entropy) with distinct colors and legend.

    Args:
        df: DataFrame with 'degree', 'entropy', and cluster column
        cluster_col: Name of column containing cluster labels
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=PLOT_SETTINGS['figsize'])

    unique_clusters = sorted(df[cluster_col].unique())
    n_clusters = len(unique_clusters)
    colormap = plt.cm.get_cmap('tab20', n_clusters)

    cluster_colors = {cluster: colormap(i) for i, cluster in enumerate(unique_clusters)}

    for cluster in unique_clusters:
        subset = df[df[cluster_col] == cluster]
        plt.scatter(subset['degree'], subset['entropy'],
                    color=cluster_colors[cluster],
                    label=f'Cluster {cluster}', alpha=0.7, s=60)

    plt.xlabel('Degree')
    plt.ylabel('Entropy')
    plt.title(f'{cluster_col.upper()} Clustering of Addresses')
    plt.legend(title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=PLOT_SETTINGS['dpi'])

    plt.show()

def plot_cohort_comparisons(evolution_df):
    """
    Create visualizations comparing cohorts

    Args:
        evolution_df: DataFrame with evolution data for each cohort
    """
    logger.info("Creating cohort comparison visualizations...")

    # 1. Plot degree change by cohort
    plt.figure(figsize=PLOT_SETTINGS['figsize'])
    if 'cohort' in evolution_df.columns:
        sns.boxplot(x='cohort', y='degree_change', data=evolution_df)
        plt.title('Degree Change by Cohort')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("cohort_degree_change.png", dpi=PLOT_SETTINGS['dpi'])
        plt.show()
    else:
        logger.error("Error: 'cohort' column not found in evolution_df.")

    # 2. Plot entropy change by cohort
    plt.figure(figsize=PLOT_SETTINGS['figsize'])
    sns.boxplot(x='cohort', y='entropy_change', data=evolution_df)
    plt.title('Entropy Change by Cohort')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("cohort_entropy_change.png", dpi=PLOT_SETTINGS['dpi'])
    plt.show()

    # 3. Create a scatterplot of initial vs final entropy by cohort
    plt.figure(figsize=PLOT_SETTINGS['figsize'])
    for cohort in evolution_df['cohort'].unique():
        cohort_data = evolution_df[evolution_df['cohort'] == cohort]
        plt.scatter(cohort_data['start_entropy'],
                   cohort_data['end_entropy'],
                   label=cohort,
                   alpha=0.7,
                   s=80)

    # Add reference line
    max_val = max(evolution_df['start_entropy'].max(), evolution_df['end_entropy'].max()) + 0.5
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)

    plt.xlabel('Initial Entropy')
    plt.ylabel('Final Entropy')
    plt.title('Entropy Evolution by Cohort')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("cohort_entropy_evolution.png", dpi=PLOT_SETTINGS['dpi'])
    plt.show()

def plot_cohort_aggregate_evolution(temporal_clusters, cohort_samples, cohort_name):
    """
    Plot aggregate evolution metrics for a cohort over time.

    Args:
        temporal_clusters: DataFrame with temporal clustering results
        cohort_samples: Dictionary with sample addresses for each cohort
        cohort_name: Name of the cohort to plot
    """
    # Map cohort name to actual sample key
    key_map = {
        "High Stability": "high_stability_sample",
        "Low Stability": "low_stability_sample",
        "Frequent Appearance": "frequent_sample",
        "Entropy Transition": "entropy_transition_sample"
    }

    sample_key = key_map.get(cohort_name)
    if not sample_key or sample_key not in cohort_samples:
        logger.warning(f"No addresses found for cohort: {cohort_name}")
        return

    addresses = cohort_samples[sample_key]

    if not addresses:
        logger.warning(f"No addresses found for cohort: {cohort_name}")
        return

    cohort_data = temporal_clusters[temporal_clusters['address'].isin(addresses)]

    if cohort_data.empty:
        logger.warning(f"No data found for {cohort_name} cohort.")
        return

    grouped = cohort_data.groupby('window_start')

    mean_entropy = grouped['entropy'].mean()
    std_entropy = grouped['entropy'].std()
    mean_degree = grouped['degree'].mean()
    std_degree = grouped['degree'].std()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

    # Entropy Plot
    axes[0].plot(mean_entropy.index, mean_entropy, label='Mean Entropy', marker='o')
    axes[0].fill_between(mean_entropy.index, mean_entropy - std_entropy, mean_entropy + std_entropy, alpha=0.3)
    axes[0].set_title(f'{cohort_name}: Entropy Evolution')
    axes[0].set_ylabel('Entropy')
    axes[0].set_xlabel('Time')
    axes[0].grid(True)

    # Degree Plot
    axes[1].plot(mean_degree.index, mean_degree, label='Mean Degree', marker='o')
    axes[1].fill_between(mean_degree.index, mean_degree - std_degree, mean_degree + std_degree, alpha=0.3)
    axes[1].set_title(f'{cohort_name}: Degree Evolution')
    axes[1].set_ylabel('Degree')
    axes[1].set_xlabel('Time')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
