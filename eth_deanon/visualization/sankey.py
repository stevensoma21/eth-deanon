"""
Sankey diagram visualization functions for Ethereum transaction analysis
"""

import logging
import pandas as pd
import plotly.graph_objects as go
from ..config.settings import PLOT_SETTINGS
import numpy as np

logger = logging.getLogger(__name__)

def create_sankey_diagram(transition_df, title=None, save_path=None):
    """
    Create a Sankey diagram from a transition matrix

    Args:
        transition_df: DataFrame with transition probabilities between clusters
        title: Optional title for the diagram
        save_path: Optional path to save the diagram

    Returns:
        Plotly figure object
    """
    logger.info("Creating Sankey diagram...")
    
    # Get cluster labels
    clusters = transition_df.index.tolist()
    n_clusters = len(clusters)
    
    # Create node labels
    node_labels = [f"Cluster {c}" for c in clusters]
    
    # Create source and target indices
    source = []
    target = []
    value = []
    
    for i, src in enumerate(clusters):
        for j, tgt in enumerate(clusters):
            if transition_df.loc[src, tgt] > 0:
                source.append(i)
                target.append(j + n_clusters)  # Offset target indices
                value.append(transition_df.loc[src, tgt])
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels + node_labels,  # Duplicate labels for source and target
            color=PLOT_SETTINGS['color_palette'][:n_clusters] * 2  # Duplicate colors
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=[PLOT_SETTINGS['color_palette'][i % len(PLOT_SETTINGS['color_palette'])] for i in source]
        )
    )])
    
    # Update layout
    fig.update_layout(
        title_text=title or "Cluster Transition Flow",
        font_size=10,
        height=800,
        width=1200
    )
    
    if save_path:
        fig.write_html(save_path)
        logger.info(f"Sankey diagram saved to {save_path}")
    
    return fig

def create_cohort_sankey(cohort_evolution_df, save_path=None):
    """
    Create a Sankey diagram showing cohort evolution over time

    Args:
        cohort_evolution_df: DataFrame with cohort evolution data
        save_path: Optional path to save the diagram

    Returns:
        Plotly figure object
    """
    logger.info("Creating cohort evolution Sankey diagram...")
    
    # Get unique cohorts
    cohorts = cohort_evolution_df['cohort'].unique()
    
    # Create node labels
    node_labels = []
    for cohort in cohorts:
        node_labels.extend([
            f"{cohort} Start",
            f"{cohort} End"
        ])
    
    # Create source and target indices
    source = []
    target = []
    value = []
    
    for i, cohort in enumerate(cohorts):
        cohort_data = cohort_evolution_df[cohort_evolution_df['cohort'] == cohort]
        
        # Add transitions from start to end
        source.append(i * 2)  # Start node index
        target.append(i * 2 + 1)  # End node index
        value.append(len(cohort_data))
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color=PLOT_SETTINGS['color_palette'][:len(cohorts)] * 2  # Duplicate colors
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=[PLOT_SETTINGS['color_palette'][i % len(PLOT_SETTINGS['color_palette'])] for i in source]
        )
    )])
    
    # Update layout
    fig.update_layout(
        title_text="Cohort Evolution Flow",
        font_size=10,
        height=600,
        width=800
    )
    
    if save_path:
        fig.write_html(save_path)
        logger.info(f"Cohort Sankey diagram saved to {save_path}")
    
    return fig

def create_entropy_transition_sankey(temporal_clusters, save_path=None):
    """
    Create a Sankey diagram showing entropy transitions

    Args:
        temporal_clusters: DataFrame with temporal clustering results
        save_path: Optional path to save the diagram

    Returns:
        Plotly figure object
    """
    logger.info("Creating entropy transition Sankey diagram...")
    
    # Calculate entropy transitions
    entropy_transitions = []
    for addr in temporal_clusters['address'].unique():
        addr_data = temporal_clusters[temporal_clusters['address'] == addr].sort_values('window_start')
        if len(addr_data) > 1:
            start_entropy = addr_data['entropy'].iloc[0]
            end_entropy = addr_data['entropy'].iloc[-1]
            entropy_transitions.append({
                'address': addr,
                'start_entropy': start_entropy,
                'end_entropy': end_entropy
            })
    
    # Create bins for entropy values
    entropy_bins = np.linspace(0, 1, 6)  # 5 bins
    bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
    
    # Create transition matrix
    transition_counts = np.zeros((len(bin_labels), len(bin_labels)))
    for trans in entropy_transitions:
        start_bin = np.digitize(trans['start_entropy'], entropy_bins) - 1
        end_bin = np.digitize(trans['end_entropy'], entropy_bins) - 1
        if 0 <= start_bin < len(bin_labels) and 0 <= end_bin < len(bin_labels):
            transition_counts[start_bin, end_bin] += 1
    
    # Create source and target indices
    source = []
    target = []
    value = []
    
    for i in range(len(bin_labels)):
        for j in range(len(bin_labels)):
            if transition_counts[i, j] > 0:
                source.append(i)
                target.append(j + len(bin_labels))  # Offset target indices
                value.append(transition_counts[i, j])
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=bin_labels + bin_labels,  # Duplicate labels for source and target
            color=PLOT_SETTINGS['color_palette'][:len(bin_labels)] * 2  # Duplicate colors
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color=[PLOT_SETTINGS['color_palette'][i % len(PLOT_SETTINGS['color_palette'])] for i in source]
        )
    )])
    
    # Update layout
    fig.update_layout(
        title_text="Entropy Transition Flow",
        font_size=10,
        height=600,
        width=800
    )
    
    if save_path:
        fig.write_html(save_path)
        logger.info(f"Entropy transition Sankey diagram saved to {save_path}")
    
    return fig
