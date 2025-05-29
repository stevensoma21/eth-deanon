import networkx as nx
from graph_utils import calculate_entropy
import pandas as pd 
import hdbscan
import numpy as np

def analyze_cohort_evolution(cohorts, temporal_clusters, sample_size=50):
    """
    Analyze evolution of addresses in each cohort and gracefully skip empty ones.

    Args:
        cohorts: Dictionary with cohort DataFrames
        temporal_clusters: DataFrame with temporal clustering results
        sample_size: Number of addresses to sample from each cohort

    Returns:
        Dictionary containing evolution DataFrame and sample lists per cohort
    """
    print("\nAnalyzing behavior evolution for cohorts...")

    evolution_data = {}
    evolution_df_rows = []

    # Print cohort counts for debugging
    print("\nCohort counts (before sampling):")
    for cohort_name, df_key in [
        ('High', 'high_stability_df'),
        ('Low', 'low_stability_df'),
        ('Frequent', 'frequent_appearance_df'),
        ('Entropy', 'entropy_transition_df')
    ]:
        if df_key in cohorts and not cohorts[df_key].empty:
            print(f"{cohort_name}: {len(cohorts[df_key])}")
        else:
            print(f"{cohort_name}: 0")

    # Process each cohort
    for cohort_name, df_key in [
        ('High Stability', 'high_stability_df'),
        ('Low Stability', 'low_stability_df'),
        ('Frequent Appearance', 'frequent_appearance_df'),
        ('Entropy Transition', 'entropy_transition_df')
    ]:
        df = cohorts.get(df_key)
        if df is None or df.empty:
            print(f"Skipping {cohort_name} cohort — no data.")
            evolution_data[f"{cohort_name.lower().replace(' ', '_')}_sample"] = []
            continue

        # Ensure we have the address column
        if 'address' not in df.columns:
            print(f"Warning: No 'address' column in {cohort_name} cohort DataFrame")
            print("Available columns:", df.columns.tolist())
            continue

        # Sample addresses
        sample_size_actual = min(sample_size, len(df))
        sample_addrs = df['address'].sample(sample_size_actual, replace=False).tolist()
        evolution_data[f"{cohort_name.lower().replace(' ', '_')}_sample"] = sample_addrs

        print(f"{cohort_name}: {sample_size_actual} addresses sampled.")

        # Extract evolution data for each sampled address
        for addr in sample_addrs:
            addr_data = temporal_clusters[temporal_clusters['address'] == addr].sort_values('window_start')
            if not addr_data.empty:
                evolution_df_rows.append({
                    'address': addr,
                    'cohort': cohort_name,
                    'start_entropy': addr_data['entropy'].iloc[0],
                    'end_entropy': addr_data['entropy'].iloc[-1],
                    'degree_change': addr_data['degree'].max() - addr_data['degree'].min(),
                    'entropy_change': addr_data['entropy'].max() - addr_data['entropy'].min(),
                    'n_windows': addr_data['window_start'].nunique(),
                    'cluster_changes': (addr_data['cluster'] != addr_data['cluster'].shift()).sum() - 1
                })

    # Create evolution DataFrame
    if evolution_df_rows:
        evolution_df = pd.DataFrame(evolution_df_rows)
        evolution_data['evolution_df'] = evolution_df
        evolution_df.to_csv('evolution_df.csv', index=False)
    else:
        print("Warning: No evolution data collected")
        evolution_data['evolution_df'] = pd.DataFrame()

    return evolution_data

def extract_address_evolution(address, temporal_clusters):
    """
    Extract the degree and entropy over time for a given address.
    """
    addr_df = temporal_clusters[temporal_clusters['address'] == address].sort_values('window_start')
    if addr_df.empty:
        return None

    return {
        'address': address,
        'start_entropy': addr_df['entropy'].iloc[0],
        'end_entropy': addr_df['entropy'].iloc[-1],
        'degree_change': addr_df['degree'].max() - addr_df['degree'].min(),
        'entropy_change': addr_df['entropy'].max() - addr_df['entropy'].min(),
        'n_windows': addr_df['window_start'].nunique(),
        'cluster_changes': (addr_df['cluster'] != addr_df['cluster'].shift()).sum() - 1
    }

def perform_temporal_clustering(df, time_windows):
    """
    Perform temporal clustering on transaction data

    Args:
        df: DataFrame with transaction data
        time_windows: List of (start_time, end_time) tuples

    Returns:
        Dictionary of results including temporal clusters and transitions
    """
    print("\n=== BEGINNING TEMPORAL CLUSTER ANALYSIS ===\n")

    # Track cluster assignments over time
    address_cluster_history = {}
    cluster_evolution = []

    for i, (start_time, end_time) in enumerate(time_windows):
        print(f"Processing window {i+1}/{len(time_windows)}: {start_time} to {end_time}")

        # Filter transactions for current time window
        window_df = df[(df['timestamp'] >= start_time) & (df['timestamp'] < end_time)]

        if len(window_df) < 10:  # Skip windows with too few transactions
            print(f"  Skipping window - only {len(window_df)} transactions found")
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
            print(f"  Skipping window - only {len(window_summary)} addresses found")
            continue

        # Extract features for clustering
        features = window_summary[['degree', 'entropy']].values

        # Apply clustering
        try:
            hdb = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=20)  # Adjusted for smaller window sizes
            window_summary['cluster'] = hdb.fit_predict(features)
        except Exception as e:
            print(f"  Clustering error: {e}")
            continue

        # Track cluster assignments
        for _, row in window_summary.iterrows():
            addr = str(row['address'])  # Ensure it's a string
            cluster = row['cluster']
            if addr not in address_cluster_history:
                address_cluster_history[addr] = []
            address_cluster_history[addr].append((start_time, cluster))

        # Store window results
        cluster_evolution.append(window_summary)

    if not cluster_evolution:
        print("No valid time windows could be processed. Try adjusting parameters.")
        return None

    # Combine all window results
    temporal_clusters = pd.concat(cluster_evolution, ignore_index=True)

    print("\nTemporal clustering complete!")
    print(f"- Processed addresses across {len(cluster_evolution)} time windows")
    print(f"- Tracking {len(address_cluster_history)} unique addresses")

    # Find addresses that appear in multiple time windows
    multi_window_addresses = [addr for addr, history in address_cluster_history.items() if len(history) > 1]
    print(f"- Found {len(multi_window_addresses)} addresses appearing in multiple time windows")

    # Track cluster transitions
    transitions = []
    for addr in multi_window_addresses:
        history = address_cluster_history[addr]
        for i in range(len(history) - 1):
            time1, cluster1 = history[i]
            time2, cluster2 = history[i + 1]
            if cluster1 != cluster2:  # Only record when cluster changes
                transitions.append({
                    'address': str(addr),  # Ensure it's a string
                    'from_time': time1,
                    'to_time': time2,
                    'from_cluster': cluster1,
                    'to_cluster': cluster2
                })

    transitions_df = pd.DataFrame(transitions)
    print(f"- Detected {len(transitions_df)} cluster transitions")

    # Return results dictionary
    results = {
        'temporal_clusters': temporal_clusters,
        'address_cluster_history': address_cluster_history,
        'multi_window_addresses': multi_window_addresses,
        'transitions_df': transitions_df,
        'cluster_evolution': cluster_evolution
    }
    
    return results

def analyze_cohort_evolution(results):
    """
    Analyze real cohort evolution from temporal cluster data.
    """
    print("Analyzing cohort evolution from temporal clustering results...")
    df = results['temporal_clusters'].copy()

    # Sort and compute per-address changes
    df = df.sort_values(['address', 'window_start'])
    grouped = df.groupby('address')

    deltas = grouped.agg(
        start_entropy=('entropy', 'first'),
        end_entropy=('entropy', 'last'),
        degree_change=('degree', lambda x: x.max() - x.min()),
        entropy_change=('entropy', lambda x: x.max() - x.min()),
        n_windows=('window_start', 'nunique'),
        cluster_changes=('cluster', lambda x: (x != x.shift()).sum() - 1)
    ).reset_index()

    # Assign cohorts based on actual behavior
    deltas['cohort'] = 'Unassigned'
    deltas.loc[deltas['cluster_changes'] == 0, 'cohort'] = 'High Stability'
    deltas.loc[deltas['cluster_changes'] > 2, 'cohort'] = 'Entropy Transition'
    deltas.loc[deltas['n_windows'] >= 4, 'cohort'] = 'Frequent Appearance'
    deltas.loc[(deltas['cohort'] == 'Unassigned') & (deltas['entropy_change'] > 0.5), 'cohort'] = 'Low Stability'

    # Sample cohort addresses
    sample_addrs = {}
    for label in ['High Stability', 'Low Stability', 'Frequent Appearance', 'Entropy Transition']:
        key = label.lower().replace(' ', '_') + '_sample'
        sample_addrs[key] = deltas[deltas['cohort'] == label]['address'].head(5).tolist()

    return deltas, sample_addrs
