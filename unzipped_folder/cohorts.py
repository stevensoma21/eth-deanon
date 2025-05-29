# cohorts.py
# === cohorts.py ===
import pandas as pd


def calculate_stability_metrics(address_cluster_history):
    """
    Compute a stability score for each address based on cluster changes over time.
    """
    print("Calculating stability metrics...")
    address_stability = {}

    for addr, history in address_cluster_history.items():
        if len(history) <= 1:
            continue
        clusters = history#[c for _, c in history]
        unique_clusters = set(clusters)
        stability_score = len(unique_clusters) / len(history)
        address_stability[addr] = stability_score

    stability_df = pd.DataFrame({
        'address': list(address_stability.keys()),
        'stability_score': list(address_stability.values()),
        'num_windows': [len(address_cluster_history[addr]) for addr in address_stability.keys()]
    })

    return stability_df


def define_cohorts(stability_df, temporal_clusters, address_cluster_history):
    """
    Segment addresses into cohorts based on stability and entropy transitions.
    """
    print("Defining cohorts...")
    
    # Debug stability metrics
    print("\n[Stability Metrics Debugging]")
    print("Stability score distribution:")
    print(stability_df['stability_score'].describe())
    print("\nNumber of windows distribution:")
    print(stability_df['num_windows'].describe())
    
    # Modified thresholds for high and low stability
    high_stability = stability_df[(stability_df['stability_score'] < 0.5) & (stability_df['num_windows'] > 1)]
    low_stability = stability_df[(stability_df['stability_score'] > 0.5) & (stability_df['num_windows'] > 1)]
    
    # Modified frequent appearance to use percentage of total windows
    total_windows = temporal_clusters['window_start'].nunique()
    min_windows = max(2, int(total_windows * 0.3))  # Appear in at least 30% of windows
    frequent_appearance = stability_df[stability_df['num_windows'] >= min_windows]
    
    print(f"\n[Frequent Appearance Debugging]")
    print(f"Total number of time windows: {total_windows}")
    print(f"Minimum windows for frequent appearance: {min_windows}")
    print(f"Number of frequent addresses: {len(frequent_appearance)}")

    # Modified entropy transition calculation
    entropy_delta = temporal_clusters.groupby('address')['entropy'].apply(lambda x: x.max() - x.min()).reset_index()
    entropy_delta.columns = ['address', 'entropy_delta']
    
    # Use absolute threshold instead of quantile since distribution is skewed
    threshold = 0.1  # Minimum entropy change to be considered significant
    entropy_transition_addresses = entropy_delta[entropy_delta['entropy_delta'] > threshold]['address'].tolist()

    print("\n[Entropy Delta Debugging]")
    print("Entropy delta stats:")
    print(entropy_delta['entropy_delta'].describe())
    print("Entropy threshold:", threshold)
    print("Addresses above threshold:", len(entropy_transition_addresses))

    # Assign cohort labels to temporal clusters
    cohort_assignments = pd.DataFrame({'address': stability_df['address']})
    cohort_assignments['cohort'] = 'Unlabeled'
    cohort_assignments.loc[cohort_assignments['address'].isin(high_stability['address']), 'cohort'] = 'High Stability'
    cohort_assignments.loc[cohort_assignments['address'].isin(low_stability['address']), 'cohort'] = 'Low Stability'
    cohort_assignments.loc[cohort_assignments['address'].isin(frequent_appearance['address']), 'cohort'] = 'Frequent Appearance'
    cohort_assignments.loc[cohort_assignments['address'].isin(entropy_transition_addresses), 'cohort'] = 'Entropy Transition'

    # Debug cohort assignments
    print("\n[Cohort Assignment Debugging]")
    print("Cohort distribution:")
    print(cohort_assignments['cohort'].value_counts())

    temporal_clusters = temporal_clusters.merge(cohort_assignments, on='address', how='left')
    entropy_transition_df = stability_df[stability_df['address'].isin(entropy_transition_addresses)]

    return {
        'high_stability_df': high_stability,
        'low_stability_df': low_stability,
        'frequent_appearance_df': frequent_appearance,
        'entropy_transition_df': entropy_transition_df,
        'entropy_transition_addresses': entropy_transition_addresses,
        'temporal_clusters': temporal_clusters
    }


