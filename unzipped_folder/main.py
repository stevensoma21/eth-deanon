# main.py
import sys
import os
from datetime import timedelta
import pandas as pd
import pytz
from google.colab import drive

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Now import the local modules
from visualization import (
    plot_clusters,
    plot_cohort_comparisons,
    plot_address_evolution,
    plot_3d_visualization,
    plot_cohort_aggregate_evolution
)
from temporal_clustering import perform_temporal_clustering, analyze_cohort_evolution
from config import BQ_DATASET, BQ_TABLE, BQ_CREDS_PATH, DEFAULT_START_DATE, DEFAULT_END_DATE, DEFAULT_TX_LIMIT
from data_loader import setup_google_cloud, load_transaction_data
from graph_utils import build_transaction_graph, compute_address_metrics
from clustering import apply_clustering
from cohorts import calculate_stability_metrics, define_cohorts
from reporting import analyze_trajectory_patterns, generate_comprehensive_report
from cohort_tracking import create_cohort_tracking_df, analyze_cohort_overlap, export_cohort_tracking

def main():
    # Set up BigQuery
    client = setup_google_cloud(BQ_CREDS_PATH)

    # Debug mode - use small subset of data
    DEBUG_MODE = True  # Set to False for full analysis
    if DEBUG_MODE:
        print("Running in DEBUG mode with small dataset...")
        # Load only last 2 days of data
        debug_start_date = DEFAULT_END_DATE - timedelta(days=2)
        df = load_transaction_data(client, debug_start_date, DEFAULT_END_DATE, 10000, BQ_DATASET, BQ_TABLE)
        # Use 12-hour windows instead of 3-day windows
        time_windows = [(utc.localize(start), utc.localize(start + timedelta(hours=12))) 
                       for start in pd.date_range(debug_start_date, DEFAULT_END_DATE, freq='12H')]
    else:
        # Load full dataset
        df = load_transaction_data(client, DEFAULT_START_DATE, DEFAULT_END_DATE, DEFAULT_TX_LIMIT, BQ_DATASET, BQ_TABLE)
        time_windows = [(utc.localize(start), utc.localize(start + timedelta(days=3))) 
                       for start in pd.date_range(DEFAULT_START_DATE, DEFAULT_END_DATE, freq='3D')]

    print(f"Loaded {len(df)} transactions.")

    # Build transaction graph
    G_complete, G_filtered = build_transaction_graph(df, filter_min_degree=2)
    print(f"Graph: {G_filtered.number_of_nodes()} filtered nodes")

    # Compute address-level metrics
    summary_df = compute_address_metrics(G_filtered)

    # Apply clustering
    summary_df = apply_clustering(summary_df, methods=['dbscan', 'hdbscan'])

    # Plot clusters
    plot_clusters(summary_df, cluster_col='dbscan_cluster')
    plot_clusters(summary_df, cluster_col='hdbscan_cluster')

    # Perform temporal clustering (add this!)
    #results = perform_temporal_clustering(df, time_windows=[7])  # or whatever time windows you're using

    # Build actual time windows
    #time_windows = [(start, start + timedelta(days=7)) for start in pd.date_range(DEFAULT_START_DATE, DEFAULT_END_DATE, freq='7D')]

    import pytz
    utc = pytz.UTC
    # Changed from 7-day windows to 3-day windows for more granular analysis
    time_windows = [(utc.localize(start), utc.localize(start + timedelta(days=3))) 
                   for start in pd.date_range(DEFAULT_START_DATE, DEFAULT_END_DATE, freq='3D')]

    results = perform_temporal_clustering(df, time_windows)

    # Calculate cohort stability
    stability_df = calculate_stability_metrics(results['address_cluster_history'])

    # Define cohorts
    cohorts = define_cohorts(
        stability_df,
        results['temporal_clusters'],
        results['address_cluster_history']
    )

    # Analyze cohort evolution
    print("\nAnalyzing cohort evolution...")
    evolution_results = analyze_cohort_evolution(cohorts, results['temporal_clusters'])

    if not evolution_results or 'evolution_df' not in evolution_results:
        logger.error("Failed to analyze cohort evolution")
        return

    # Create visualizations
    logger.info("Creating visualizations...")
    if not evolution_results['evolution_df'].empty:
        plot_cohort_comparisons(evolution_results['evolution_df'])
        plot_address_evolution(evolution_results, results['temporal_clusters'])
        plot_3d_visualization(evolution_results, results['temporal_clusters'])

        # Plot aggregate evolution for each cohort
        for cohort_name in ['High Stability', 'Low Stability', 'Frequent Appearance', 'Entropy Transition']:
            plot_cohort_aggregate_evolution(results['temporal_clusters'], evolution_results, cohort_name)

    # Create cohort tracking DataFrame
    print("\nCreating cohort tracking DataFrame...")
    tracking_df = create_cohort_tracking_df(cohorts, results['temporal_clusters'])
    
    # Analyze cohort overlap
    print("\nAnalyzing cohort overlap...")
    overlap_matrix = analyze_cohort_overlap(tracking_df)
    print("\nCohort Overlap Matrix:")
    print(overlap_matrix)
    
    # Export tracking data
    print("\nExporting cohort tracking data...")
    export_cohort_tracking(tracking_df, 'cohort_tracking.csv')
    
    # Print tracking statistics
    print("\nCohort Tracking Statistics:")
    print(f"Total unique addresses tracked: {tracking_df['address'].nunique()}")
    print(f"Addresses in multiple cohorts: {tracking_df[tracking_df['in_multiple_cohorts']]['address'].nunique()}")
    
    # Analyze cohort distribution over time
    cohort_time_dist = tracking_df.groupby(['window_start', 'cohort']).size().unstack(fill_value=0)
    print("\nCohort distribution over time:")
    print(cohort_time_dist)
    
    # Merge tracking data with transaction data
    print("\nMerging tracking data with transaction data...")
    tx_df = df.rename(columns={'from_address': 'address'})
    tx_df_with_tracking = tx_df.merge(
        tracking_df[['address', 'window_start', 'cohort', 'degree', 'entropy', 'in_multiple_cohorts']], 
        on='address', 
        how='inner'
    )
    
    # Export merged data
    tx_df_with_tracking.to_csv('tx_df_with_tracking.csv', index=False)
    print("Exported merged transaction and tracking data to tx_df_with_tracking.csv")

    # Analyze time deltas by cohort
    for cohort_name in ['High Stability', 'Low Stability', 'Frequent Appearance', 'Entropy Transition']:
        cohort_data = tx_df_with_tracking[tx_df_with_tracking['cohort'] == cohort_name]
        if not cohort_data.empty:
            analyze_time_deltas_by_cohort(cohort_data, cohort_name)
        else:
            logger.warning(f"No data found for cohort: {cohort_name}")

    print("Pipeline completed.")

if __name__ == "__main__":
    main()
