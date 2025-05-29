"""
Main execution module for Ethereum transaction analysis
"""

import logging
import os
from google.colab import drive
from .data.loader import setup_google_cloud, load_transaction_data, normalize_addresses
from .analysis.entropy import compute_address_metrics, analyze_time_deltas_by_cohort
from .analysis.clustering import apply_clustering
from .visualization.plots import plot_clusters, plot_cohort_comparisons, plot_cohort_aggregate_evolution
from .config.settings import MIN_DEGREE_FILTER, WINDOW_SIZE_DAYS

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def main():
    """
    Main execution function
    """
    try:
        # Mount Google Drive and setup BigQuery
        drive.mount('/content/drive')
        creds_path = "/content/drive/MyDrive/Colab Notebooks/crypto-argon-351622-5587062ee54e.json"
        client = setup_google_cloud(creds_path)

        if client is None:
            logger.error("Failed to setup BigQuery client")
            return

        # Load transaction data
        df = load_transaction_data(client)
        if df is None or df.empty:
            logger.error("Failed to load transaction data")
            return
        df.to_csv('initial_df.csv')

        # Build transaction graph
        logger.info("Building transaction graphs...")
        G_complete, G_filtered = build_transaction_graph(df, filter_min_degree=MIN_DEGREE_FILTER)

        if G_filtered.number_of_nodes() == 0:
            logger.error("Filtered graph is empty")
            return

        # Compute address metrics
        logger.info("Computing address metrics...")
        summary_df = compute_address_metrics(G_filtered)

        if summary_df is None or summary_df.empty:
            logger.error("Failed to compute address metrics")
            return

        # Extract and plot top subgraphs
        logger.info("Extracting and plotting top subgraphs...")
        try:
            extract_and_plot_top_subgraph(G_complete, summary_df, top_n=3, min_nodes=5, max_nodes=80)
            logger.info("Successfully created subgraph visualization")
        except Exception as e:
            logger.error(f"Failed to create subgraph visualization: {str(e)}")

        # Apply clustering
        logger.info("Applying clustering...")
        summary_df = apply_clustering(
            df=summary_df,
            methods=['dbscan', 'hdbscan'],
            optimize=True
        )

        # Log clustering results
        if 'dbscan_cluster' in summary_df.columns:
            logger.info("DBSCAN cluster counts:\n%s",
                       summary_df['dbscan_cluster'].value_counts().sort_index())
        if 'hdbscan_cluster' in summary_df.columns:
            logger.info("\nHDBSCAN cluster counts:\n%s",
                       summary_df['hdbscan_cluster'].value_counts().sort_index())

        # Visualize clusters
        logger.info("Creating cluster visualizations...")
        plot_clusters(summary_df, cluster_col='dbscan_cluster',
                     save_path="dbscan_clusters.png")
        plot_clusters(summary_df, cluster_col='hdbscan_cluster',
                     save_path="hdbscan_clusters.png")

        # Perform temporal clustering
        logger.info("Performing temporal clustering...")
        time_windows = create_time_windows(df, window_size_days=WINDOW_SIZE_DAYS)
        results = perform_temporal_clustering(df, time_windows)

        if results is None:
            logger.error("Temporal clustering failed")
            return

        # Calculate stability metrics
        logger.info("Calculating stability metrics...")
        stability_df = calculate_stability_metrics(results['address_cluster_history'])
        stability_df.to_csv("stability_metrics.csv", index=False)
        if stability_df is None or stability_df.empty:
            logger.error("Failed to calculate stability metrics")
            return

        # Define cohorts
        logger.info("Defining cohorts...")
        cohorts = define_cohorts(
            stability_df=stability_df,
            temporal_clusters=results['temporal_clusters'],
            address_cluster_history=results['address_cluster_history'],
            transitions_df=results.get('transitions_df')
        )

        if not cohorts:
            logger.error("Failed to define cohorts")
            return

        # Analyze cohort evolution
        logger.info("Analyzing cohort evolution...")
        evolution_results = analyze_cohort_evolution(
            cohorts=cohorts,
            temporal_clusters=cohorts['temporal_clusters']
        )

        if evolution_results is None or 'evolution_df' not in evolution_results:
            logger.error("Failed to analyze cohort evolution")
            return

        # Create visualizations
        logger.info("Creating visualizations...")
        plot_cohort_comparisons(evolution_results['evolution_df'])
        plot_address_evolution(evolution_results, cohorts['temporal_clusters'])
        plot_3d_visualization(evolution_results, cohorts['temporal_clusters'])

        # Plot aggregate evolution for each cohort
        logger.info("Plotting cohort aggregate evolution...")
        for cohort_name in ["High Stability", "Low Stability", "Frequent Appearance", "Entropy Transition"]:
            plot_cohort_aggregate_evolution(cohorts['temporal_clusters'], evolution_results, cohort_name)

        # Create Sankey diagram if transitions data exists
        if 'transitions_df' in results and not results['transitions_df'].empty:
            create_sankey_diagram(results['transitions_df'])

        # Analyze trajectory patterns
        trajectory_summary = analyze_trajectory_patterns(evolution_results['evolution_df'])

        # Generate comprehensive report
        generate_comprehensive_report(results, cohorts, evolution_results['evolution_df'])

        logger.info("Analysis complete!")

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()
