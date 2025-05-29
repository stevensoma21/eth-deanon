import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def plot_clusters(df, cluster_col='dbscan_cluster', save_path=None):
    plt.figure(figsize=(10, 6))
    unique_clusters = sorted(df[cluster_col].unique())
    n_clusters = len(unique_clusters)
    colormap = cm.get_cmap('tab20', n_clusters)
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
        plt.savefig(save_path, dpi=300)
    else:
        plt.savefig(f"{cluster_col}_clusters.png", dpi=300)

    plt.show()

def plot_3d_visualization(cohort_samples, temporal_clusters):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    print("\n[3D Visualization Debugging]")
    print("Available keys in cohort_samples:", list(cohort_samples.keys()))
    
    for key in cohort_samples:
        if isinstance(cohort_samples[key], pd.DataFrame):
            print(f"\n{key} DataFrame shape:", cohort_samples[key].shape)
            print(f"Columns in {key}:", cohort_samples[key].columns.tolist())
            print(f"Sample of {key} data:")
            print(cohort_samples[key].head())
        else:
            print(f"\n{key} is not a DataFrame, type:", type(cohort_samples[key]))

    print("\nTemporal clusters shape:", temporal_clusters.shape)
    print("Temporal clusters columns:", temporal_clusters.columns.tolist())
    print("Sample of temporal clusters:")
    print(temporal_clusters.head())

    try:
        address_to_cohort = {}
        cohort_mapping = {
            'high_stability_df': 'High Stability',
            'low_stability_df': 'Low Stability',
            'frequent_appearance_df': 'Frequent Appearance',
            'entropy_transition_df': 'Entropy Transition'
        }
        
        for key, cohort_name in cohort_mapping.items():
            if key in cohort_samples and not cohort_samples[key].empty:
                print(f"\nProcessing {cohort_name} cohort...")
                print(f"Number of addresses in {key}:", len(cohort_samples[key]))
                for addr in cohort_samples[key]['address']:
                    address_to_cohort[addr] = cohort_name

        print("\nFinal address_to_cohort mapping:")
        print("Number of addresses mapped:", len(address_to_cohort))
        print("Sample of mapped addresses:", list(address_to_cohort.items())[:5])

        if not address_to_cohort:
            print("No addresses found in any cohort")
            return

        sampled_addresses = list(address_to_cohort.keys())
        print("\nNumber of sampled addresses:", len(sampled_addresses))
        
        sample_temporal_clusters = temporal_clusters[temporal_clusters['address'].isin(sampled_addresses)].copy()
        print("\nSample temporal clusters shape after filtering:", sample_temporal_clusters.shape)
        
        sample_temporal_clusters['cohort'] = sample_temporal_clusters['address'].map(address_to_cohort)
        print("\nCohort distribution in sample:")
        print(sample_temporal_clusters['cohort'].value_counts())

        time_to_num = {t: i for i, t in enumerate(sorted(sample_temporal_clusters['window_start'].unique()))}
        sample_temporal_clusters['time_num'] = sample_temporal_clusters['window_start'].map(time_to_num)

        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')

        cohort_colors = {
            'High Stability': 'blue',
            'Low Stability': 'red',
            'Frequent Appearance': 'green',
            'Entropy Transition': 'purple'
        }

        for cohort, color in cohort_colors.items():
            data = sample_temporal_clusters[sample_temporal_clusters['cohort'] == cohort]
            if not data.empty:
                print(f"\nPlotting {cohort} cohort:")
                print(f"Number of points to plot: {len(data)}")
                ax.scatter(data['degree'], data['entropy'], data['time_num'], 
                          c=color, label=cohort, alpha=0.7, s=50)

        ax.set_xlabel('Degree')
        ax.set_ylabel('Entropy')
        ax.set_zlabel('Time Window')
        plt.title('3D Temporal Visualization by Cohort')
        ax.legend()
        ax.view_init(elev=30, azim=45)

        z_ticks = list(time_to_num.values())
        z_labels = [t.strftime('%m-%d') for t in sorted(time_to_num.keys())]
        ax.set_zticks(z_ticks)
        ax.set_zticklabels(z_labels)

        plt.tight_layout()
        plt.savefig("3d_cohort_visualization.png", dpi=300)
        plt.show()
    except Exception as e:
        print(f"Error creating 3D visualization: {e}")
        import traceback
        traceback.print_exc()

def plot_cohort_comparisons(evolution_df):
    print("Creating cohort comparison visualizations…")

    plt.figure(figsize=(12,8))
    if 'degree_change_log' in evolution_df.columns:
        sns.boxplot(x='cohort', y='degree_change_log', data=evolution_df)
        plt.ylabel('signed log(1 + |Δdegree|)')
        plt.title('Signed‐Log Degree Change by Cohort')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("cohort_degree_change_log.png", dpi=300)
        plt.show()

    plt.figure(figsize=(12, 8))
    sns.boxplot(x='cohort', y='entropy_change', data=evolution_df)
    plt.title('Entropy Change by Cohort')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("cohort_entropy_change.png", dpi=300)
    plt.show()

    plt.figure(figsize=(12, 8))
    for cohort in evolution_df['cohort'].unique():
        subset = evolution_df[evolution_df['cohort'] == cohort]
        plt.scatter(subset['start_entropy'], subset['end_entropy'], label=cohort, alpha=0.7, s=80)
    max_val = max(evolution_df['start_entropy'].max(), evolution_df['end_entropy'].max()) + 0.5
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
    plt.xlabel('Initial Entropy')
    plt.ylabel('Final Entropy')
    plt.title('Entropy Evolution by Cohort')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("cohort_entropy_evolution.png", dpi=300)
    plt.show()


def plot_address_evolution(cohort_samples, temporal_clusters):
    cohort_mapping = {
        'High Stability': 'high_stability_df',
        'Low Stability': 'low_stability_df',
        'Frequent Appearance': 'frequent_appearance_df',
        'Entropy Transition': 'entropy_transition_df'
    }

    for cohort_name, df_key in cohort_mapping.items():
        if df_key not in cohort_samples or cohort_samples[df_key].empty:
            print(f"No addresses found for cohort: {cohort_name}")
            continue

        plot_addresses = cohort_samples[df_key]['address'].head(5).tolist()
        plt.figure(figsize=(14, 7))

        for i, addr in enumerate(plot_addresses):
            addr_data = temporal_clusters[temporal_clusters['address'] == addr].sort_values('window_start')
            if len(addr_data) > 1:
                plt.subplot(1, 2, 1)
                plt.plot(addr_data['window_start'], addr_data['entropy'], 'o-', label=f"Addr {i+1}")
                plt.subplot(1, 2, 2)
                plt.plot(addr_data['window_start'], addr_data['degree'], 'o-', label=f"Addr {i+1}")

        plt.subplot(1, 2, 1)
        plt.title(f'{cohort_name} Cohort: Entropy Evolution')
        plt.xlabel('Time')
        plt.ylabel('Entropy')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.title(f'{cohort_name} Cohort: Degree Evolution')
        plt.xlabel('Time')
        plt.ylabel('Degree')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(f"{cohort_name.lower().replace(' ', '_')}_evolution.png", dpi=300)
        plt.show()


def plot_cohort_aggregate_evolution(temporal_clusters, cohort_samples, cohort_name):
    cohort_mapping = {
        "High Stability": "high_stability_df",
        "Low Stability": "low_stability_df",
        "Frequent Appearance": "frequent_appearance_df",
        "Entropy Transition": "entropy_transition_df"
    }

    df_key = cohort_mapping.get(cohort_name)
    if not df_key or df_key not in cohort_samples or cohort_samples[df_key].empty:
        print(f"No addresses found for cohort: {cohort_name}")
        return

    addresses = cohort_samples[df_key]['address'].tolist()
    cohort_data = temporal_clusters[temporal_clusters['address'].isin(addresses)]
    if cohort_data.empty:
        print(f"No data found for {cohort_name} cohort.")
        return

    grouped = cohort_data.groupby('window_start')
    mean_entropy = grouped['entropy'].mean()
    std_entropy = grouped['entropy'].std()
    mean_degree = grouped['degree'].mean()
    std_degree = grouped['degree'].std()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)

    axes[0].plot(mean_entropy.index, mean_entropy, marker='o', label='Mean Entropy')
    axes[0].fill_between(mean_entropy.index, mean_entropy - std_entropy, mean_entropy + std_entropy, alpha=0.3)
    axes[0].set_title(f'{cohort_name}: Entropy Evolution')
    axes[0].set_ylabel('Entropy')
    axes[0].set_xlabel('Time')
    axes[0].grid(True)

    axes[1].plot(mean_degree.index, mean_degree, marker='o', label='Mean Degree')
    axes[1].fill_between(mean_degree.index, mean_degree - std_degree, mean_degree + std_degree, alpha=0.3)
    axes[1].set_title(f'{cohort_name}: Degree Evolution')
    axes[1].set_ylabel('Degree')
    axes[1].set_xlabel('Time')
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(f"{cohort_name.lower().replace(' ', '_')}_aggregate_evolution.png", dpi=300)
    plt.show()
