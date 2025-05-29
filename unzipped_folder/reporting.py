# reporting.py
# === reporting.py ===
import pandas as pd

def analyze_trajectory_patterns(evolution_df):
    """
    Summary stats for behavior trajectory across cohorts.
    """
    trajectory_summary = evolution_df.groupby('cohort').agg({
        'degree_change': ['mean', 'median', 'std'],
        'entropy_change': ['mean', 'median', 'std'],
        'cluster_changes': ['mean', 'median', 'max']
    }).reset_index()
    return trajectory_summary


def generate_comprehensive_report(results, cohorts, evolution_df):
    """
    Print formatted summary of key analysis insights.
    """
    start_date = results['temporal_clusters']['window_start'].min()
    end_date = results['temporal_clusters']['window_start'].max()

    print(f"""
=== COMPREHENSIVE TEMPORAL CLUSTER ANALYSIS REPORT ===
Analysis period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}
Time windows: {len(results['cluster_evolution'])}
Unique addresses tracked: {len(results['address_cluster_history'])}
Addresses in multiple windows: {len(results['multi_window_addresses'])}

=== COHORT DISTRIBUTION ===
High stability: {len(cohorts['high_stability'])}
Low stability: {len(cohorts['low_stability'])}
Frequent appearances: {len(cohorts['frequent_appearance'])}
Entropy transitions: {len(cohorts['entropy_transition_addresses'])}

=== BEHAVIOR TRAJECTORY STATS ===
""")

    for label in ['High Stability', 'Low Stability', 'Frequent Appearance', 'Entropy Transition']:
        print(f"-- {label} --")
        df = evolution_df[evolution_df['cohort'] == label]
        print(f"Avg ΔDegree: {df['degree_change'].mean():.2f}, Avg ΔEntropy: {df['entropy_change'].mean():.2f}, Cluster changes: {df['cluster_changes'].mean():.2f}\n")

    print("""
=== INTERPRETATION ===
- High stability addresses show minimal behavioral drift—strong candidates for persistent tagging.
- Entropy-transitioning addresses show shape-shifting behavior that may indicate role changes or obfuscation.
- Degree patterns and entropy arcs solidify over time—supporting behavior-based deanonymization methods.
""")
