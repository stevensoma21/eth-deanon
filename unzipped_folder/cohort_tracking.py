import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

def create_cohort_tracking_df(cohorts: Dict, temporal_clusters: pd.DataFrame) -> pd.DataFrame:
    """
    Create a DataFrame tracking addresses across cohorts and time.
    
    Args:
        cohorts: Dictionary containing cohort information
        temporal_clusters: DataFrame with temporal clustering results
        
    Returns:
        DataFrame with columns: [address, window_start, cohort, degree, entropy]
    """
    # Get all unique addresses from each cohort
    cohort_addresses = {
        'High Stability': set(cohorts['high_stability_df']['address']),
        'Low Stability': set(cohorts['low_stability_df']['address']),
        'Frequent Appearance': set(cohorts['frequent_appearance_df']['address']),
        'Entropy Transition': set(cohorts['entropy_transition_df']['address'])
    }
    
    # Create a mapping of addresses to their cohorts
    address_to_cohort = {}
    for cohort_name, addresses in cohort_addresses.items():
        for addr in addresses:
            if addr not in address_to_cohort:
                address_to_cohort[addr] = []
            address_to_cohort[addr].append(cohort_name)
    
    # Create the tracking DataFrame
    tracking_data = []
    
    for addr, cohort_list in address_to_cohort.items():
        # Get all temporal data for this address
        addr_data = temporal_clusters[temporal_clusters['address'] == addr]
        
        for _, row in addr_data.iterrows():
            for cohort in cohort_list:
                tracking_data.append({
                    'address': addr,
                    'window_start': row['window_start'],
                    'cohort': cohort,
                    'degree': row['degree'],
                    'entropy': row['entropy']
                })
    
    tracking_df = pd.DataFrame(tracking_data)
    
    # Sort by address and time
    tracking_df = tracking_df.sort_values(['address', 'window_start'])
    
    # Add a column indicating if address is in multiple cohorts
    tracking_df['in_multiple_cohorts'] = tracking_df.groupby('address')['cohort'].transform('nunique') > 1
    
    return tracking_df

def analyze_cohort_overlap(tracking_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze overlap between cohorts.
    
    Args:
        tracking_df: DataFrame from create_cohort_tracking_df
        
    Returns:
        DataFrame showing cohort overlap statistics
    """
    # Get unique addresses in each cohort
    cohort_sizes = tracking_df.groupby('cohort')['address'].nunique()
    
    # Create overlap matrix
    cohorts = tracking_df['cohort'].unique()
    overlap_matrix = pd.DataFrame(index=cohorts, columns=cohorts)
    
    for c1 in cohorts:
        for c2 in cohorts:
            if c1 == c2:
                overlap_matrix.loc[c1, c2] = cohort_sizes[c1]
            else:
                overlap = len(set(tracking_df[tracking_df['cohort'] == c1]['address']) & 
                            set(tracking_df[tracking_df['cohort'] == c2]['address']))
                overlap_matrix.loc[c1, c2] = overlap
    
    return overlap_matrix

def get_address_timeline(tracking_df: pd.DataFrame, address: str) -> pd.DataFrame:
    """
    Get the complete timeline of an address across cohorts.
    
    Args:
        tracking_df: DataFrame from create_cohort_tracking_df
        address: Address to analyze
        
    Returns:
        DataFrame with the address's timeline
    """
    return tracking_df[tracking_df['address'] == address].sort_values('window_start')

def export_cohort_tracking(tracking_df: pd.DataFrame, output_path: str = 'cohort_tracking.csv'):
    """
    Export the cohort tracking data to CSV.
    
    Args:
        tracking_df: DataFrame from create_cohort_tracking_df
        output_path: Path to save the CSV file
    """
    tracking_df.to_csv(output_path, index=False)
    logger.info(f"Exported cohort tracking data to {output_path}")

def main():
    """
    Example usage of the cohort tracking functionality.
    """
    # This would be called from your main analysis script
    # tracking_df = create_cohort_tracking_df(cohorts, temporal_clusters)
    # overlap_matrix = analyze_cohort_overlap(tracking_df)
    # export_cohort_tracking(tracking_df)
    pass

if __name__ == "__main__":
    main() 