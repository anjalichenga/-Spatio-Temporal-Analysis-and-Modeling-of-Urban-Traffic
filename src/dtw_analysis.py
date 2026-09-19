import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import os

def simple_dtw(s1, s2):
    """
    Computes a basic Dynamic Time Warping (DTW) distance between two 1D arrays.
    """
    n, m = len(s1), len(s2)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(s1[i - 1] - s2[j - 1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j],    # insertion
                                          dtw_matrix[i, j - 1],    # deletion
                                          dtw_matrix[i - 1, j - 1]) # match
    return dtw_matrix[n, m]

def create_daily_profiles(df):
    """
    Constructs a 24-hour normalized traffic profile for each day.
    """
    network_hourly = df.groupby(['Date', 'Hour'])['Vehicles'].sum().reset_index()
    
    # Pivot to get Date as index and Hour 0-23 as columns
    profiles = network_hourly.pivot(index='Date', columns='Hour', values='Vehicles')
    
    # Drop any days that don't have full 24 hours of data
    profiles = profiles.dropna()
    
    # Normalize profiles (min-max normalization per day)
    profiles_min = profiles.min(axis=1)
    profiles_max = profiles.max(axis=1)
    # Avoid division by zero
    diff = profiles_max - profiles_min
    diff[diff == 0] = 1
    profiles_norm = profiles.subtract(profiles_min, axis=0).divide(diff, axis=0)
    
    return profiles_norm

def calculate_dtw_matrix(profiles):
    """
    Calculates the DTW distance matrix for the daily profiles.
    """
    n_days = len(profiles)
    dates = profiles.index.values
    data = profiles.values
    
    dist_matrix = np.zeros((n_days, n_days))
    
    print("Calculating DTW distance matrix (this may take a moment)...")
    for i in range(n_days):
        for j in range(i + 1, n_days):
            dist = simple_dtw(data[i], data[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
            
    return dist_matrix, dates

def run_dtw_analysis(df, event_days, figures_dir, tables_dir):
    """
    Executes the DTW bonus analysis.
    """
    profiles = create_daily_profiles(df)
    if profiles.empty:
        print("Not enough complete daily profiles for DTW.")
        return
        
    dist_matrix, dates = calculate_dtw_matrix(profiles)
    
    # Convert square distance matrix to condensed distance matrix for scipy linkage
    from scipy.spatial.distance import squareform
    condensed_dist = squareform(dist_matrix, checks=False)
    
    # Hierarchical Clustering
    Z = linkage(condensed_dist, method='ward')
    
    # Determine a cutoff for 3 clusters as a simple approach
    max_d = 0.5 * max(Z[:, 2])
    clusters = fcluster(Z, 3, criterion='maxclust')
    
    # Save clustering results
    results_df = pd.DataFrame({'Date': dates, 'Cluster': clusters})
    results_df.to_csv(os.path.join(tables_dir, 'dtw_clusters.csv'), index=False)
    
    # Plot Dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram(Z, truncate_mode='lastp', p=30, show_leaf_counts=True)
    plt.title('Hierarchical Clustering Dendrogram (DTW distances)')
    plt.xlabel('Cluster size')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'dtw_dendrogram.png'))
    plt.close()
    
    # Compare with special events
    if event_days is not None and not event_days.empty:
        # Convert Dates to same type if needed
        results_df['Date'] = pd.to_datetime(results_df['Date']).dt.date
        event_days_copy = event_days.copy()
        event_days_copy['Date'] = pd.to_datetime(event_days_copy['Date']).dt.date
        
        comparison = pd.merge(event_days_copy[['Date', 'TotalTraffic', 'IsSpecialEvent']], 
                              results_df, on='Date', how='inner')
        comparison.to_csv(os.path.join(tables_dir, 'dtw_special_events_comparison.csv'), index=False)
        print(f"Saved DTW results and comparison to {tables_dir}")
