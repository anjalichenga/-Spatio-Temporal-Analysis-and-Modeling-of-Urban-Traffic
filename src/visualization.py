import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_heatmap_data(df, junction):
    """
    Creates a pivot table for the given junction for heatmap visualization.
    Rows: Hour (0-23)
    Cols: DayOfWeek (Monday-Sunday)
    Values: Mean Vehicles
    """
    junction_data = df[df['Junction'] == junction]
    
    # Calculate mean vehicles by DayOfWeek and Hour
    grouped = junction_data.groupby(['DayName', 'Hour'])['Vehicles'].mean().reset_index()
    
    # Pivot table
    pivot = grouped.pivot(index='Hour', columns='DayName', values='Vehicles')
    
    # Ensure columns are ordered from Monday to Sunday
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot = pivot.reindex(columns=days_order)
    
    return pivot

def plot_junction_heatmaps(df, figures_dir, tables_dir):
    """
    Generates and saves spatio-temporal heatmaps, summary stats, and hourly profiles 
    for each junction independently.
    """
    junctions = df['Junction'].unique()
    
    # Ensure directories exist
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    
    summary_stats = []

    for j in sorted(junctions):
        # 1. Pivot table for heatmap
        pivot = create_heatmap_data(df, j)
        
        # 2. Plot Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot, cmap='coolwarm', cbar_kws={'label': 'Average Vehicles'})
        plt.title(f'Average Vehicles - Junction {j}')
        plt.ylabel('Hour')
        plt.xlabel('Weekday')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f'heatmap_junction_{j}.png'))
        plt.close()
        
        # 3. Create hourly traffic profile plot
        junction_data = df[df['Junction'] == j]
        hourly_avg = junction_data.groupby('Hour')['Vehicles'].mean().reset_index()
        
        plt.figure(figsize=(8, 5))
        plt.plot(hourly_avg['Hour'], hourly_avg['Vehicles'], marker='o', linestyle='-', color='b')
        plt.title(f'Hourly Traffic Profile - Junction {j}')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Vehicles')
        plt.grid(True)
        plt.xticks(range(0, 24))
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f'hourly_profile_junction_{j}.png'))
        plt.close()
        
        # 4. Summary statistics
        stats = junction_data['Vehicles'].describe()
        stats['Junction'] = j
        summary_stats.append(stats)

    # Save summary statistics for all junctions
    summary_df = pd.DataFrame(summary_stats)
    cols = ['Junction'] + [c for c in summary_df.columns if c != 'Junction']
    summary_df = summary_df[cols]
    summary_df.to_csv(os.path.join(tables_dir, 'junction_summary_stats.csv'), index=False)
    
    print(f"Saved {len(junctions)} heatmaps and profiles to {figures_dir}")
    print(f"Saved junction summary statistics to {tables_dir}")
