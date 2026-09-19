import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def find_peaks(hourly_avg):
    """
    Finds the morning and evening peak hours and their averages.
    """
    morning_data = hourly_avg[(hourly_avg['Hour'] >= config.MORNING_PEAK_START) & 
                              (hourly_avg['Hour'] <= config.MORNING_PEAK_END)]
    evening_data = hourly_avg[(hourly_avg['Hour'] >= config.EVENING_PEAK_START) & 
                              (hourly_avg['Hour'] <= config.EVENING_PEAK_END)]
    
    m_peak = morning_data.loc[morning_data['Vehicles'].idxmax()]
    e_peak = evening_data.loc[evening_data['Vehicles'].idxmax()]
    
    return int(m_peak['Hour']), m_peak['Vehicles'], int(e_peak['Hour']), e_peak['Vehicles']

def analyze_junction_peaks(df, figures_dir, tables_dir):
    """
    Analyzes intersection-level peak periods.
    """
    junctions = df['Junction'].unique()
    peak_results = []
    
    for j in sorted(junctions):
        junction_data = df[df['Junction'] == j]
        hourly_avg = junction_data.groupby('Hour')['Vehicles'].mean().reset_index()
        
        m_hour, m_avg, e_hour, e_avg = find_peaks(hourly_avg)
        
        peak_results.append({
            'Junction': j,
            'Morning Peak Hour': m_hour,
            'Morning Average': m_avg,
            'Evening Peak Hour': e_hour,
            'Evening Average': e_avg
        })
        
        # Plot with peaks highlighted
        plt.figure(figsize=(8, 5))
        plt.plot(hourly_avg['Hour'], hourly_avg['Vehicles'], marker='o', label='Hourly Average')
        plt.axvline(x=m_hour, color='r', linestyle='--', label=f'Morning Peak ({m_hour}:00)')
        plt.axvline(x=e_hour, color='g', linestyle='--', label=f'Evening Peak ({e_hour}:00)')
        plt.title(f'Peak Analysis - Junction {j}')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Vehicles')
        plt.xticks(range(0, 24))
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f'peak_profile_junction_{j}.png'))
        plt.close()
        
    peaks_df = pd.DataFrame(peak_results)
    peaks_df.to_csv(os.path.join(tables_dir, 'junction_peak_hours.csv'), index=False)
    print(f"Junction peak hours saved to {tables_dir}/junction_peak_hours.csv")
    return peaks_df

def analyze_network_peaks(df, figures_dir, tables_dir):
    """
    Analyzes network-level peak periods (aggregating all junctions).
    """
    # Aggregate traffic across all junctions by taking the sum per DateTime
    # Wait, the assignment says "aggregate traffic across all junctions". 
    # That means for a given hour in the network, we sum the vehicles of all junctions.
    network_hourly = df.groupby(['DateTime', 'Hour'])['Vehicles'].sum().reset_index()
    
    # Calculate average traffic by hour at network level
    net_hourly_avg = network_hourly.groupby('Hour')['Vehicles'].mean().reset_index()
    
    m_hour, m_avg, e_hour, e_avg = find_peaks(net_hourly_avg)
    
    network_peaks = pd.DataFrame([{
        'Network': 'All Junctions',
        'Morning Peak Hour': m_hour,
        'Morning Average': m_avg,
        'Evening Peak Hour': e_hour,
        'Evening Average': e_avg
    }])
    
    network_peaks.to_csv(os.path.join(tables_dir, 'network_peak_hours.csv'), index=False)
    
    plt.figure(figsize=(8, 5))
    plt.plot(net_hourly_avg['Hour'], net_hourly_avg['Vehicles'], marker='o', color='purple', label='Network Average')
    plt.axvline(x=m_hour, color='r', linestyle='--', label=f'Morning Peak ({m_hour}:00)')
    plt.axvline(x=e_hour, color='g', linestyle='--', label=f'Evening Peak ({e_hour}:00)')
    plt.title('Network-Level Peak Analysis')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Total Vehicles')
    plt.xticks(range(0, 24))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'network_peak_profile.png'))
    plt.close()
    
    print(f"Network peak hours saved to {tables_dir}/network_peak_hours.csv")
    return network_peaks
