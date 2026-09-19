import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def detect_special_events(df, figures_dir, tables_dir):
    """
    Detects days with exceptionally high traffic using Z-scores based on weekday grouping.
    """
    # 1. Calculate daily total traffic
    daily_traffic = df.groupby(['Date', 'DayOfWeek', 'DayName'])['Vehicles'].sum().reset_index()
    daily_traffic.rename(columns={'Vehicles': 'TotalTraffic'}, inplace=True)
    
    # 2. Group by weekday to calculate mean and standard deviation
    weekday_stats = daily_traffic.groupby('DayOfWeek')['TotalTraffic'].agg(['mean', 'std']).reset_index()
    weekday_stats.rename(columns={'mean': 'WeekdayMean', 'std': 'WeekdayStd'}, inplace=True)
    
    # 3. Merge stats back to daily data
    daily_traffic = pd.merge(daily_traffic, weekday_stats, on='DayOfWeek', how='left')
    
    # 4. Calculate Z-score
    daily_traffic['ZScore'] = (daily_traffic['TotalTraffic'] - daily_traffic['WeekdayMean']) / daily_traffic['WeekdayStd']
    
    # 5. Flag special events based on Z-score threshold
    daily_traffic['IsSpecialEvent'] = daily_traffic['ZScore'] >= config.Z_SCORE_THRESHOLD
    
    # Filter event days
    event_days = daily_traffic[daily_traffic['IsSpecialEvent']].copy()
    
    # Reorder columns as requested
    cols = ['Date', 'DayOfWeek', 'TotalTraffic', 'WeekdayMean', 'WeekdayStd', 'ZScore', 'IsSpecialEvent']
    daily_traffic = daily_traffic[cols]
    event_days = event_days[cols]
    
    # Save table
    event_days.to_csv(os.path.join(tables_dir, 'special_event_days.csv'), index=False)
    print(f"Detected {len(event_days)} candidate special event days.")
    print(f"Saved special event days to {tables_dir}/special_event_days.csv")
    
    # 6. Complete daily traffic time series with anomalies highlighted
    plt.figure(figsize=(15, 6))
    plt.plot(daily_traffic['Date'], daily_traffic['TotalTraffic'], label='Daily Total Traffic', color='gray', alpha=0.7)
    
    plt.scatter(event_days['Date'], event_days['TotalTraffic'], color='red', label=f'Special Event (Z >= {config.Z_SCORE_THRESHOLD})', zorder=5)
    
    plt.title('Daily Total Traffic Network-Wide with Special Events Highlighted')
    plt.xlabel('Date')
    plt.ylabel('Total Vehicles')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'special_events_timeseries.png'))
    plt.close()
    
    return daily_traffic, event_days
