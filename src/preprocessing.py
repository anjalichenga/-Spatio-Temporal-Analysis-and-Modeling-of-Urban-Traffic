import pandas as pd

def create_time_features(df):
    """
    Creates time-based features from the DateTime column.
    """
    df = df.copy()
    if 'DateTime' not in df.columns:
        print("DateTime column not found for feature extraction.")
        return df

    # Create new features
    df['Date'] = df['DateTime'].dt.date
    df['Hour'] = df['DateTime'].dt.hour
    df['DayOfWeek'] = df['DateTime'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['DayName'] = df['DateTime'].dt.day_name()
    df['Month'] = df['DateTime'].dt.month
    df['Year'] = df['DateTime'].dt.year
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

    print("\n--- Preprocessing Summary ---")
    print("New features created: Date, Hour, DayOfWeek, DayName, Month, Year, IsWeekend")
    print("\nDescriptive Statistics (Vehicles):")
    if 'Vehicles' in df.columns:
        print(df['Vehicles'].describe())
    
    return df
