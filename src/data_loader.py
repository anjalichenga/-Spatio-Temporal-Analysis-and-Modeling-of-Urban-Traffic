import pandas as pd
import numpy as np

def load_data(filepath):
    """
    Loads the traffic data, validates columns, converts DateTime,
    and prints a data quality summary.
    """
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    # Validate columns
    expected_columns = ['DateTime', 'Junction', 'Vehicles', 'ID']
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing expected columns: {missing_cols}")
    else:
        print("All expected columns are present.")

    # Convert DateTime
    if 'DateTime' in df.columns:
        df['DateTime'] = pd.to_datetime(df['DateTime'])

    # Basic data quality summary
    print("\n--- Data Quality Summary ---")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    if 'DateTime' in df.columns:
        print(f"Date Range: {df['DateTime'].min()} to {df['DateTime'].max()}")
    if 'Junction' in df.columns:
        print(f"Number of Junctions: {df['Junction'].nunique()}")
        print(f"Junctions: {df['Junction'].unique()}")
    
    missing_values = df.isnull().sum()
    print(f"\nMissing Values per Column:\n{missing_values}")
    
    duplicates = df.duplicated().sum()
    print(f"Duplicate Rows: {duplicates}")
    
    print("-" * 28 + "\n")
    
    return df
