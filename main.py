import os
import sys

import config
from src.data_loader import load_data
from src.preprocessing import create_time_features
from src.visualization import plot_junction_heatmaps
from src.peak_analysis import analyze_junction_peaks, analyze_network_peaks
from src.event_detection import detect_special_events
from src.dtw_analysis import run_dtw_analysis
from src.models import evaluate_models

def main():
    print("="*50)
    print("URBAN TRAFFIC ANALYSIS PIPELINE")
    print("="*50)
    
    # Ensure outputs directories exist
    os.makedirs(config.FIGURES_DIR, exist_ok=True)
    os.makedirs(config.TABLES_DIR, exist_ok=True)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    # 1. Load Data
    df = load_data(config.DATA_FILE)
    if df is None:
        print("Pipeline aborted due to data loading failure.")
        return
        
    # 2. Preprocessing
    df = create_time_features(df)
    
    # 3. Spatio-Temporal Visualization (Q1)
    print("\n--- Running Q1: Spatio-Temporal Visualization ---")
    plot_junction_heatmaps(df, config.FIGURES_DIR, config.TABLES_DIR)
    
    # 4. Intersection-Level Peak Analysis (Q2)
    print("\n--- Running Q2: Intersection Peak Analysis ---")
    analyze_junction_peaks(df, config.FIGURES_DIR, config.TABLES_DIR)
    
    # 5. Network-Level Peak Analysis (Q3)
    print("\n--- Running Q3: Network Peak Analysis ---")
    analyze_network_peaks(df, config.FIGURES_DIR, config.TABLES_DIR)
    
    # 6. Special Event Detection (Q4)
    print("\n--- Running Q4: Special Event Detection ---")
    daily_traffic, event_days = detect_special_events(df, config.FIGURES_DIR, config.TABLES_DIR)
    
    # 7. DTW Analysis (Q5 - Bonus)
    print("\n--- Running Q5: DTW Time-Series Similarity Analysis ---")
    run_dtw_analysis(df, event_days, config.FIGURES_DIR, config.TABLES_DIR)
    
    # 8. Machine Learning (Q6)
    print("\n--- Running Q6: Machine Learning Estimation ---")
    evaluate_models(df, config.FIGURES_DIR, config.TABLES_DIR, config.MODELS_DIR)
    
    print("\n" + "="*50)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*50)
    print(f"Results are saved in {config.OUTPUT_DIR}")

if __name__ == "__main__":
    main()
