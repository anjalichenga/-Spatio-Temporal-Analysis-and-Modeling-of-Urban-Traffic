# Spatio-Temporal Analysis and Machine Learning Modeling of Urban Traffic

## Project Objective
This project provides a reproducible, academic data-science pipeline that analyzes hourly traffic data from four urban junctions. It visualizes traffic patterns, identifies peak periods at intersection and network levels, detects special event anomalies, clusters time-series patterns using Dynamic Time Warping (DTW), and estimates future traffic using a dynamic Multi-Layer Perceptron (MLP) architecture.

## Dataset Description
The dataset consists of hourly aggregated vehicle counts from four major intersections.
Expected attributes include `DateTime`, `Junction`, `Vehicles`, and `ID`.
The data is parsed to ensure reliability and robustness against missing or duplicate values.

## Methodology
The pipeline adheres to the following steps:
1. **Data Preprocessing**: Checks data quality and extracts granular temporal features.
2. **Spatio-Temporal Visualization**: Creates heatmaps of traffic flow (Hour vs. DayOfWeek) and hourly profiles.
3. **Peak Analysis**: Identifies typical morning (05:00-11:00) and evening (15:00-22:00) peaks at both intersection and network levels.
4. **Special Event Detection**: Uses a Z-score statistical threshold (default: >= 2) on weekday-grouped daily volumes to flag anomalous candidate event days.
5. **DTW Analysis**: Computes similarity across 24-hour daily profiles and groups days using hierarchical clustering.
6. **Machine Learning**: Trains a Linear Regression and two MLP configurations. The MLP architecture dynamically scales based on student credentials configurable in `config.py`.

## Project Structure
```
urban-traffic-analysis/
├── data/
│   └── traffic.csv
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── visualization.py
│   ├── peak_analysis.py
│   ├── event_detection.py
│   ├── dtw_analysis.py
│   └── models.py
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── models/
├── notebooks/
│   └── traffic_analysis.ipynb
├── config.py
├── main.py
├── requirements.txt
├── README.md
└── report/
    └── report_outline.md
```

## Installation Instructions
1. Ensure Python 3.8+ is installed.
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run
1. Place the dataset `traffic.csv` inside the `data/` folder.
2. Edit `config.py` to insert your `FULL_NAME` and `ROLL_NUMBER`.
3. Execute the pipeline:
   ```bash
   python main.py
   ```
4. Review the generated tables and figures in the `outputs/` folder.
5. An interactive notebook version is available in `notebooks/traffic_analysis.ipynb`.

## Assumptions
- The network's behavior is approximated by summing the volumes across all intersections.
- The machine learning evaluation interprets the dataset collectively at the network level by training across all individual `Junction` instances simultaneously.
- Missing records are safely ignored, but the data loader reports their presence.

## Expected Outputs
- Heatmap and Profile plots for each junction (`outputs/figures/`)
- Summary statistics, peak periods, special event days (`outputs/tables/`)
- A model comparison report highlighting MAE, RMSE, and R2 trade-offs.
