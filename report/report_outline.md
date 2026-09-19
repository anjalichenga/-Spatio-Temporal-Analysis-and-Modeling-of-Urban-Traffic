# Spatio-Temporal Analysis and Machine Learning Modeling of Urban Traffic

## 1. Introduction
- **Project Objective**: To analyze urban traffic patterns across multiple junctions and develop predictive machine learning models to estimate future traffic.
- **Scope**: Covers spatio-temporal visualization, intersection and network-level peak analysis, anomaly detection, DTW time-series clustering, and MLP-based traffic estimation.

## 2. Dataset Description
- **Source Data**: Contains hourly traffic counts for four major junctions.
- **Attributes**: `DateTime`, `Junction`, `Vehicles`, `ID`.
- **Characteristics**: Evaluated total rows, missing values, date ranges, and duplicates to verify data integrity.

## 3. Data Preprocessing
- **Transformations**: Extracted date and time features (`Date`, `Hour`, `DayOfWeek`, `DayName`, `Month`, `Year`, `IsWeekend`) from `DateTime`.
- **Validation**: Ensured chronologically sorted data and confirmed no critical data leakage before feature engineering.

## 4. Q1 Spatio-Temporal Analysis
- **Methodology**: Created pivot tables representing average hourly traffic across days of the week. Generated heatmaps and hourly profile graphs for each junction.
- **Results**: (Insert discussion on observed daily recurring patterns and variation between weekdays vs weekends).

## 5. Q2 Intersection Peak Analysis
- **Methodology**: Evaluated morning (05:00-11:00) and evening (15:00-22:00) windows to find the highest average hourly traffic per junction.
- **Results**: Reference `outputs/tables/junction_peak_hours.csv`. (Discuss the variance of peak hours across different junctions).

## 6. Q3 Network Peak Analysis
- **Methodology**: Aggregated all junction traffic by date and hour to determine system-wide morning and evening peak hours.
- **Results**: Reference `outputs/tables/network_peak_hours.csv`. (Compare network peaks with individual junction peaks).

## 7. Q4 Special Event Detection
- **Methodology**: Grouped daily traffic by weekday and computed the Z-score for each day relative to its group. Configured threshold at Z >= 2.0.
- **Results**: Days exceeding the threshold are labeled candidate special-event days (`outputs/tables/special_event_days.csv`). (Discuss any visually detected anomalies).

## 8. Q5 DTW Analysis
- **Methodology**: Created 24-hour normalized profiles for each day. Calculated pairwise Dynamic Time Warping (DTW) distance and applied hierarchical clustering (Ward's method).
- **Results**: Extracted traffic pattern clusters and compared them to the Z-score detected special event days.

## 9. Q6 Machine Learning Estimation
- **Architecture**: Dynamically designed Multi-Layer Perceptron (MLP). Hidden layers = length of student name. Neurons per layer = sum of last two digits of roll number.
- **Features Used**: Base (`Junction`, `Hour`, `DayOfWeek`, `Month`, `DaysSinceStart`, `IsWeekend`) and Engineered (`PreviousHourVehicles`).
- **Data Splitting**: 70% Train, 15% Validation, 15% Test. (Chronological split).

## 10. Model Comparison
- **Methodology**: Compared Linear Regression vs. MLP (Base features) vs. MLP (Base + Engineered features).
- **Evaluation Metrics**: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² Score.
- **Results**: Reference `outputs/tables/model_comparison.csv`.

## 11. Assumptions
- Traffic readings from junctions can be evaluated collectively (stacked) for network estimation rather than treating the entire network as a single spatially unified multivariate output.
- Missing values created by shifting for the lag feature can be dropped safely.
- Candidate special-event days represent anomalies, not necessarily validated real-world events.

## 12. Limitations
- Lag features restrict prediction horizons (e.g., cannot effectively predict a week out without iterative forecasting).
- Simple Z-score anomaly detection assumes normal distribution of traffic per weekday.
- Naive DTW scaling may perform poorly on significantly large datasets.

## 13. Conclusion
- Summary of traffic characteristics, optimal models, and potential future improvements.
