import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os
import sys
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def prepare_ml_data(df):
    """
    Prepares data for machine learning models.
    """
    df = df.copy()
    
    # Sort chronologically
    df.sort_values(by=['Junction', 'DateTime'], inplace=True)
    
    # 1. Base Features
    df['DaysSinceStart'] = (df['DateTime'] - df['DateTime'].min()).dt.days
    
    # 2. Engineered feature: PreviousHourVehicles (per junction)
    df['PreviousHourVehicles'] = df.groupby('Junction')['Vehicles'].shift(1)
    
    # Drop first rows containing NaNs due to shift
    df.dropna(subset=['PreviousHourVehicles'], inplace=True)
    
    # Final sort chronologically for time-series split
    df.sort_values(by='DateTime', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df

def get_mlp_architecture():
    """
    Dynamically determines MLP architecture based on student info.
    """
    num_layers = len(config.FULL_NAME)
    
    # Sum of the last two digits of the roll number
    roll_str = str(config.ROLL_NUMBER).strip()
    if len(roll_str) >= 2 and roll_str[-1].isdigit() and roll_str[-2].isdigit():
        num_neurons = int(roll_str[-1]) + int(roll_str[-2])
    else:
        num_neurons = 10 # Default fallback
    
    # Ensure at least 1 neuron and 1 layer just in case
    num_layers = max(1, num_layers)
    num_neurons = max(1, num_neurons)
    
    hidden_layer_sizes = tuple([num_neurons] * num_layers)
    return hidden_layer_sizes

def train_and_evaluate(model, X_train, y_train, X_test, y_test, preprocessor):
    """
    Trains a model and returns evaluation metrics and predictions.
    """
    # Transform features
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    
    # Train
    model.fit(X_train_proc, y_train)
    
    # Predict
    preds = model.predict(X_test_proc)
    
    # Metrics
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    return model, preds, mae, rmse, r2

def evaluate_models(df, figures_dir, tables_dir, models_dir=None):
    """
    Trains and compares Linear Regression and MLP models, saving models and results.
    """
    if models_dir is None:
        models_dir = config.MODELS_DIR
    os.makedirs(models_dir, exist_ok=True)
    # 1. Prepare Data
    ml_df = prepare_ml_data(df)
    
    # Features
    categorical_features = ['Junction', 'DayOfWeek', 'Month']
    numerical_features_base = ['Hour', 'DaysSinceStart', 'IsWeekend']
    numerical_features_engineered = ['PreviousHourVehicles']
    
    target = 'Vehicles'
    
    # 2. Chronological Split
    n_total = len(ml_df)
    train_end = int(n_total * config.TRAIN_RATIO)
    val_end = int(n_total * (config.TRAIN_RATIO + config.VAL_RATIO))
    
    train_df = ml_df.iloc[:train_end]
    # For simplicity, combine val and test or just use test for final reporting
    # The prompt says evaluate collectively at network level
    test_df = ml_df.iloc[val_end:] 
    
    y_train = train_df[target]
    y_test = test_df[target]
    
    # Base Feature Preprocessor
    preprocessor_base = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features_base),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
    
    # Engineered Feature Preprocessor
    preprocessor_eng = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features_base + numerical_features_engineered),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])

    results = []
    
    # --- MODEL 1: Linear Regression (Base Features) ---
    X_train_base = train_df[categorical_features + numerical_features_base]
    X_test_base = test_df[categorical_features + numerical_features_base]
    
    lr_model = LinearRegression()
    _, lr_preds, mae, rmse, r2 = train_and_evaluate(
        lr_model, X_train_base, y_train, X_test_base, y_test, preprocessor_base)
    
    results.append({'Model': 'Linear Regression', 'Features': 'Base', 'MAE': mae, 'RMSE': rmse, 'R2': r2})
    
    # --- MODEL 2: MLP (Base Features) ---
    hidden_layer_sizes = get_mlp_architecture()
    mlp_base = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, random_state=config.RANDOM_STATE, max_iter=200)
    
    _, mlp_base_preds, mae, rmse, r2 = train_and_evaluate(
        mlp_base, X_train_base, y_train, X_test_base, y_test, preprocessor_base)
    
    results.append({'Model': 'MLP', 'Features': 'Base', 'MAE': mae, 'RMSE': rmse, 'R2': r2})
    
    # --- MODEL 3: MLP (Base + Engineered Features) ---
    X_train_eng = train_df[categorical_features + numerical_features_base + numerical_features_engineered]
    X_test_eng = test_df[categorical_features + numerical_features_base + numerical_features_engineered]
    
    mlp_eng = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, random_state=config.RANDOM_STATE, max_iter=200)
    
    # Train with loss curve capture capability
    X_train_eng_proc = preprocessor_eng.fit_transform(X_train_eng)
    X_test_eng_proc = preprocessor_eng.transform(X_test_eng)
    
    mlp_eng.fit(X_train_eng_proc, y_train)
    mlp_eng_preds = mlp_eng.predict(X_test_eng_proc)
    
    mae = mean_absolute_error(y_test, mlp_eng_preds)
    rmse = np.sqrt(mean_squared_error(y_test, mlp_eng_preds))
    r2 = r2_score(y_test, mlp_eng_preds)
    
    results.append({'Model': 'MLP', 'Features': 'Base + Lag', 'MAE': mae, 'RMSE': rmse, 'R2': r2})
    
    # Save trained models and preprocessors
    joblib.dump(lr_model, os.path.join(models_dir, 'linear_regression_model.joblib'))
    joblib.dump(preprocessor_base, os.path.join(models_dir, 'preprocessor_base.joblib'))
    joblib.dump(mlp_base, os.path.join(models_dir, 'mlp_base_model.joblib'))
    joblib.dump(mlp_eng, os.path.join(models_dir, 'mlp_engineered_model.joblib'))
    joblib.dump(preprocessor_eng, os.path.join(models_dir, 'preprocessor_engineered.joblib'))
    print(f"Trained models and preprocessors saved to {models_dir}")

    # Save Results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(tables_dir, 'model_comparison.csv'), index=False)
    
    # Plot Loss Curve for best MLP
    if hasattr(mlp_eng, 'loss_curve_'):
        plt.figure(figsize=(8, 5))
        plt.plot(mlp_eng.loss_curve_)
        plt.title('MLP Training Loss Curve (Base + Lag)')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'mlp_loss_curve.png'))
        plt.close()
    
    # Actual vs Predicted plot for best model (MLP with lag)
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, mlp_eng_preds, alpha=0.3, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title('Actual vs Predicted Traffic (MLP Base+Lag)')
    plt.xlabel('Actual Vehicles')
    plt.ylabel('Predicted Vehicles')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'actual_vs_predicted.png'))
    plt.close()
    
    # Residual Plot
    residuals = y_test - mlp_eng_preds
    plt.figure(figsize=(8, 5))
    plt.scatter(mlp_eng_preds, residuals, alpha=0.3, color='purple')
    plt.axhline(0, color='r', linestyle='--')
    plt.title('Residual Plot (MLP Base+Lag)')
    plt.xlabel('Predicted Vehicles')
    plt.ylabel('Residuals')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'residual_plot.png'))
    plt.close()
    
    print(f"Model comparison saved to {tables_dir}/model_comparison.csv")
    print(results_df)
