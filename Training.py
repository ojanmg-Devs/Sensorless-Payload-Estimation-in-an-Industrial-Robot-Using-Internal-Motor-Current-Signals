# -*- coding: utf-8 -*-
"""
SVM Baseline Re-run & Systematic Hyperparameter Optimization
Addresses Reviewer Comment on SVR Feature Scaling and Parameter Tuning.
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =====================================================
# 1. Feature Extraction & Dataset Building
# =====================================================
def extract_features_from_file(filepath):
    """Extract statistical features from one CSV file."""
    df = pd.read_csv(filepath, sep=";")
    
    if "Sample" in df.columns:
        df = df.drop(columns=["Sample"])
    
    df = df.select_dtypes(include=[np.number])
    
    features = {}
    for col in df.columns:
        values = df[col].dropna().values
        features[f"{col}_mean"] = np.mean(values)
        features[f"{col}_std"] = np.std(values)
        features[f"{col}_min"] = np.min(values)
        features[f"{col}_max"] = np.max(values)
        features[f"{col}_median"] = np.median(values)
        features[f"{col}_energy"] = np.sum(values ** 2) / len(values)
    return features

def build_dataset(folder):
    pattern = re.compile(r"collector_(\d+)_(\d+)\.csv")
    rows = []
    for fname in os.listdir(folder):
        match = pattern.match(fname)
        if not match:
            continue
        weight = int(match.group(1))
        run = int(match.group(2))
        filepath = os.path.join(folder, fname)
        feats = extract_features_from_file(filepath)
        feats["Weight"] = weight
        feats["Run"] = run
        feats["File"] = fname
        rows.append(feats)
    return pd.DataFrame(rows)


# =====================================================
# 2. Main Optimization & Evaluation Script
# =====================================================
def main():
    folder = "./"  # Set path to CSV data files
    df = build_dataset(folder)
    print(f"Loaded dataset with shape: {df.shape}")

    # Dataset splitting: Train (Runs 0, 1, 3) / Test (Run 2)
    train_df = df[df["Run"].isin([0, 1, 3])].copy()
    test_df = df[df["Run"].isin([2])].copy()

    X_train_raw = train_df.drop(columns=["Weight", "Run", "File"]).values
    y_train_raw = train_df["Weight"].values.reshape(-1, 1)
    X_test_raw  = test_df.drop(columns=["Weight", "Run", "File"]).values
    y_test_raw  = test_df["Weight"].values.reshape(-1, 1)

    # -------------------------------------------------
    # Step A: Z-score Standardization (Identical to MLP)
    # -------------------------------------------------
    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train_raw)
    X_test = x_scaler.transform(X_test_raw)

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train_raw).ravel()
    y_test = y_scaler.transform(y_test_raw).ravel()

    # -------------------------------------------------
    # Step B: Systematic Hyperparameter Grid Search
    # -------------------------------------------------
    param_grid = {
        'kernel': ['rbf', 'linear', 'poly'],
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'epsilon': [0.001, 0.01, 0.1, 0.2]
    }

    print("\nStarting SVR Hyperparameter Grid Search...")
    grid_search = GridSearchCV(
        estimator=SVR(),
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=5,
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)

    best_svr = grid_search.best_estimator_
    print("\n" + "="*50)
    print("OPTIMAL HYPERPARAMETERS FOUND:")
    print("="*50)
    for param, value in grid_search.best_params_.items():
        print(f"  - {param}: {value}")

    # -------------------------------------------------
    # Step C: Model Evaluation & Metric Calculation
    # -------------------------------------------------
    # Obtain predictions in scaled space and invert back to original units (grams)
    y_train_pred_scaled = best_svr.predict(X_train).reshape(-1, 1)
    y_test_pred_scaled = best_svr.predict(X_test).reshape(-1, 1)

    y_train_pred = y_scaler.inverse_transform(y_train_pred_scaled).ravel()
    y_test_pred = y_scaler.inverse_transform(y_test_pred_scaled).ravel()
    
    y_train_true = y_train_raw.ravel()
    y_test_true = y_test_raw.ravel()

    def print_metrics(y_true, y_pred, split_name="Testing"):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        r2 = r2_score(y_true, y_pred)
        acc_50 = np.mean(np.abs(y_true - y_pred) <= 50) * 100

        print(f"\n{split_name} Performance Metrics (Corrected SVR):")
        print(f"  - MAE:             {mae:.4f} g")
        print(f"  - RMSE:            {rmse:.4f} g")
        print(f"  - MSE:             {mse:.4f} g²")
        print(f"  - R²:              {r2:.4f}")
        print(f"  - Accuracy (±50g): {acc_50:.2f}%")
        
        return {"MAE": mae, "RMSE": rmse, "MSE": mse, "R2": r2, "Acc_50": acc_50}

    print_metrics(y_train_true, y_train_pred, "Training")
    test_metrics = print_metrics(y_test_true, y_test_pred, "Testing")

    # Save summary table for insertion into paper
    summary_df = pd.DataFrame([{
        "Model": "Optimized SVR (Corrected)",
        "Kernel": grid_search.best_params_['kernel'],
        "C": grid_search.best_params_['C'],
        "gamma": grid_search.best_params_['gamma'],
        "epsilon": grid_search.best_params_['epsilon'],
        "MAE": test_metrics["MAE"],
        "RMSE": test_metrics["RMSE"],
        "MSE": test_metrics["MSE"],
        "R2": test_metrics["R2"],
        "Accuracy (±50g)": test_metrics["Acc_50"]
    }])
    summary_df.to_csv("svr_optimized_baseline.csv", index=False)
    print("\nCorrected metrics saved to 'svr_optimized_baseline.csv'.")


if __name__ == "__main__":
    main()