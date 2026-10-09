# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 11:47:56 2025
Updated with metrics and plots

@author: Maj0131
"""

import os
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
# =====================================================
# 1. File Loading and Feature Extraction
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
# 2. Define PyTorch Model
# =====================================================
class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x)

# =====================================================
# 3. Main Training Script
# =====================================================
def main():
    folder = "./"  # <-- change if needed
    df = build_dataset(folder)
    print("Dataset shape:", df.shape)
    print(df.head())

    # Train (run 0,1) / Test (run 2)
    train_df = df[df["Run"].isin([0,1,3])].copy()
    test_df = df[df["Run"].isin([2])].copy()

    X_train = train_df.drop(columns=["Weight", "Run", "File"]).values
    y_train = train_df["Weight"].values.reshape(-1, 1)
    X_test  = test_df.drop(columns=["Weight", "Run", "File"]).values
    y_test  = test_df["Weight"].values.reshape(-1, 1)

    # Normalize features
    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train)
    X_test = x_scaler.transform(X_test)

    # Normalize target
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test_scaled, dtype=torch.float32)

    # Model
    model = MLPRegressor(X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 200
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # =====================================================
    # Predictions with inverse scaling
    # =====================================================
    model.eval()
    with torch.no_grad():
        y_train_pred_scaled = model(X_train_t).numpy()
        y_test_pred_scaled = model(X_test_t).numpy()

    y_train_pred = y_scaler.inverse_transform(y_train_pred_scaled)
    y_test_pred = y_scaler.inverse_transform(y_test_pred_scaled)

    # Save results
    train_results = pd.DataFrame({
        "File": train_df["File"],
        "Run": train_df["Run"],
        "Actual": y_train.flatten(),
        "Predicted": y_train_pred.flatten()
    })
    test_results = pd.DataFrame({
        "File": test_df["File"],
        "Run": test_df["Run"],
        "Actual": y_test.flatten(),
        "Predicted": y_test_pred.flatten()
    })

    all_results = pd.concat([
        train_results.assign(Split="Train"),
        test_results.assign(Split="Test")
    ])
    all_results.to_csv("all_results_scaled.csv", index=False)

    # =====================================================
    # Print results to console
    # =====================================================
    print("\nFull Predictions (Training):")
    print(train_results.to_string(index=False))
    print("\nFull Predictions (Testing):")
    print(test_results.to_string(index=False))

    # =====================================================
    # Evaluation Metrics
    # =====================================================
    def evaluate(y_true, y_pred, split=""):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        r2 = r2_score(y_true, y_pred)
        accuracy = np.mean(np.abs(y_true.flatten() - y_pred.flatten()) <= 50) * 100
        print(f"\n{split} Metrics:")
        print(f" MAE:      {mae:.4f}")
        print(f" RMSE:     {rmse:.4f}")
        print(f" R²:       {r2:.4f}")
        print(f" Accuracy: {accuracy:.2f}% (within ±50g)")
        return mae, rmse, r2, accuracy

    evaluate(y_train, y_train_pred, "Training")
    evaluate(y_test, y_test_pred, "Testing")
    
    # =====================================================
    # Accuracy Table with Different Tolerances
    # =====================================================
    def accuracy_table(y_true, y_pred, split=""):
        tolerances = [5, 10, 15 ,20, 25, 30, 35, 40, 50]
        rows = []
        for tol in tolerances:
            acc = np.mean(np.abs(y_true.flatten() - y_pred.flatten()) <= tol) * 100
            rows.append({"Tolerance (g)": f"±{tol}", "Accuracy (%)": f"{acc:.2f}"})
        df_acc = pd.DataFrame(rows)
        print(f"\nAccuracy Table ({split}):")
        print(df_acc.to_string(index=False))
        return df_acc

    train_acc_table = accuracy_table(y_train, y_train_pred, "Training")
    test_acc_table  = accuracy_table(y_test, y_test_pred, "Testing")

    # =====================================================
    # Plots
    # =====================================================
    plt.figure(figsize=(10,6))
    plt.scatter(y_train, y_train_pred, color="green", alpha=0.7)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], "r--")
    plt.xlabel("Actual Weight (g)")
    plt.ylabel("Predicted Weight (g)")
    plt.title("Actual vs Predicted Weight (Training)")
    plt.show()

    plt.figure(figsize=(10,6))
    plt.scatter(y_test, y_test_pred, color="blue", alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual Weight (g)")
    plt.ylabel("Predicted Weight (g)")
    plt.title("Actual vs Predicted Weight (Testing)")
    plt.show()
    
    # =====================================================
    # Feature Importance (Permutation)
    # =====================================================
    

    def permutation_importance(model, X, y, feature_names, y_scaler, n_repeats=5):
       baseline_pred = y_scaler.inverse_transform(
           model(torch.tensor(X, dtype=torch.float32)).detach().numpy()
       )
       baseline_rmse = mean_squared_error(y, baseline_pred, squared=False)

       importances = {}
       for i, col in enumerate(feature_names):
           scores = []
           for _ in range(n_repeats):
               X_perm = X.copy()
               np.random.shuffle(X_perm[:, i])  # shuffle one column
               perm_pred = y_scaler.inverse_transform(
                   model(torch.tensor(X_perm, dtype=torch.float32)).detach().numpy()
               )
               perm_rmse = mean_squared_error(y, perm_pred, squared=False)
               scores.append(perm_rmse - baseline_rmse)
           importances[col] = np.mean(scores)

       importance_df = pd.DataFrame({
           "Feature": feature_names,
           "Importance": [importances[f] for f in feature_names]
       }).sort_values("Importance", ascending=False)

       print("\nFeature Importance (Permutation):")
       print(importance_df.to_string(index=False))

       # Plot
       plt.figure(figsize=(10,6))
       plt.barh(importance_df["Feature"], importance_df["Importance"])
       plt.xlabel("Increase in RMSE when permuted")
       plt.title("Feature Importance (Permutation)")
       plt.gca().invert_yaxis()
       plt.show()

       return importance_df

    feature_names = train_df.drop(columns=["Weight", "Run", "File"]).columns
    permutation_importance(model, X_test, y_test, feature_names, y_scaler)

    # =====================================================
    # Model Comparison Table (SVM, Random Forest, LSTM, Regression)
    # =====================================================
    def compute_metrics(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        r2 = r2_score(y_true, y_pred)
        accuracy = np.mean(np.abs(y_true.flatten() - y_pred.flatten()) <= 50) * 100
        return {
            "MAE": mae,
            "RMSE": rmse,
            "R²": r2,
            "Accuracy (±50g)": accuracy,
        }

    comparison_rows = []

    # Regression (current MLP)
    regression_metrics = compute_metrics(y_test, y_test_pred)
    comparison_rows.append({"Model": "Regression (MLP)", **regression_metrics})

    # SVM
    svm_model = SVR(C=100, gamma="scale", kernel="rbf")
    svm_model.fit(X_train, y_train.flatten())
    svm_pred = svm_model.predict(X_test).reshape(-1, 1)
    comparison_rows.append({"Model": "SVM", **compute_metrics(y_test, svm_pred)})

    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=300, random_state=42)
    rf_model.fit(X_train, y_train.flatten())
    rf_pred = rf_model.predict(X_test).reshape(-1, 1)
    comparison_rows.append({"Model": "Random Forest", **compute_metrics(y_test, rf_pred)})

    # LSTM
    class LSTMRegressor(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.lstm = nn.LSTM(input_size=input_dim, hidden_size=64, num_layers=2, batch_first=True)
            self.fc = nn.Linear(64, 1)

        def forward(self, x):
            output, _ = self.lstm(x)
            return self.fc(output[:, -1, :])

    torch.manual_seed(42)
    lstm_model = LSTMRegressor(X_train.shape[1])
    lstm_criterion = nn.MSELoss()
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

    X_train_lstm = X_train.reshape(-1, 1, X_train.shape[1])
    X_test_lstm = X_test.reshape(-1, 1, X_test.shape[1])

    X_train_lstm_t = torch.tensor(X_train_lstm, dtype=torch.float32)
    X_test_lstm_t = torch.tensor(X_test_lstm, dtype=torch.float32)
    y_train_scaled_t = torch.tensor(y_train_scaled.reshape(-1), dtype=torch.float32).unsqueeze(1)
    y_test_scaled_t = torch.tensor(y_test_scaled.reshape(-1), dtype=torch.float32).unsqueeze(1)

    for epoch in range(120):
        lstm_model.train()
        lstm_optimizer.zero_grad()
        lstm_pred_train = lstm_model(X_train_lstm_t)
        loss = lstm_criterion(lstm_pred_train, y_train_scaled_t)
        loss.backward()
        lstm_optimizer.step()

    lstm_model.eval()
    with torch.no_grad():
        lstm_pred_scaled = lstm_model(X_test_lstm_t).numpy()
    lstm_pred = y_scaler.inverse_transform(lstm_pred_scaled)
    comparison_rows.append({"Model": "LSTM", **compute_metrics(y_test, lstm_pred)})

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df = comparison_df[["Model", "MAE", "RMSE", "R²", "Accuracy (±50g)"]]
    print("\nModel Comparison Table:")
    print(comparison_df.to_string(index=False))
    comparison_df.to_csv("model_comparison_table.csv", index=False)


if __name__ == "__main__":
    main()
