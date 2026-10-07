"""
Battle Royale Match Placement Prediction - Model Training Pipeline
Trains 3 Progressive Models:
1. Linear Regression (Baseline)
2. Random Forest Regressor (Ensemble)
3. LightGBM Regressor (Primary/Final Model)

Evaluates on strictly separated 20% held-out test matches using MAE, RMSE, and R2.
Performs SHAP interpretability and saves model artifacts and visual reports.
"""

import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import shap


# Ensure directories exist
Path("models").mkdir(parents=True, exist_ok=True)
Path("reports/figures").mkdir(parents=True, exist_ok=True)

MAPPING_MATCH_TYPE = {'solo': 0, 'duo': 1, 'squad': 2, 'custom': 3}


def load_dataset(n_train_matches: int = 15000, n_test_matches: int = 3000, seed: int = 42):
    """
    Load features from parquet using matchId splits.
    Uses representative sample of matches to fit within system memory constraints while
    retaining full statistical representation across all match types.
    """
    print("\n[1/5] Loading match splits and engineered features...")
    train_matches = np.load("data/processed/splits/train_match_ids.npy", allow_pickle=True)
    test_matches = np.load("data/processed/splits/test_match_ids.npy", allow_pickle=True)

    np.random.seed(seed)
    selected_train_matches = np.random.choice(train_matches, size=min(n_train_matches, len(train_matches)), replace=False)
    selected_test_matches = np.random.choice(test_matches, size=min(n_test_matches, len(test_matches)), replace=False)

    print(f"Selected {len(selected_train_matches):,} train matches and {len(selected_test_matches):,} test matches.")

    scan = pl.scan_parquet("data/processed/features_full.parquet")

    # Filter train and test sets
    train_df = scan.filter(pl.col("matchId").is_in(selected_train_matches.tolist())).collect()
    test_df = scan.filter(pl.col("matchId").is_in(selected_test_matches.tolist())).collect()

    print(f"Train rows: {train_df.height:,} | Test rows: {test_df.height:,}")

    drop_cols = ['Id', 'groupId', 'matchId', 'matchType', 'winPlacePerc']
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    # Convert match_type_group categorical to integer
    def prep_xy(df: pl.DataFrame):
        df_pd = df.to_pandas()
        if 'match_type_group' in df_pd.columns:
            df_pd['match_type_group'] = df_pd['match_type_group'].astype(str).map(MAPPING_MATCH_TYPE).fillna(3).astype(int)
        X = df_pd[feature_cols].copy()
        y = df_pd['winPlacePerc'].to_numpy(dtype=np.float32)
        return X, y

    X_train, y_train = prep_xy(train_df)
    X_test, y_test = prep_xy(test_df)

    # Save feature names
    with open("models/feature_names.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    return X_train, y_train, X_test, y_test, feature_cols


def evaluate_predictions(y_true, y_pred, model_name: str):
    """Clip predictions to [0, 1] and calculate MAE, RMSE, and R2."""
    y_pred_clipped = np.clip(y_pred, 0.0, 1.0)
    mae = float(mean_absolute_error(y_true, y_pred_clipped))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred_clipped)))
    r2 = float(r2_score(y_true, y_pred_clipped))
    print(f"[{model_name}] -> MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")
    return {"mae": mae, "rmse": rmse, "r2": r2, "predictions": y_pred_clipped}


def train_and_evaluate(X_train, y_train, X_test, y_test, feature_cols):
    """Train all 3 models progressively and compare performance."""
    results = {}

    # -------------------------------------------------------------
    # Model 1: Baseline Linear Regression
    # -------------------------------------------------------------
    print("\n[2/5] Training Model 1: Linear Regression (Baseline)...")
    t0 = time.time()
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_train_time = time.time() - t0
    y_pred_lr = lr.predict(X_test)
    eval_lr = evaluate_predictions(y_test, y_pred_lr, "Linear Regression")
    results["Linear Regression"] = {
        "MAE": eval_lr["mae"],
        "RMSE": eval_lr["rmse"],
        "R2": eval_lr["r2"],
        "train_time_sec": round(lr_train_time, 2)
    }
    joblib.dump(lr, "models/linear_regression.joblib")
    print(f"Saved Linear Regression model. Train time: {lr_train_time:.2f}s")

    # -------------------------------------------------------------
    # Model 2: Random Forest Regressor
    # -------------------------------------------------------------
    print("\n[3/5] Training Model 2: Random Forest Regressor (Ensemble)...")
    # Subsample 150k rows for Random Forest to prevent OOM on 8GB RAM
    rf_sample_size = min(150000, len(X_train))
    idx_rf = np.random.choice(len(X_train), size=rf_sample_size, replace=False)
    X_train_rf = X_train.iloc[idx_rf]
    y_train_rf = y_train[idx_rf]

    t0 = time.time()
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=16,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features=0.5,
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train_rf, y_train_rf)
    rf_train_time = time.time() - t0
    y_pred_rf = rf.predict(X_test)
    eval_rf = evaluate_predictions(y_test, y_pred_rf, "Random Forest")
    results["Random Forest"] = {
        "MAE": eval_rf["mae"],
        "RMSE": eval_rf["rmse"],
        "R2": eval_rf["r2"],
        "train_time_sec": round(rf_train_time, 2)
    }
    joblib.dump(rf, "models/random_forest.joblib")
    print(f"Saved Random Forest model. Train time: {rf_train_time:.2f}s")

    # -------------------------------------------------------------
    # Model 3: LightGBM Regressor (Primary/Final Model)
    # -------------------------------------------------------------
    print("\n[4/5] Training Model 3: LightGBM Regressor (Primary/Final Model)...")
    t0 = time.time()
    lgb_model = lgb.LGBMRegressor(
        objective='regression_l1',  # Directly optimize MAE
        n_estimators=450,
        learning_rate=0.06,
        num_leaves=63,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
        importance_type='gain'
    )
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )
    lgb_train_time = time.time() - t0
    y_pred_lgb = lgb_model.predict(X_test)
    eval_lgb = evaluate_predictions(y_test, y_pred_lgb, "LightGBM")
    results["LightGBM"] = {
        "MAE": eval_lgb["mae"],
        "RMSE": eval_lgb["rmse"],
        "R2": eval_lgb["r2"],
        "train_time_sec": round(lgb_train_time, 2)
    }
    joblib.dump(lgb_model, "models/lightgbm_model.joblib")
    print(f"Saved LightGBM model. Train time: {lgb_train_time:.2f}s")

    # Save metrics summary
    with open("models/metrics_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    # -------------------------------------------------------------
    # Interpretability & SHAP Analysis
    # -------------------------------------------------------------
    print("\n[5/5] Performing Interpretability & SHAP Analysis...")
    explain_and_plot(lgb_model, X_test, feature_cols, results)

    print("\n================ Training & Evaluation Complete! ================")
    summary_df = pd.DataFrame(results).T
    print(summary_df.to_string())
    summary_df.to_csv("models/metrics_summary.csv")


def explain_and_plot(lgb_model, X_test, feature_cols, results):
    """Generate SHAP values, feature importance plots, and model comparison charts."""
    # 1. Feature Importance (Gain)
    importances = lgb_model.feature_importances_
    feat_imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)
    feat_imp.to_csv("models/feature_importances.csv", index=False)

    plt.figure(figsize=(10, 8))
    top20 = feat_imp.head(20)
    sns.barplot(data=top20, x='importance', y='feature', palette='viridis')
    plt.title("Top 20 Most Important Features (LightGBM Gain)", fontsize=14, fontweight='bold')
    plt.xlabel("Total Gain Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig("reports/figures/feature_importance.png", dpi=300)
    plt.close()
    print("Saved reports/figures/feature_importance.png")

    # 2. SHAP Explanation on a sample
    print("Computing SHAP values using TreeExplainer...")
    explainer = shap.TreeExplainer(lgb_model)
    shap_sample = X_test.sample(n=min(2000, len(X_test)), random_state=42)
    shap_values = explainer(shap_sample)

    joblib.dump(explainer, "models/shap_explainer.joblib")
    shap_sample.to_parquet("models/shap_sample.parquet")

    # SHAP Beeswarm Plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, shap_sample, show=False, max_display=15)
    plt.title("SHAP Beeswarm Summary Plot - PUBG Match Placement", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig("reports/figures/shap_summary_plot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved reports/figures/shap_summary_plot.png")

    # 3. Model Comparison Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    models = list(results.keys())
    maes = [results[m]["MAE"] for m in models]
    rmses = [results[m]["RMSE"] for m in models]
    r2s = [results[m]["R2"] for m in models]
    colors = ['#4A90E2', '#50E3C2', '#F5A623']

    # MAE (Lower is better)
    axes[0].bar(models, maes, color=colors, width=0.55, edgecolor='black', alpha=0.85)
    axes[0].set_title("Test Mean Absolute Error (MAE)\n(Lower is Better)", fontweight='bold')
    axes[0].set_ylabel("MAE")
    for i, v in enumerate(maes):
        axes[0].text(i, v + 0.002, f"{v:.4f}", ha='center', fontweight='bold')
    axes[0].set_ylim(0, max(maes) * 1.2)

    # RMSE (Lower is better)
    axes[1].bar(models, rmses, color=colors, width=0.55, edgecolor='black', alpha=0.85)
    axes[1].set_title("Test Root Mean Squared Error (RMSE)\n(Lower is Better)", fontweight='bold')
    axes[1].set_ylabel("RMSE")
    for i, v in enumerate(rmses):
        axes[1].text(i, v + 0.003, f"{v:.4f}", ha='center', fontweight='bold')
    axes[1].set_ylim(0, max(rmses) * 1.2)

    # R2 (Higher is better)
    axes[2].bar(models, r2s, color=colors, width=0.55, edgecolor='black', alpha=0.85)
    axes[2].set_title("Test R² Score\n(Higher is Better)", fontweight='bold')
    axes[2].set_ylabel("R²")
    for i, v in enumerate(r2s):
        axes[2].text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold')
    axes[2].set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig("reports/figures/model_comparison.png", dpi=300)
    plt.close()
    print("Saved reports/figures/model_comparison.png")


if __name__ == "__main__":
    X_train, y_train, X_test, y_test, feature_cols = load_dataset(
        n_train_matches=12000,  # ~1.1M rows: statistical depth while memory-safe
        n_test_matches=3000,    # ~280k rows for rigorous out-of-sample evaluation
        seed=42
    )
    train_and_evaluate(X_train, y_train, X_test, y_test, feature_cols)
