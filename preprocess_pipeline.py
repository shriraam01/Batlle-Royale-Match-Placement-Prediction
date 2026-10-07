"""
PUBG Placement Prediction - Data Preprocessing & Feature Engineering Pipeline
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import GroupShuffleSplit


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop missing target, remove cheater/glitch rows, and downcast datatypes."""
    # 1. Drop missing target row
    df = df.dropna(subset=['winPlacePerc']).copy()

    # 2. Filter out cheaters and game glitches
    total_dist = df['walkDistance'] + df['rideDistance'] + df['swimDistance']
    cheaters = (
        ((total_dist == 0) & (df['kills'] > 0)) |                       # Kills with no movement
        ((df['kills'] >= 10) & (df['headshotKills'] == df['kills'])) |  # 100% headshot aimbots
        (df['longestKill'] > 1000) |                                    # Impossible kill distance (>1km)
        (df['weaponsAcquired'] > 50)                                    # Weapon inventory glitch
    )
    df = df[~cheaters].copy()

    # 3. Downcast numeric dtypes to save memory
    for col in df.select_dtypes(include=['int', 'int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='unsigned')
    for col in df.select_dtypes(include=['float', 'float64']).columns:
        df[col] = df[col].astype(np.float32)

    return df


def engineer_features(df_polars: pl.DataFrame) -> pl.DataFrame:
    """Generate 4 feature families: efficiency, match context, team aggregates, and match ranks."""
    # Family 1: Efficiency & Combat Ratios
    total_dist = pl.col('walkDistance') + pl.col('rideDistance') + pl.col('swimDistance')
    df = df_polars.with_columns([
        total_dist.alias('totalDistance'),
        (pl.col('damageDealt') / (pl.col('kills') + 1.0)).alias('damage_per_kill'),
        (pl.col('headshotKills') / (pl.col('kills') + 1.0)).alias('headshot_rate'),
        (pl.col('kills') / (pl.col('walkDistance') + 1.0)).alias('kills_per_walkDistance'),
        (pl.col('heals') / (pl.col('walkDistance') + 1.0)).alias('heals_per_walkDistance'),
        (pl.col('boosts') / (pl.col('walkDistance') + 1.0)).alias('boosts_per_walkDistance'),
        (pl.col('heals') + pl.col('boosts')).alias('heals_and_boosts'),
        ((pl.col('heals') + pl.col('boosts') + pl.col('weaponsAcquired')) / (pl.col('walkDistance') + 1.0)).alias('items_per_walkDistance'),
        (pl.col('kills') / (total_dist + 1.0)).alias('kills_per_totalDistance'),
    ])

    # Family 2: Match Context
    df = df.with_columns([
        pl.len().over('matchId').cast(pl.UInt16).alias('players_in_match'),
        pl.col('groupId').n_unique().over('matchId').cast(pl.UInt16).alias('groups_in_match'),
        pl.len().over(['matchId', 'groupId']).cast(pl.UInt8).alias('group_size'),
        pl.when(pl.col('matchType').cast(pl.String).str.contains('solo')).then(pl.lit('solo'))
          .when(pl.col('matchType').cast(pl.String).str.contains('duo')).then(pl.lit('duo'))
          .when(pl.col('matchType').cast(pl.String).str.contains('squad')).then(pl.lit('squad'))
          .otherwise(pl.lit('custom')).cast(pl.Categorical).alias('match_type_group')
    ])

    # Family 3: Team Aggregates (mean, max, min, sum across teammates)
    team_cols = ['walkDistance', 'totalDistance', 'damageDealt', 'kills', 'boosts', 'heals', 'weaponsAcquired', 'DBNOs', 'killPlace']
    team_exprs = []
    for col in team_cols:
        for stat in ['mean', 'max', 'min', 'sum']:
            team_exprs.append(getattr(pl.col(col), stat)().over(['matchId', 'groupId']).cast(pl.Float32).alias(f'{col}_team_{stat}'))
    df = df.with_columns(team_exprs)

    # Family 4: Match-Relative Percentiles & Ratios
    rel_cols = ['walkDistance', 'totalDistance', 'damageDealt', 'kills', 'boosts', 'heals', 'weaponsAcquired', 'killPlace']
    rel_exprs = []
    for col in rel_cols:
        rel_exprs.append((pl.col(col) / (pl.col(col).mean().over('matchId') + 1e-4)).cast(pl.Float32).alias(f'{col}_match_mean_ratio'))
        rel_exprs.append((pl.col(col) / (pl.col(col).max().over('matchId') + 1e-4)).cast(pl.Float32).alias(f'{col}_match_max_ratio'))
        rel_exprs.append(
            ((pl.col(col).rank(method='average').over('matchId') - 1.0) /
             (pl.col('players_in_match').cast(pl.Float32) - 1.0 + 1e-4)).cast(pl.Float32).alias(f'{col}_match_rank_perc')
        )
    df = df.with_columns(rel_exprs)

    return df


def split_matches(match_ids: pd.Series, test_size: float = 0.2, seed: int = 42):
    """Split 80/20 train/test strictly by matchId to prevent teammate data leakage."""
    matches = match_ids.unique()
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(matches, groups=matches))
    return matches[train_idx], matches[test_idx]


def run_pipeline(raw_csv_path: str = "data/raw/train_V2.csv"):
    """Execute full data cleaning, feature engineering, and match-based train/test splitting."""
    splits_dir = Path("data/processed/splits")
    splits_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading raw data from {raw_csv_path}...")
    df_raw = pl.read_csv(raw_csv_path)
    print(f"Raw data shape: {df_raw.shape}")

    print("Cleaning data...")
    total_dist = pl.col('walkDistance') + pl.col('rideDistance') + pl.col('swimDistance')
    cheaters = (
        ((total_dist == 0) & (pl.col('kills') > 0)) |
        ((pl.col('kills') >= 10) & (pl.col('headshotKills') == pl.col('kills'))) |
        (pl.col('longestKill') > 1000) |
        (pl.col('weaponsAcquired') > 50)
    )
    df_clean = df_raw.filter(pl.col('winPlacePerc').is_not_null() & ~cheaters)
    print(f"Cleaned data shape: {df_clean.shape}")

    clean_path = "data/processed/train_cleaned.parquet"
    print(f"Saving cleaned data to {clean_path}...")
    df_clean.write_parquet(clean_path, compression="snappy")

    print("Engineering features...")
    df_feat = engineer_features(df_clean)
    print(f"Engineered features shape: {df_feat.shape}")

    feat_path = "data/processed/features_full.parquet"
    print(f"Saving engineered features to {feat_path}...")
    df_feat.write_parquet(feat_path, compression="snappy")

    print("Splitting matches 80/20 by matchId...")
    match_ids = df_feat.select("matchId").to_series().to_numpy()
    train_matches, test_matches = split_matches(pd.Series(match_ids), test_size=0.2, seed=42)

    np.save("data/processed/splits/train_match_ids.npy", train_matches)
    np.save("data/processed/splits/test_match_ids.npy", test_matches)
    print("Saved match splits successfully.")

    verify_processed_data()


def verify_processed_data():
    """Verify shapes and match splits of existing processed data."""
    clean_df = pl.scan_parquet("data/processed/train_cleaned.parquet")
    feat_df = pl.scan_parquet("data/processed/features_full.parquet")
    train_matches = np.load("data/processed/splits/train_match_ids.npy", allow_pickle=True)
    test_matches = np.load("data/processed/splits/test_match_ids.npy", allow_pickle=True)

    print("=== Dataset Summary ===")
    print(f"Cleaned Data:    {clean_df.select(pl.len()).collect().item():,} rows, {len(clean_df.collect_schema().names())} cols")
    print(f"Engineered Data: {feat_df.select(pl.len()).collect().item():,} rows, {len(feat_df.collect_schema().names())} cols")
    print(f"Train Matches:   {len(train_matches):,} (80%)")
    print(f"Test Matches:    {len(test_matches):,} (20%)")


if __name__ == "__main__":
    if not (Path("data/processed/features_full.parquet").exists() and
            Path("data/processed/splits/train_match_ids.npy").exists()):
        print("Processed data not found. Running end-to-end pipeline...")
        run_pipeline()
    else:
        verify_processed_data()


