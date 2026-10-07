# 🎯 Battle Royale Match Placement Prediction

> **A High-Performance Supervised Machine Learning & SHAP Interpretability System for PUBG / BGMI Finish Placement Prediction**  
> **Author:** Deepak R (24PD09)  
> **Course / Project:** Machine Learning Package

---

## 📌 Executive Summary & Abstract

In multiplayer battle royale games such as **PUBG / BGMI**, a player's final placement in a match depends on a complex interplay of individual performance, team behavior, and match context. Existing in-game statistics provide raw numbers (kills, damage dealt, distance traveled) but lack predictive insight into how these factors jointly determine survival and ranking.

This project delivers an end-to-end supervised regression pipeline that predicts a player's final match placement percentile (`winPlacePerc`) from in-match and pre-match statistics. The solution implements:
1. **High-Throughput Preprocessing & Feature Engineering** via **Polars** across **4,445,240 records**, constructing **4 distinct feature families (97 features)**.
2. **Strict Match-Level Partitioning (80:20 Split)** across 47,963 unique matches using `GroupShuffleSplit`, completely preventing teammate data leakage.
3. **Progressive 3-Model Benchmark**:
   - **Baseline:** Linear Regression (MAE: `0.0533`, R²: `0.9418`)
   - **Ensemble:** Random Forest Regressor (MAE: `0.0435`, R²: `0.9604`)
   - **Primary Model:** LightGBM Regressor (MAE: `0.0377`, R²: `0.9705`) — **Top Performing Model**
4. **Model Interpretability with SHAP (SHapley Additive exPlanations)**, deciphering the exact behavioral mechanisms driving survival.
5. **Interactive Streamlit Web Application** featuring tactical presets, real-time multi-model inference, and dynamic local SHAP feature attributions.

---

## 📊 Model Evaluation & Benchmark Results

All three models were evaluated on the exact same held-out test set comprising **278,001 records across 3,000 matches** (20% match split) with zero data leakage:

| Model Architecture | Test MAE (Primary) ⬇️ | Test RMSE ⬇️ | Test R² Score ⬆️ | Training Time | Primary Advantages & Findings |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Linear Regression** *(Baseline)* | `0.0533` | `0.0741` | `0.9418` | ~11.4s | Establishes a fast baseline; captures linear correlations but misses non-linear team dynamics. |
| **Random Forest Regressor** *(Ensemble)* | `0.0435` | `0.0612` | `0.9604` | ~83.4s | Substantial non-linear improvement; captures tree splits across movement and combat thresholds. |
| **LightGBM Regressor** *(Primary Model)* | **`0.0377`** | **`0.0528`** | **`0.9705`** | ~90.4s | **Decisive winner across all metrics.** Directly minimizes MAE (`regression_l1`), scales to large-scale data, handles high-dimensional interactions. |

> **Evaluation Visualizations:** Generated plots comparing MAE, RMSE, and R² are saved in `reports/figures/model_comparison.png`.

---

## 🛠️ Feature Engineering Pipeline (97 Features)

Placement in a Battle Royale is strictly **relative**: scoring 3 kills in a lobby of 100 players has a fundamentally different meaning than 3 kills in a 20-player custom game. Our pipeline engineers 4 feature families:

### Family 1: Combat Efficiency Ratios
- `totalDistance`: `walkDistance + rideDistance + swimDistance`
- `damage_per_kill`: `damageDealt / (kills + 1.0)` (identifies high-impact snipers vs. kill-stealers)
- `headshot_rate`: `headshotKills / (kills + 1.0)` (mechanical marksmanship index)
- `kills_per_walkDistance`: Aggression density relative to circle rotation
- `items_per_walkDistance`: `(heals + boosts + weaponsAcquired) / (walkDistance + 1.0)`
- `heals_and_boosts`: Total medical consumable utilization

### Family 2: Match Context Dynamics
- `players_in_match`: Actual player count in lobby (accounts for underfilled games)
- `groups_in_match`: Distinct squads alive in the match
- `group_size`: Solo (1), Duo (2), Squad (3–4)
- `match_type_group`: Categorical representation (`solo`, `duo`, `squad`, `custom`)

### Family 3: Team Aggregates (Squad Synergy)
Over `[matchId, groupId]`, calculates `mean`, `max`, `min`, and `sum` across teammates for:
- `walkDistance`, `totalDistance`, `damageDealt`, `kills`, `boosts`, `heals`, `weaponsAcquired`, `DBNOs`, `killPlace`
- *Insight:* A squad that moves, loots, and fights together exhibits drastically higher survival rates than isolated players.

### Family 4: Match-Relative Percentiles & Ratios
- `col_match_mean_ratio`: Player metric divided by match average `(val / match_mean)`
- `col_match_max_ratio`: Player metric divided by match maximum `(val / match_max)`
- `col_match_rank_perc`: Normalized rank percentile `(rank - 1) / (players - 1)` within the match lobby

---

## 🧠 Model Interpretability & SHAP Insights

Using **TreeSHAP** (`shap.TreeExplainer`) on the champion LightGBM model, we analyzed global feature attributions across held-out matches:

### Top Placement Drivers
1. **`killPlace_team_max` & `killPlace_match_rank_perc`:** The player's kill ranking within their match is the single strongest indicator of placement.
2. **`walkDistance_match_rank_perc` & `walkDistance_team_mean`:** Movement into shrinking playzones correlates almost linearly with survival. Players in the top 10% walk distance consistently finish in the top 15% placement.
3. **`totalDistance_team_min`:** Squad vulnerability bottleneck — if the least-mobile teammate lags behind, the squad is frequently wiped out early.
4. **`boosts_team_mean`:** High boost usage provides movement speed and passive health regeneration necessary for surviving late-game zone damage.

> **SHAP Visualizations:** Detailed plots are generated in `reports/figures/shap_summary_plot.png` and `reports/figures/feature_importance.png`.

---

## 🎮 Interactive Streamlit Web Application

The interactive web interface is designed with a sleek tactical dark theme and includes:
- **Real-Time Predictor:** Sliders and number inputs for all major combat, movement, looting, and squad synergy metrics.
- **Tactical Presets:** Instant loading of classic playstyles:
  - 🏆 *Chicken Dinner Champion* (high movement, coordinated squad, steady kills)
  - 🎯 *Long-Range Sniper* (high damage, deep kill distance, moderate movement)
  - 🏕️ *Tactical Snake / Camper* (high stealth walk distance, low combat engagement, high heals)
  - 💥 *Hot-Drop Rusher* (early high kills, low movement, early elimination risk)
  - 🏥 *Combat Medic* (high heals, revives, support coordination)
- **Multi-Model Comparison:** Instant side-by-side placement percentile estimates from LightGBM, Random Forest, and Linear Regression.
- **Local SHAP Feature Breakdown:** An interactive bar chart dynamically explaining exactly which stats pushed the player's prediction up (+) or down (-).
- **Benchmark Dashboard:** Interactive inspection of model metrics, feature importance rankings, and global SHAP beeswarm distributions.

---

## 🚀 Getting Started

### 1. Requirements
Ensure Python `>= 3.10` (Python 3.12 recommended) is installed.

```bash
pip install -r requirements.txt
```

### 2. Preprocess Data & Run Pipeline
To clean the raw dataset (`train_V2.csv`), generate the 97 features with Polars, and save match splits:

```bash
python preprocess_pipeline.py
```

### 3. Train Models & Generate Visualizations
To train Linear Regression, Random Forest, and LightGBM, compute evaluation metrics, and generate SHAP plots:

```bash
python train_models.py
```

### 4. Launch the Interactive Web Application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`.

---

## 📂 Project Structure

```
Batlle-Royale-Match-Placement-Prediction/
├── app.py                      # Interactive Streamlit Web Application
├── preprocess_pipeline.py      # Polars cleaning, feature engineering & match splits
├── train_models.py             # 3-model training pipeline, evaluation & SHAP analysis
├── requirements.txt            # Python dependencies
├── README.md                   # Complete project documentation
│
├── data/
│   ├── raw/
│   │   └── train_V2.csv        # Raw PUBG match dataset (~660 MB, ~4.45M rows)
│   └── processed/
│       ├── train_cleaned.parquet      # Cleaned dataset without glitches/cheaters
│       ├── features_full.parquet      # 102 columns (97 engineered features + IDs)
│       └── splits/
│           ├── train_match_ids.npy    # 80% Match IDs (38,370 matches)
│           └── test_match_ids.npy     # 20% Match IDs (9,593 matches)
│
├── models/
│   ├── linear_regression.joblib       # Trained baseline model
│   ├── random_forest.joblib           # Trained ensemble model
│   ├── lightgbm_model.joblib          # Trained primary model (Champion)
│   ├── shap_explainer.joblib          # Pre-computed TreeSHAP explainer
│   ├── feature_names.json             # 97 aligned feature names
│   ├── benchmark_stats.json           # Match lobby averages for inference
│   ├── metrics_summary.json           # MAE, RMSE, R² comparison metrics
│   └── feature_importances.csv        # Ranked feature importance table
│
└── reports/
    └── figures/
        ├── model_comparison.png       # Comparative MAE, RMSE, R² bar plots
        ├── feature_importance.png     # Top 20 LightGBM gain features
        └── shap_summary_plot.png      # Global SHAP beeswarm summary plot
```
