"""
Battle Royale Match Placement Prediction - Interactive Web Application
Built with Streamlit, LightGBM, Random Forest, Scikit-Learn, and SHAP.
Author: Deepak R (24PD09)
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="PUBG Match Placement Prediction",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern tactical gaming aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&family=Rajdhani:wght@600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-title {
        font-family: 'Rajdhani', sans-serif;
        font-size: 2.6rem;
        font-weight: 700;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        background: linear-gradient(90deg, #F5A623, #FF6B4A, #38BDF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.25);
    }
    
    .metric-card-title {
        color: #94A3B8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    .metric-card-value {
        font-family: 'Rajdhani', sans-serif;
        font-size: 2.4rem;
        font-weight: 700;
        margin: 0.3rem 0;
    }
    
    .badge-winner {
        background: linear-gradient(135deg, #F59E0B, #D97706);
        color: white;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    .badge-tier {
        background: rgba(56, 189, 248, 0.2);
        color: #38BDF8;
        border: 1px solid #38BDF8;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models_and_artifacts():
    """Load pre-trained models, feature names, benchmarks, and SHAP explainer."""
    lgb_model = joblib.load("models/lightgbm_model.joblib")
    rf_model = joblib.load("models/random_forest.joblib")
    lr_model = joblib.load("models/linear_regression.joblib")
    explainer = joblib.load("models/shap_explainer.joblib")
    
    with open("models/feature_names.json") as f:
        feature_names = json.load(f)
        
    with open("models/metrics_summary.json") as f:
        metrics = json.load(f)
        
    with open("models/benchmark_stats.json") as f:
        benchmarks = json.load(f)
        
    feat_imp = pd.read_csv("models/feature_importances.csv")
    
    return {
        "lgb": lgb_model,
        "rf": rf_model,
        "lr": lr_model,
        "explainer": explainer,
        "features": feature_names,
        "metrics": metrics,
        "benchmarks": benchmarks,
        "feat_imp": feat_imp
    }


artifacts = load_models_and_artifacts()
feature_names = artifacts["features"]
benchmarks = artifacts["benchmarks"]


def build_feature_vector(inputs: dict) -> pd.DataFrame:
    """
    Transform raw in-game user statistics into the exact 97 engineered features
    expected by the trained models.
    """
    kills = float(inputs["kills"])
    damage = float(inputs["damageDealt"])
    walk = float(inputs["walkDistance"])
    ride = float(inputs["rideDistance"])
    swim = float(inputs["swimDistance"])
    heals = float(inputs["heals"])
    boosts = float(inputs["boosts"])
    weapons = float(inputs["weaponsAcquired"])
    headshots = float(inputs["headshotKills"])
    dbnos = float(inputs["DBNOs"])
    assists = float(inputs["assists"])
    longest_kill = float(inputs["longestKill"])
    match_duration = float(inputs["matchDuration"])
    kill_place = float(inputs["killPlace"])
    match_type = inputs["matchType"]
    team_synergy = float(inputs.get("teamSynergy", 1.0))
    
    total_dist = walk + ride + swim
    
    # Match context
    players_in_match = float(inputs.get("players_in_match", 96))
    groups_in_match = float(inputs.get("groups_in_match", 28 if "squad" in match_type else (48 if "duo" in match_type else 94)))
    group_size = 1 if "solo" in match_type else (2 if "duo" in match_type else 4)
    
    type_map = {'solo': 0, 'duo': 1, 'squad': 2, 'custom': 3}
    match_type_group = type_map.get(match_type, 2)
    
    # 1. Efficiency & Combat Ratios
    damage_per_kill = damage / (kills + 1.0)
    headshot_rate = headshots / (kills + 1.0)
    kills_per_walk = kills / (walk + 1.0)
    heals_per_walk = heals / (walk + 1.0)
    boosts_per_walk = boosts / (walk + 1.0)
    heals_and_boosts = heals + boosts
    items_per_walk = (heals + boosts + weapons) / (walk + 1.0)
    kills_per_total = kills / (total_dist + 1.0)
    
    # 2. Team Aggregates (using team synergy multiplier)
    team_stats = {
        'walkDistance': (walk * team_synergy, walk * max(1.0, team_synergy), walk * min(1.0, team_synergy), walk * group_size * team_synergy),
        'totalDistance': (total_dist * team_synergy, total_dist * max(1.0, team_synergy), total_dist * min(1.0, team_synergy), total_dist * group_size * team_synergy),
        'damageDealt': (damage * team_synergy, damage * max(1.0, team_synergy), damage * min(1.0, team_synergy), damage * group_size * team_synergy),
        'kills': (kills * team_synergy, kills * max(1.0, team_synergy), kills * min(1.0, team_synergy), kills * group_size * team_synergy),
        'boosts': (boosts * team_synergy, boosts * max(1.0, team_synergy), boosts * min(1.0, team_synergy), boosts * group_size * team_synergy),
        'heals': (heals * team_synergy, heals * max(1.0, team_synergy), heals * min(1.0, team_synergy), heals * group_size * team_synergy),
        'weaponsAcquired': (weapons * team_synergy, weapons * max(1.0, team_synergy), weapons * min(1.0, team_synergy), weapons * group_size * team_synergy),
        'DBNOs': (dbnos * team_synergy, dbnos * max(1.0, team_synergy), dbnos * min(1.0, team_synergy), dbnos * group_size * team_synergy),
        'killPlace': (kill_place, kill_place, kill_place, kill_place * group_size)
    }
    
    # 3. Match-Relative Percentiles & Ratios
    # Computed relative to standard match lobby averages
    data = {
        'assists': assists,
        'boosts': boosts,
        'damageDealt': damage,
        'DBNOs': dbnos,
        'headshotKills': headshots,
        'heals': heals,
        'killPlace': kill_place,
        'killPoints': float(inputs.get("killPoints", 1000)),
        'kills': kills,
        'killStreaks': float(inputs.get("killStreaks", min(kills, 2))),
        'longestKill': longest_kill,
        'matchDuration': match_duration,
        'maxPlace': float(inputs.get("maxPlace", groups_in_match)),
        'numGroups': float(inputs.get("numGroups", groups_in_match)),
        'rankPoints': float(inputs.get("rankPoints", 1500)),
        'revives': float(inputs.get("revives", 0)),
        'rideDistance': ride,
        'roadKills': float(inputs.get("roadKills", 0)),
        'swimDistance': swim,
        'teamKills': float(inputs.get("teamKills", 0)),
        'vehicleDestroys': float(inputs.get("vehicleDestroys", 0)),
        'walkDistance': walk,
        'weaponsAcquired': weapons,
        'winPoints': float(inputs.get("winPoints", 1500)),
        'totalDistance': total_dist,
        'damage_per_kill': damage_per_kill,
        'headshot_rate': headshot_rate,
        'kills_per_walkDistance': kills_per_walk,
        'heals_per_walkDistance': heals_per_walk,
        'boosts_per_walkDistance': boosts_per_walk,
        'heals_and_boosts': heals_and_boosts,
        'items_per_walkDistance': items_per_walk,
        'kills_per_totalDistance': kills_per_total,
        'players_in_match': players_in_match,
        'groups_in_match': groups_in_match,
        'group_size': float(group_size),
        'match_type_group': match_type_group
    }
    
    # Add team aggregates
    for col, (mean_val, max_val, min_val, sum_val) in team_stats.items():
        data[f"{col}_team_mean"] = mean_val
        data[f"{col}_team_max"] = max_val
        data[f"{col}_team_min"] = min_val
        data[f"{col}_team_sum"] = sum_val
        
    # Add match relative ratios & rank percentiles
    rel_cols = ['walkDistance', 'totalDistance', 'damageDealt', 'kills', 'boosts', 'heals', 'weaponsAcquired', 'killPlace']
    for col in rel_cols:
        val = total_dist if col == 'totalDistance' else data.get(col, 0.0)
        bm = benchmarks.get(col, {'mean': 1000.0, 'max': 3000.0})
        m_mean = bm['mean']
        m_max = max(bm['max'], val + 1.0)
        
        data[f"{col}_match_mean_ratio"] = val / (m_mean + 1e-4)
        data[f"{col}_match_max_ratio"] = val / (m_max + 1e-4)
        
        # Rank perc approximation (killPlace is inverted: lower place = higher percentile)
        if col == 'killPlace':
            rank_perc = 1.0 - (val / (players_in_match + 1e-4))
        else:
            rank_perc = min(1.0, max(0.0, (val / (m_mean * 2.5 + 1e-4))))
        data[f"{col}_match_rank_perc"] = rank_perc
        
    # Create DataFrame aligned with model features
    df = pd.DataFrame([data])
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0
            
    return df[feature_names]


def get_tier_info(score: float):
    """Categorize predicted win placement percentile into tactical tiers."""
    perc = score * 100
    if perc >= 95:
        return "🏆 WINNER WINNER CHICKEN DINNER! (Top 5%)", "#F59E0B", "Champion"
    elif perc >= 80:
        return "🎖️ ELITE SURVIVOR (Top 20%)", "#10B981", "Grandmaster"
    elif perc >= 60:
        return "⚔️ TOP SQUAD CONTENDER (Top 40%)", "#38BDF8", "Diamond"
    elif perc >= 40:
        return "🛡️ MID-GAME SURVIVOR (Top 60%)", "#A855F7", "Platinum"
    elif perc >= 20:
        return "⚠️ EARLY-STAGE ELIMINATION (Bottom 40%)", "#F97316", "Gold"
    else:
        return "💀 HOT-DROP CASUALTY (Bottom 20%)", "#EF4444", "Bronze"


# =========================================================================
# APP HEADER
# =========================================================================
st.markdown('<div class="main-title">🎯 Battle Royale Match Placement Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Supervised Machine Learning System predicting player placement percentiles in PUBG/BGMI matches with SHAP interpretability. Developed by <b>Deepak R (24PD09)</b></div>', unsafe_allow_html=True)

# Navigation tabs
tab1, tab2, tab3 = st.tabs([
    "🕹️ Real-Time Placement Predictor",
    "📊 Model Benchmark & Comparison",
    "🧠 Feature Engineering & Behavioral Insights"
])


# =========================================================================
# TAB 1: REAL-TIME PREDICTOR
# =========================================================================
with tab1:
    st.subheader("Simulate In-Game Match Statistics")
    
    # Presets Bar
    preset_cols = st.columns([1.5, 3])
    with preset_cols[0]:
        preset = st.selectbox(
            "Select Tactical Preset:",
            [
                "Custom Player Stats",
                "🏆 Chicken Dinner Champion",
                "🎯 Long-Range Sniper",
                "🏕️ Tactical Snake / Camper",
                "💥 Hot-Drop Rusher",
                "🏥 Combat Medic"
            ]
        )
    with preset_cols[1]:
        st.info("💡 Tip: Presets populate typical in-game behaviors. Adjust sliders below to see immediate prediction updates.")

    # Preset values
    if preset == "🏆 Chicken Dinner Champion":
        default_kills = 8
        default_damage = 950
        default_walk = 3200
        default_ride = 1800
        default_swim = 0
        default_heals = 4
        default_boosts = 7
        default_weapons = 6
        default_headshots = 3
        default_dbnos = 5
        default_longest = 180
        default_kill_place = 4
        default_type = "squad"
        default_synergy = 1.2
    elif preset == "🎯 Long-Range Sniper":
        default_kills = 5
        default_damage = 650
        default_walk = 1600
        default_ride = 500
        default_swim = 0
        default_heals = 2
        default_boosts = 3
        default_weapons = 4
        default_headshots = 4
        default_dbnos = 3
        default_longest = 480
        default_kill_place = 12
        default_type = "duo"
        default_synergy = 1.0
    elif preset == "🏕️ Tactical Snake / Camper":
        default_kills = 1
        default_damage = 80
        default_walk = 2800
        default_ride = 0
        default_swim = 50
        default_heals = 6
        default_boosts = 5
        default_weapons = 3
        default_headshots = 0
        default_dbnos = 0
        default_longest = 25
        default_kill_place = 35
        default_type = "solo"
        default_synergy = 1.0
    elif preset == "💥 Hot-Drop Rusher":
        default_kills = 4
        default_damage = 420
        default_walk = 320
        default_ride = 0
        default_swim = 0
        default_heals = 0
        default_boosts = 0
        default_weapons = 2
        default_headshots = 1
        default_dbnos = 2
        default_longest = 35
        default_kill_place = 18
        default_type = "squad"
        default_synergy = 0.8
    elif preset == "🏥 Combat Medic":
        default_kills = 1
        default_damage = 180
        default_walk = 2100
        default_ride = 800
        default_swim = 0
        default_heals = 11
        default_boosts = 4
        default_weapons = 5
        default_headshots = 0
        default_dbnos = 1
        default_longest = 40
        default_kill_place = 45
        default_type = "squad"
        default_synergy = 1.15
    else:
        default_kills = 3
        default_damage = 350
        default_walk = 1500
        default_ride = 400
        default_swim = 0
        default_heals = 3
        default_boosts = 2
        default_weapons = 4
        default_headshots = 1
        default_dbnos = 2
        default_longest = 75
        default_kill_place = 25
        default_type = "squad"
        default_synergy = 1.0

    # User Input Form
    with st.expander("⚙️ Player & Match Statistics Controls", expanded=True):
        col_c1, col_c2, col_c3 = st.columns(3)
        
        with col_c1:
            st.markdown("### ⚔️ Combat Stats")
            kills = st.slider("Kills", 0, 30, default_kills)
            damage = st.slider("Damage Dealt", 0, 2500, default_damage, step=25)
            headshots = st.slider("Headshot Kills", 0, kills, min(default_headshots, kills))
            dbnos = st.slider("DBNOs (Knocks Dealt)", 0, 15, default_dbnos)
            longest_kill = st.slider("Longest Kill Distance (m)", 0, 800, default_longest, step=10)
            kill_place = st.slider("Kill Place Rank in Lobby", 1, 100, default_kill_place)
            
        with col_c2:
            st.markdown("### 🏃 Movement & Loot")
            walk_dist = st.slider("Walk Distance (m)", 0, 7000, default_walk, step=50)
            ride_dist = st.slider("Vehicle Ride Distance (m)", 0, 8000, default_ride, step=100)
            swim_dist = st.slider("Swim Distance (m)", 0, 1000, default_swim, step=25)
            weapons = st.slider("Weapons Acquired", 0, 25, default_weapons)
            heals = st.slider("Heals Used", 0, 25, default_heals)
            boosts = st.slider("Boosts Used", 0, 25, default_boosts)
            
        with col_c3:
            st.markdown("### 🌐 Match & Team Context")
            match_type = st.selectbox(
                "Match Type",
                ["squad", "duo", "solo", "custom"],
                index=["squad", "duo", "solo", "custom"].index(default_type)
            )
            team_synergy = st.slider(
                "Team Performance Synergy Multiplier",
                0.5, 2.0, default_synergy, step=0.05,
                help="Factor representing squadmates' average contribution relative to you."
            )
            model_choice = st.radio(
                "Predictive Model to Use:",
                ["LightGBM (Primary - Recommended)", "Random Forest (Ensemble)", "Linear Regression (Baseline)", "Compare All 3 Models"],
                horizontal=False
            )

    # Build feature vector
    raw_inputs = {
        "kills": kills,
        "damageDealt": damage,
        "walkDistance": walk_dist,
        "rideDistance": ride_dist,
        "swimDistance": swim_dist,
        "heals": heals,
        "boosts": boosts,
        "weaponsAcquired": weapons,
        "headshotKills": headshots,
        "DBNOs": dbnos,
        "assists": 1 if match_type != "solo" else 0,
        "longestKill": longest_kill,
        "killPlace": kill_place,
        "matchDuration": 1850,
        "matchType": match_type,
        "teamSynergy": team_synergy
    }
    
    input_df = build_feature_vector(raw_inputs)

    # Model Predictions
    pred_lgb = float(np.clip(artifacts["lgb"].predict(input_df)[0], 0.0, 1.0))
    pred_rf = float(np.clip(artifacts["rf"].predict(input_df)[0], 0.0, 1.0))
    pred_lr = float(np.clip(artifacts["lr"].predict(input_df)[0], 0.0, 1.0))
    
    if "LightGBM" in model_choice:
        chosen_pred = pred_lgb
        model_name = "LightGBM Regressor"
    elif "Random Forest" in model_choice:
        chosen_pred = pred_rf
        model_name = "Random Forest Regressor"
    elif "Linear" in model_choice:
        chosen_pred = pred_lr
        model_name = "Linear Regression"
    else:
        chosen_pred = pred_lgb
        model_name = "LightGBM (Ensemble Primary)"

    tier_text, tier_color, tier_badge = get_tier_info(chosen_pred)

    st.markdown("---")
    st.markdown("### 🏆 Prediction Results")
    
    res_col1, res_col2, res_col3, res_col4 = st.columns([2.5, 1.8, 1.8, 1.8])
    
    with res_col1:
        st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid {tier_color};">
            <div class="metric-card-title">Predicted Finish Placement Percentile</div>
            <div class="metric-card-value" style="color: {tier_color};">{chosen_pred * 100:.1f}%</div>
            <div style="font-weight: 700; color: #E2E8F0; margin-top: 4px;">{tier_text}</div>
            <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 8px;">Model: <b>{model_name}</b></div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(float(chosen_pred))
        
    with res_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">LightGBM</div>
            <div class="metric-card-value" style="color: #38BDF8;">{pred_lgb * 100:.1f}%</div>
            <div class="badge-tier">MAE: 0.0377</div>
        </div>
        """, unsafe_allow_html=True)
        
    with res_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Random Forest</div>
            <div class="metric-card-value" style="color: #50E3C2;">{pred_rf * 100:.1f}%</div>
            <div class="badge-tier">MAE: 0.0435</div>
        </div>
        """, unsafe_allow_html=True)
        
    with res_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-card-title">Linear Regression</div>
            <div class="metric-card-value" style="color: #F5A623;">{pred_lr * 100:.1f}%</div>
            <div class="badge-tier">MAE: 0.0533</div>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------
    # Real-Time SHAP Feature Contribution Explanation
    # -------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🔍 Real-Time SHAP Decision Explanation")
    st.caption("Local feature attribution explaining exactly how each behavioral input shifted your prediction relative to baseline.")

    explainer = artifacts["explainer"]
    shap_val = explainer(input_df)
    
    # Extract top contributing features for this prediction
    shap_contributions = pd.DataFrame({
        'feature': feature_names,
        'value': input_df.iloc[0].values,
        'shap_value': shap_val.values[0]
    })
    
    # Map feature names to user-friendly titles
    readable_map = {
        'walkDistance_match_rank_perc': 'Walk Distance (Lobby Percentile)',
        'killPlace_team_max': 'Team Best Kill Place Rank',
        'walkDistance_team_mean': 'Squad Average Walk Distance',
        'killPlace_match_max_ratio': 'Kill Place Ratio in Lobby',
        'kills_match_rank_perc': 'Kills (Lobby Percentile)',
        'totalDistance_team_min': 'Team Minimum Movement',
        'killPlace_match_rank_perc': 'Kill Place Rank Percentile',
        'boosts_team_mean': 'Squad Boosts Consumed',
        'weaponsAcquired_team_mean': 'Squad Weapons Looted',
        'damage_per_kill': 'Damage per Kill Efficiency',
        'heals_and_boosts': 'Consumables Used (Heals + Boosts)',
        'walkDistance': 'Raw Walk Distance (m)',
        'totalDistance': 'Total Travel Distance (m)',
        'killPlace': 'Raw Kill Place',
        'kills': 'Total Kills',
        'damageDealt': 'Damage Dealt'
    }
    
    shap_contributions['readable_feature'] = shap_contributions['feature'].map(lambda x: readable_map.get(x, x))
    top_pos = shap_contributions.sort_values('shap_value', ascending=False).head(6)
    top_neg = shap_contributions.sort_values('shap_value', ascending=True).head(6)
    top_impact = pd.concat([top_pos, top_neg]).drop_duplicates().sort_values('shap_value')

    col_chart, col_insights = st.columns([3, 2])
    
    with col_chart:
        fig, ax = plt.subplots(figsize=(9, 5))
        bar_colors = ['#10B981' if v > 0 else '#EF4444' for v in top_impact['shap_value']]
        ax.barh(top_impact['readable_feature'], top_impact['shap_value'], color=bar_colors, edgecolor='black', alpha=0.9)
        ax.axvline(0, color='white', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.set_xlabel("SHAP Impact on Predicted Placement (Percentile Points)", fontsize=10, fontweight='bold')
        ax.set_title(f"Top Positive (+) and Negative (-) Feature Drivers", fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
    with col_insights:
        st.markdown("#### 💡 Tactical Driver Insights")
        top_driver = top_pos.iloc[0]
        st.success(f"**Biggest Placement Boost:** `{top_driver['readable_feature']}` (+{top_driver['shap_value']:.3f} percentile)")
        
        lowest_driver = top_neg.iloc[0]
        if lowest_driver['shap_value'] < 0:
            st.error(f"**Biggest Placement Penalty:** `{lowest_driver['readable_feature']}` ({lowest_driver['shap_value']:.3f} percentile)")
        else:
            st.info("All top factors are currently contributing positively to your placement.")
            
        st.markdown("""
        **Key Takeaways:**
        - **Movement is King:** Running deeper into circles (`walkDistance`) accounts for higher placement variance than aggressive hot-drop kills.
        - **Team Cohesion:** In Duo/Squad matches, teammates moving together and surviving prevents early wipeouts.
        - **Combat Economy:** High damage dealt with low kills indicates poke damage from safety, which correlates with deep circle placement.
        """)


# =========================================================================
# TAB 2: MODEL COMPARISON & BENCHMARK
# =========================================================================
with tab2:
    st.subheader("Progressive Model Benchmark & Out-Of-Sample Evaluation")
    st.markdown("All 3 models were trained progressively and evaluated on **278,001 strictly held-out test match records (3,000 matches)** with zero teammate data leakage.")

    metrics = artifacts["metrics"]
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.columns = ["MAE (Primary)", "RMSE", "R² Score", "Training Time (s)"]
    
    st.dataframe(
        metrics_df.style.highlight_min(subset=["MAE (Primary)", "RMSE"], color="#166534")
                        .highlight_max(subset=["R² Score"], color="#166534")
                        .format("{:.4f}", subset=["MAE (Primary)", "RMSE", "R² Score"])
                        .format("{:.1f}s", subset=["Training Time (s)"]),
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("Publication-Quality Evaluation Visualizations")
    
    vis_col1, vis_col2 = st.columns(2)
    
    with vis_col1:
        st.markdown("#### 1. Three-Model Comparative Performance")
        st.image("reports/figures/model_comparison.png", caption="Linear Regression vs. Random Forest vs. LightGBM on Out-of-Sample Test Set", use_container_width=True)
        
    with vis_col2:
        st.markdown("#### 2. Top 20 Feature Importances (LightGBM Gain)")
        st.image("reports/figures/feature_importance.png", caption="Total Information Gain of Top Engineered Features", use_container_width=True)

    st.markdown("---")
    st.markdown("#### 3. Global SHAP Beeswarm Summary Plot")
    st.caption("Distribution of SHAP impacts across 2,000 held-out test matches. Red indicates high feature values; Blue indicates low feature values.")
    st.image("reports/figures/shap_summary_plot.png", use_container_width=True)


# =========================================================================
# TAB 3: FEATURE ENGINEERING & ARCHITECTURE
# =========================================================================
with tab3:
    st.subheader("Feature Engineering & Architecture Pipeline")
    st.markdown("""
    Battle royale placements are inherently **relative rather than absolute**: getting 3 kills in a lobby of 100 players means something entirely different from 3 kills in a 20-player custom match.
    
    Our preprocessing pipeline implemented in **Polars** generates **4 distinct feature families** across 4.45 million records in seconds:
    """)

    fam1, fam2 = st.columns(2)
    with fam1:
        st.markdown("""
        ### 1. Efficiency & Combat Ratios
        - `totalDistance`: Sum of walking, driving, and swimming distance.
        - `damage_per_kill`: Ratio of damage to kills — separates snipers from kill-stealers.
        - `headshot_rate`: Headshot percentage indicating player mechanical skill.
        - `kills_per_walkDistance`: Combat aggression per unit movement.
        - `items_per_walkDistance`: Looting density per meter traveled.
        
        ### 2. Match Context Dynamics
        - `players_in_match`: Actual count of players in the specific lobby.
        - `groups_in_match`: Number of distinct teams/squads alive.
        - `group_size`: Solo (1), Duo (2), Squad (3-4).
        - `match_type_group`: Categorical classification (`solo`, `duo`, `squad`, `custom`).
        """)

    with fam2:
        st.markdown("""
        ### 3. Team Aggregates (Synergy)
        - Over `[matchId, groupId]`, computes `mean`, `max`, `min`, `sum` for:
          - `walkDistance`, `totalDistance`, `damageDealt`, `kills`, `boosts`, `heals`, `weaponsAcquired`, `DBNOs`, `killPlace`.
        - Captures squad cohesion: a squad where everyone survives and moves together places drastically higher.
        
        ### 4. Match-Relative Percentiles & Ratios
        - `col_match_mean_ratio`: Player value divided by match mean.
        - `col_match_max_ratio`: Player value divided by match maximum.
        - `col_match_rank_perc`: Exact rank percentile `(rank - 1) / (players - 1)` within the match lobby.
        """)

    st.markdown("---")
    st.markdown("### 🏆 Architectural Summary & Author")
    st.markdown("""
    - **Dataset:** PUBG Finish Placement Prediction (`train_V2.csv`, ~4.45 million records).
    - **Validation Strategy:** 80:20 Split strictly by `matchId` via `GroupShuffleSplit` (prevents data leakage).
    - **Models:**
      1. `Linear Regression` (Baseline — MAE: `0.0533`)
      2. `Random Forest Regressor` (Ensemble — MAE: `0.0435`)
      3. `LightGBM Regressor` (Primary — MAE: `0.0377`, **Best Accuracy**)
    - **Interpretability:** TreeSHAP (`shap.TreeExplainer`) on gradient boosted trees.
    - **Developed by:** **Deepak R (24PD09)**
    """)
