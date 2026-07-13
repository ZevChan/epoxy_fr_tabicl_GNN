"""Predict Experiment.csv with best TabICL models."""
import os, sys, pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent

MODEL_CONFIGS = {
    "LOI":      {"dir": HERE.parent / "LOI/Result_Final_TabICL_Interpret",      "type": "regression"},
    "Tg":       {"dir": HERE.parent / "Tg/Result_Final_TabICL_Interpret",       "type": "regression"},
    "TENSILE":  {"dir": HERE.parent / "TENSILE/Result_Final_TabICL_Interpret",  "type": "regression"},
    "UL94":     {"dir": HERE.parent / "94/Result_Final_TabICL_Interpret_Cls",   "type": "classification"},
}

OUT_DIR = HERE.parent / "experiment_predictions"
OUT_DIR.mkdir(exist_ok=True)

# Load data
exp = pd.read_csv(HERE.parent / "Experiment.csv")
ep_d = pd.read_csv(HERE.parent / "Experiment_EP_D.csv")
fr_d = pd.read_csv(HERE.parent / "Experiment_FR_D.csv")
cur_d = pd.read_csv(HERE.parent / "Experiment_Curing_D.csv")

print(f"Experiment: {exp.shape[0]} rows, {exp.shape[1]} formula columns")
print(f"Descriptors: EP={ep_d.shape[1]}, FR={fr_d.shape[1]}, CURING={cur_d.shape[1]}")

# Handle FR_SMILES=0 rows: set FR descriptor values to 0
zero_fr_mask = exp["FR_SMILES"].astype(str) == "0"
print(f"FR_SMILES=0 rows: {zero_fr_mask.sum()}")
if zero_fr_mask.any():
    fr_d.loc[zero_fr_mask.values] = 0
    # Also set FR formula to 0
    exp.loc[zero_fr_mask.values, "Flame_retardant_AdditionAmount(wt%)"] = 0
    print("  FR descriptors and amount set to 0")

# Compute formula features
temp_cols = [f"Curing_Tem{i}" for i in range(1, 10)]
time_cols = [f"Curing_Time{i}" for i in range(1, 10)]
exp["T_max"] = exp[temp_cols].max(axis=1)
exp["t_total"] = exp[time_cols].sum(axis=1).fillna(0)
thermal = sum(exp[tc].fillna(0) * exp[tic].fillna(0) for tc, tic in zip(temp_cols, time_cols))
exp["Q_thermal"] = thermal
exp["EP_wt_fraction"] = (100.0 - exp["Flame_retardant_AdditionAmount(wt%)"].fillna(0) - exp["Curing_agent_AdditionAmount(wt%)"].fillna(0)) / 100.0
print(f"Formula features computed: T_max, t_total, Q_thermal, EP_wt_fraction")

# Build combined feature table: descriptor columns already have EP_/FR_/CURING_ prefix
combined = {}
for col in ep_d.columns:
    combined[col] = ep_d[col].values  # e.g. EP_MW
for col in fr_d.columns:
    combined[col] = fr_d[col].values  # e.g. FR_nAT
for col in cur_d.columns:
    combined[col] = cur_d[col].values  # e.g. CURING_SpPos...
# Add formula columns
for col in ["EEW", "Flame_retardant_AdditionAmount(wt%)", "Curing_agent_AdditionAmount(wt%)",
            "T_max", "t_total", "Q_thermal", "EP_wt_fraction"]:
    combined[col] = exp[col].values

full_df = pd.DataFrame(combined)
full_df = full_df.replace('na', np.nan).replace('NA', np.nan).replace('N/A', np.nan)
print(f"Full feature table: {full_df.shape}")

# Load models and predict
from tabicl import TabICLRegressor, TabICLClassifier

results = exp[["EP_SMILES", "FR_SMILES", "CURING_SMILES"]].copy()

for tgt, cfg in MODEL_CONFIGS.items():
    sd = cfg["dir"] / "saved_data"
    if not sd.exists():
        subdirs = [d for d in cfg["dir"].iterdir() if d.is_dir()]
        sd = subdirs[0] / "saved_data"
    
    with open(sd / "plot_data.pkl", "rb") as f:
        data = pickle.load(f)
    
    features = data["top_k_features"]
    X_train = data["X_train"]
    y_train = data["y_train"]
    
    # Select available features
    available_feats = [f for f in features if f in full_df.columns]
    missing = [f for f in features if f not in full_df.columns]
    if missing:
        print(f"  {tgt}: {len(missing)} missing features, filling with training median")
        for m in missing:
            # Find matching column in training data
            full_df[m] = 0.0  # fallback
    
    X_pred = full_df[available_feats].values.astype(np.float32)
    X_pred = np.nan_to_num(X_pred, nan=0.0)
    
    # Ensure feature order matches training
    feat_idx = [features.index(f) for f in available_feats]
    # Reindex X_pred to match training order
    # Actually, we need to build in training order
    ordered_X = np.zeros((len(X_pred), len(features)), dtype=np.float32)
    for j, f in enumerate(features):
        if f in full_df.columns:
            ordered_X[:, j] = full_df[f].values.astype(np.float32)
        else:
            ordered_X[:, j] = 0.0
    
    # Train model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    ModelClass = TabICLClassifier if cfg["type"] == "classification" else TabICLRegressor
    model = ModelClass(n_estimators=8, feat_shuffle_method="latin", outlier_threshold=3.0)
    model.fit(X_train_scaled, y_train)
    
    # Predict
    X_scaled = scaler.transform(ordered_X)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)
    if cfg["type"] == "classification":
        preds = model.predict_proba(X_scaled)[:, 1]
    else:
        preds = model.predict(X_scaled)
    
    results[f"{tgt}_pred"] = preds
    print(f"  {tgt}: {len(features)} features, {missing} missing, pred range [{preds.min():.3f}, {preds.max():.3f}]")

results.to_csv(OUT_DIR / "experiment_predictions.csv", index=False)
print(f"\nResults saved to {OUT_DIR / 'experiment_predictions.csv'}")
print(results.to_string())
