"""AD expansion analysis: OOF-based fine-tuning vs original model on external samples."""
import os, sys, pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, roc_auc_score

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent

MODEL_CONFIGS = {
    "LOI":      {"dir": HERE.parent / "LOI/Result_Final_TabICL_Interpret",      "type": "regression", "exp_col": "LOI",
                 "best_params": {"n_estimators": 8, "feat_shuffle_method": "latin", "outlier_threshold": 6.0}},
    "Tg":       {"dir": HERE.parent / "Tg/Result_Final_TabICL_Interpret",       "type": "regression", "exp_col": "Tg",
                 "best_params": {"n_estimators": 4, "feat_shuffle_method": "latin", "outlier_threshold": 5.0}},
    "TENSILE":  {"dir": HERE.parent / "TENSILE/Result_Final_TabICL_Interpret",  "type": "regression", "exp_col": "Tensile",
                 "best_params": {"n_estimators": 4, "feat_shuffle_method": "latin", "outlier_threshold": 5.5}},
    "UL94":     {"dir": HERE.parent / "94/Result_Final_TabICL_Interpret_Cls",   "type": "classification", "exp_col": "UL94",
                 "best_params": {"n_estimators": 8, "feat_shuffle_method": "latin", "outlier_threshold": 3.0}},
}

print("=" * 60)
print("AD EXPANSION ANALYSIS (OOF-based)")
print("=" * 60)

# Load external data
exp = pd.read_csv(HERE.parent / "Experiment.csv")
actual = pd.read_excel(HERE.parent / "Experiment_ActualData.xlsx")
ep_d = pd.read_csv(HERE.parent / "Experiment_EP_D.csv")
fr_d = pd.read_csv(HERE.parent / "Experiment_FR_D.csv")
cur_d = pd.read_csv(HERE.parent / "Experiment_Curing_D.csv")

zero_fr = exp["FR_SMILES"].astype(str) == "0"
if zero_fr.any():
    fr_d.loc[zero_fr.values] = 0
    exp.loc[zero_fr.values, "Flame_retardant_AdditionAmount(wt%)"] = 0

temp_cols = [f"Curing_Tem{i}" for i in range(1, 10)]
time_cols = [f"Curing_Time{i}" for i in range(1, 10)]
exp["T_max"] = exp[temp_cols].max(axis=1)
exp["t_total"] = exp[time_cols].sum(axis=1).fillna(0)
exp["Q_thermal"] = sum(exp[tc].fillna(0) * exp[tic].fillna(0) for tc, tic in zip(temp_cols, time_cols))
exp["EP_wt_fraction"] = (100.0 - exp["Flame_retardant_AdditionAmount(wt%)"].fillna(0) - exp["Curing_agent_AdditionAmount(wt%)"].fillna(0)) / 100.0

combined = {}
for col in ep_d.columns: combined[col] = ep_d[col].values
for col in fr_d.columns: combined[col] = fr_d[col].values
for col in cur_d.columns: combined[col] = cur_d[col].values
for col in ["EEW", "Flame_retardant_AdditionAmount(wt%)", "Curing_agent_AdditionAmount(wt%)",
            "T_max", "t_total", "Q_thermal", "EP_wt_fraction"]:
    combined[col] = exp[col].fillna(0).values
full_ext_df = pd.DataFrame(combined).replace('na', np.nan).replace('NA', np.nan)

from tabicl import TabICLRegressor, TabICLClassifier

all_records = []

for tgt, cfg in MODEL_CONFIGS.items():
    sd = cfg["dir"] / "saved_data"
    if not sd.exists():
        subdirs = [d for d in cfg["dir"].iterdir() if d.is_dir()]
        sd = subdirs[0] / "saved_data"
    with open(sd / "plot_data.pkl", "rb") as f:
        data = pickle.load(f)
    
    features = data["top_k_features"]
    X_train = data["X_train"].astype(np.float32)
    y_train = data["y_train"]
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"]
    X_full = np.vstack([X_train, X_test])
    y_full = np.concatenate([y_train, y_test])
    
    # External data
    if cfg["exp_col"] == "UL94":
        ul94_map = {'V0': 1, 'V-0': 1, 'NR': 0, 'V-2': 0}
        y_ext_raw = pd.to_numeric(actual[94].astype(str).str.strip().map(ul94_map), errors='coerce')
    else:
        y_ext_raw = pd.to_numeric(actual[cfg["exp_col"]], errors='coerce')
    valid = y_ext_raw.notna()
    n_valid = valid.sum()
    if n_valid == 0:
        print(f"  {tgt}: no external data, skip")
        continue
    
    y_ext = y_ext_raw[valid].values.astype(np.float32)
    X_ext = np.zeros((len(full_ext_df), len(features)), dtype=np.float32)
    for j, f in enumerate(features):
        if f in full_ext_df.columns:
            X_ext[:, j] = pd.to_numeric(full_ext_df[f], errors='coerce').fillna(0).astype(np.float32)
    X_ext_valid = X_ext[valid.values]
    
    print(f"\n{tgt}: {n_valid} external samples, {len(X_full)} original")
    
    # ── BASE model: train on ORIGINAL full data, predict external ──
    scaler_base = StandardScaler()
    X_base_tr = scaler_base.fit_transform(X_full)
    ModelClass = TabICLClassifier if cfg["type"] == "classification" else TabICLRegressor
    model_base = ModelClass(**cfg["best_params"])
    model_base.fit(X_base_tr, y_full)
    
    X_ext_scaled_base = scaler_base.transform(X_ext_valid)
    if cfg["type"] == "classification":
        base_preds = model_base.predict_proba(X_ext_scaled_base)[:, 1]
    else:
        base_preds = model_base.predict(X_ext_scaled_base)
    
    # Base AD distances
    nn_base = NearestNeighbors(n_neighbors=5, metric='euclidean')
    # Fit on training data in scaled space
    nn_base.fit(scaler_base.transform(X_full))
    base_dist, _ = nn_base.kneighbors(scaler_base.transform(X_ext_valid))
    base_ad = base_dist.mean(axis=1)
    
    # Training AD threshold
    train_dist, _ = nn_base.kneighbors(scaler_base.transform(X_full))
    train_ad = train_dist.mean(axis=1)
    ad_threshold = np.percentile(train_ad, 95)
    
    # Base errors
    if cfg["type"] == "classification":
        base_errors = np.abs(base_preds - y_ext)
    else:
        base_errors = np.abs(base_preds - y_ext)
    
    print(f"  Base: AD mean={base_ad.mean():.1f} (threshold={ad_threshold:.1f})")
    
    # ── Fine-tuned OOF: 5-fold CV on combined data, record OOF predictions for external samples ──
    X_combined = np.vstack([X_full, X_ext_valid])
    y_combined = np.concatenate([y_full, y_ext])
    # Track original vs external indices
    is_external = np.array([False]*len(X_full) + [True]*len(X_ext_valid))
    
    if cfg["type"] == "classification":
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    ft_oof_preds = np.full(len(X_ext_valid), np.nan)
    ft_oof_ad = np.full(len(X_ext_valid), np.nan)
    
    for tr_idx, te_idx in kf.split(X_combined, y_combined):
        X_tr, X_te = X_combined[tr_idx], X_combined[te_idx]
        y_tr, y_te = y_combined[tr_idx], y_combined[te_idx]
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        model = ModelClass(**cfg["best_params"])
        model.fit(X_tr_s, y_tr)
        
        # Find external samples in test fold
        te_is_ext = is_external[te_idx]
        ext_in_te = te_idx[te_is_ext]
        if len(ext_in_te) == 0:
            continue
        
        # Map external indices (from 0..n_valid-1) back to the external array
        ext_local_idx = ext_in_te - len(X_full)  # these are the indices in X_ext_valid
        
        X_te_ext = X_combined[ext_in_te]
        X_te_ext_s = scaler.transform(X_te_ext)
        
        if cfg["type"] == "classification":
            preds = model.predict_proba(X_te_ext_s)[:, 1]
        else:
            preds = model.predict(X_te_ext_s)
        
        ft_oof_preds[ext_local_idx] = preds
        
        # AD distance for OOF external samples
        nn_ft = NearestNeighbors(n_neighbors=5, metric='euclidean')
        nn_ft.fit(scaler.transform(X_tr))  # fit on this fold's training data
        dist, _ = nn_ft.kneighbors(X_te_ext_s)
        ft_oof_ad[ext_local_idx] = dist.mean(axis=1)
    
    ft_errors = np.abs(ft_oof_preds - y_ext)
    
    # Delta metrics
    delta_ad = ft_oof_ad - base_ad
    delta_error = ft_errors - base_errors
    
    print(f"  OOF-FT: AD mean={np.nanmean(ft_oof_ad):.1f}")
    print(f"  Delta AD: mean={np.nanmean(delta_ad):+.1f}")
    print(f"  Delta Error: mean={np.nanmean(delta_error):+.4f}")
    
    for i in range(n_valid):
        all_records.append({
            "Target": tgt,
            "Sample_Index": i,
            "AD_base": base_ad[i],
            "AD_finetune": ft_oof_ad[i],
            "Error_base": base_errors[i],
            "Error_finetune": ft_errors[i],
            "Delta_AD": delta_ad[i],
            "Delta_Error": delta_error[i],
            "y_exp": y_ext[i],
            "y_pred_base": base_preds[i],
            "y_pred_ft": ft_oof_preds[i],
            "AD_threshold": ad_threshold,
        })

# Save
records_df = pd.DataFrame(all_records)
out_path = HERE.parent / "experiment_predictions/ad_expansion_analysis.csv"
records_df.to_csv(out_path, index=False)
print(f"\nSaved {out_path}")
print(f"Total records: {len(records_df)}")
print(records_df.groupby('Target')[['Delta_AD', 'Delta_Error']].mean().to_string())
