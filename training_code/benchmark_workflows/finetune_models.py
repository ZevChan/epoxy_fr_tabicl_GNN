"""Fine-tune TabICL models with external experimental data and report performance."""
import os, sys, pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, roc_auc_score

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent

MODEL_CONFIGS = {
    "LOI":      {"dir": HERE.parent / "LOI/Result_Final_TabICL_Interpret",      "type": "regression", "exp_col": "LOI",
                 "best_params": {"n_estimators": 8, "feat_shuffle_method": "latin", "outlier_threshold": 6.0},
                 "orig_cv": 0.8498},
    "Tg":       {"dir": HERE.parent / "Tg/Result_Final_TabICL_Interpret",       "type": "regression", "exp_col": "Tg",
                 "best_params": {"n_estimators": 4, "feat_shuffle_method": "latin", "outlier_threshold": 5.0},
                 "orig_cv": 0.9038},
    "TENSILE":  {"dir": HERE.parent / "TENSILE/Result_Final_TabICL_Interpret",  "type": "regression", "exp_col": "Tensile",
                 "best_params": {"n_estimators": 4, "feat_shuffle_method": "latin", "outlier_threshold": 5.5},
                 "orig_cv": 0.7223},
    "UL94":     {"dir": HERE.parent / "94/Result_Final_TabICL_Interpret_Cls",   "type": "classification", "exp_col": "UL94",
                 "best_params": {"n_estimators": 8, "feat_shuffle_method": "latin", "outlier_threshold": 3.0},
                 "orig_cv": 0.9030},
}

print("=" * 60)
print("TabICL Fine-tuning with External Experimental Data")
print("=" * 60)

# ── Load training data ──
from tabicl import TabICLRegressor, TabICLClassifier

# Load external data
exp = pd.read_csv(HERE.parent / "Experiment.csv")
actual = pd.read_excel(HERE.parent / "Experiment_ActualData.xlsx")
ep_d = pd.read_csv(HERE.parent / "Experiment_EP_D.csv")
fr_d = pd.read_csv(HERE.parent / "Experiment_FR_D.csv")
cur_d = pd.read_csv(HERE.parent / "Experiment_Curing_D.csv")

# Handle FR_SMILES=0
zero_fr = exp["FR_SMILES"].astype(str) == "0"
if zero_fr.any():
    fr_d.loc[zero_fr.values] = 0
    exp.loc[zero_fr.values, "Flame_retardant_AdditionAmount(wt%)"] = 0

# Compute formula features
temp_cols = [f"Curing_Tem{i}" for i in range(1, 10)]
time_cols = [f"Curing_Time{i}" for i in range(1, 10)]
exp["T_max"] = exp[temp_cols].max(axis=1)
exp["t_total"] = exp[time_cols].sum(axis=1).fillna(0)
exp["Q_thermal"] = sum(exp[tc].fillna(0) * exp[tic].fillna(0) for tc, tic in zip(temp_cols, time_cols))
exp["EP_wt_fraction"] = (100.0 - exp["Flame_retardant_AdditionAmount(wt%)"].fillna(0) - exp["Curing_agent_AdditionAmount(wt%)"].fillna(0)) / 100.0

# Build external feature table
combined = {}
for col in ep_d.columns: combined[col] = ep_d[col].values
for col in fr_d.columns: combined[col] = fr_d[col].values
for col in cur_d.columns: combined[col] = cur_d[col].values
for col in ["EEW", "Flame_retardant_AdditionAmount(wt%)", "Curing_agent_AdditionAmount(wt%)",
            "T_max", "t_total", "Q_thermal", "EP_wt_fraction"]:
    combined[col] = exp[col].fillna(0).values
full_ext_df = pd.DataFrame(combined).replace('na', np.nan).replace('NA', np.nan)

print(f"External data: {len(full_ext_df)} rows, {full_ext_df.shape[1]} features")

results = []
for tgt, cfg in MODEL_CONFIGS.items():
    print(f"\n{'='*40}")
    print(f"Target: {tgt}")
    
    # Load original training data
    sd = cfg["dir"] / "saved_data"
    if not sd.exists():
        subdirs = [d for d in cfg["dir"].iterdir() if d.is_dir()]
        sd = subdirs[0] / "saved_data"
    with open(sd / "plot_data.pkl", "rb") as f:
        data = pickle.load(f)
    
    features = data["top_k_features"]
    X_train_orig = data["X_train"].astype(np.float32)
    y_train_orig = data["y_train"]
    
    # Get experimental values for this target
    exp_col_name = cfg["exp_col"]
    if exp_col_name == "UL94":
        # Map string UL94 values to binary
        ul94_map = {'V0': 1, 'V-0': 1, 'NR': 0, 'V-2': 0, 'V-1': 0, 'V1': 0, 'V2': 0}
        y_ext_raw_raw = actual[94].astype(str).str.strip().map(ul94_map)
        y_ext_raw = pd.to_numeric(y_ext_raw_raw, errors='coerce')
    else:
        y_ext_raw = pd.to_numeric(actual[exp_col_name], errors='coerce')
    
    # Keep only rows with experimental data
    valid = y_ext_raw.notna()
    n_valid = valid.sum()
    
    if n_valid == 0:
        print(f"  No experimental data available for {tgt} — skipping")
        results.append({
            "Target": tgt,
            "Type": cfg["type"],
            "Original_train_N": len(X_train_orig),
            "New_samples": 0,
            "Note": "No experimental data"
        })
        continue
    
    y_ext = y_ext_raw[valid].values.astype(np.float32)
    
    # Build X for external samples in training feature order
    X_ext = np.zeros((len(full_ext_df), len(features)), dtype=np.float32)
    for j, f in enumerate(features):
        if f in full_ext_df.columns:
            X_ext[:, j] = pd.to_numeric(full_ext_df[f], errors='coerce').fillna(0).values.astype(np.float32)
    X_ext_valid = X_ext[valid.values]
    
    print(f"  Original training: {len(X_train_orig)} samples")
    print(f"  New experimental: {n_valid} samples")
    print(f"  External y: mean={y_ext.mean():.2f}, range=[{y_ext.min():.2f}, {y_ext.max():.2f}]")
    
    # ── Original model: 5-fold CV on FULL dataset (X_train + X_test) ──
    X_train_data = data["X_train"].astype(np.float32)
    y_train_data = data["y_train"]
    X_test_data = data["X_test"].astype(np.float32)
    y_test_data = data["y_test"]
    
    # Full dataset = train + test
    X_full = np.vstack([X_train_data, X_test_data])
    y_full = np.concatenate([y_train_data, y_test_data])
    
    from sklearn.model_selection import StratifiedKFold, KFold
    if cfg["type"] == "classification":
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    orig_scores = []
    for tr_idx, te_idx in kf.split(X_full, y_full):
        X_tr, X_te = X_full[tr_idx], X_full[te_idx]
        y_tr, y_te = y_full[tr_idx], y_full[te_idx]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        ModelClass = TabICLClassifier if cfg["type"] == "classification" else TabICLRegressor
        model = ModelClass(**cfg["best_params"])
        model.fit(X_tr_s, y_tr)
        if cfg["type"] == "classification":
            orig_scores.append(roc_auc_score(y_te, model.predict_proba(X_te_s)[:, 1]))
        else:
            orig_scores.append(r2_score(y_te, model.predict(X_te_s)))
    
    orig_mean = np.mean(orig_scores)
    orig_std = np.std(orig_scores)
    print(f"  Full dataset: {len(X_full)} samples")
    print(f"  Original 5-CV: {cfg['type']=='classification' and 'AUC' or 'R2'} = {orig_mean:.4f} +/- {orig_std:.4f}")
    print(f"  Published 5-CV: {cfg['orig_cv']:.4f}")
    
    # ── Fine-tuned: add external data, same 5-fold CV ──
    X_combined = np.vstack([X_full, X_ext_valid])
    y_combined = np.concatenate([y_full, y_ext])
    
    if cfg["type"] == "classification":
        kf_ft = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        kf_ft = KFold(n_splits=5, shuffle=True, random_state=42)
    
    ft_scores = []
    for tr_idx, te_idx in kf_ft.split(X_combined, y_combined):
        X_tr, X_te = X_combined[tr_idx], X_combined[te_idx]
        y_tr, y_te = y_combined[tr_idx], y_combined[te_idx]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        ModelClass = TabICLClassifier if cfg["type"] == "classification" else TabICLRegressor
        model = ModelClass(**cfg["best_params"])
        model.fit(X_tr_s, y_tr)
        if cfg["type"] == "classification":
            ft_scores.append(roc_auc_score(y_te, model.predict_proba(X_te_s)[:, 1]))
        else:
            ft_scores.append(r2_score(y_te, model.predict(X_te_s)))
    
    ft_mean = np.mean(ft_scores)
    ft_std = np.std(ft_scores)
    print(f"  Fine-tuned 5-CV: {cfg['type']=='classification' and 'AUC' or 'R2'} = {ft_mean:.4f} +/- {ft_std:.4f}")
    print(f"  Delta vs published = {ft_mean - cfg['orig_cv']:+.4f}")
    
    results.append({
        "Target": tgt,
        "Type": cfg["type"],
        "Metric": "AUC" if cfg["type"] == "classification" else "R2",
        "Published_5CV": cfg["orig_cv"],
        "Full_dataset_N": len(X_full),
        "New_samples": n_valid,
        "Original_5CV_mean": orig_mean,
        "Original_5CV_std": orig_std,
        "FineTuned_5CV_mean": ft_mean,
        "FineTuned_5CV_std": ft_std,
        "Delta": ft_mean - orig_mean,
    })

# ── Summary ──
print(f"\n{'='*60}")
print("FINE-TUNING SUMMARY")
print(f"{'='*60}")
for r in results:
    if 'Note' in r:
        print(f"{r['Target']}: {r['Note']}")
    else:
        print(f"{r['Target']} ({r['Metric']}): published={r['Published_5CV']:.4f}, original-CV={r['Original_5CV_mean']:.4f} -> fine-tuned-CV={r['FineTuned_5CV_mean']:.4f} (delta={r['Delta']:+.4f}) [+{r['New_samples']}]")


# Save results
res_df = pd.DataFrame(results)
res_df.to_csv(HERE.parent / "experiment_predictions/finetune_results.csv", index=False)
print(f"\nSaved to experiment_predictions/finetune_results.csv")
