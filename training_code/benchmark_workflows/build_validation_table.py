"""Build comprehensive experiment validation table with AD and risk assessment."""
import os, sys, pickle, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "experiment_predictions"

# ── Load data ──
exp = pd.read_csv(HERE.parent / "Experiment.csv")
actual = pd.read_excel(HERE.parent / "Experiment_ActualData.xlsx")
pred = pd.read_csv(OUT_DIR / "experiment_predictions.csv")

# ── Build comprehensive table ──
df = pd.DataFrame()
df["Sample_ID"] = [f"S{i+1}" for i in range(len(exp))]

# SMILES-based names (simplified)
ep_map = {
    "CC(C)(C1=CC=C(OCC2CO2)C=C1)C1=CC=C(OCC2CO2)C=C1": "DGEBA (E51)",
    "C1C(C(CC2C1O2)C(=O)OCC3CO3)C(=O)OCC4CO4": "TEPIC-derived",
    "C1C=CCC(C1C(=O)OCC2CO2)C(=O)OCC3CO3": "THPA-derived",
}
cur_map = {
    "NC1=CC=C(CC2=CC=C(N)C=C2)C=C1": "DDM",
    "C1CC(CCC1CC2CCC(CC2)N)N": "PACM",
    "CC1(CC(CC(C1)(C)CN)N)C": "MXDA",
    "C1=CC(=CC=C1N)S(=O)(=O)N=C(N)N": "Sulfaguanidine",
    "C1=CC(=CC=C1N)S(=O)(=O)C2=CC=C(C=C2)N": "DDS",
}

def smi_to_name(smi, mapping, default="Unknown"):
    return mapping.get(str(smi).strip(), default)

df["EP_Name"] = exp["EP_SMILES"].apply(lambda x: smi_to_name(x, ep_map))
df["FR_Name"] = exp["FR_SMILES"].apply(lambda x: "None" if str(x) == "0" else smi_to_name(x, {}, f"FR-{hash(str(x))%10000:04d}"))
df["Curing_Name"] = exp["CURING_SMILES"].apply(lambda x: smi_to_name(x, cur_map))
df["EP_SMILES"] = exp["EP_SMILES"]
df["FR_SMILES"] = exp["FR_SMILES"]
df["CURING_SMILES"] = exp["CURING_SMILES"]

# Formula details
temp_cols = [f"Curing_Tem{i}" for i in range(1, 10)]
time_cols = [f"Curing_Time{i}" for i in range(1, 10)]
df["FR_wt%"] = exp["Flame_retardant_AdditionAmount(wt%)"]
df["Curing_wt%"] = exp["Curing_agent_AdditionAmount(wt%)"]
df["EEW"] = exp["EEW"]
df["T_max"] = exp[temp_cols].max(axis=1)
df["t_total"] = exp[time_cols].sum(axis=1).fillna(0)
df["Q_thermal"] = sum(exp[tc].fillna(0) * exp[tic].fillna(0) for tc, tic in zip(temp_cols, time_cols))

# Cure schedule description
def cure_desc(row):
    stages = []
    for i in range(1, 10):
        t = row.get(f"Curing_Tem{i}", 0)
        h = row.get(f"Curing_Time{i}", 0)
        if t and h and t > 0 and h > 0:
            stages.append(f"{int(t)}°C/{int(h)}h")
    return " + ".join(stages) if stages else "N/A"
df["Cure_Schedule"] = exp.apply(cure_desc, axis=1)

# ── Experimental values ──
exp_col_map = {"LOI": "LOI", 94: "UL94", "Tensile": "Tensile", "Tg": "Tg"}
for src_col, tgt_name in exp_col_map.items():
    if src_col == 94:
        # Map string UL94 to numeric
        ul94_map = {'V0': 1, 'V-0': 1, 'NR': 0, 'V-2': 0, 'V-1': 0}
        raw = actual[src_col].astype(str).str.strip().map(ul94_map)
        vals = pd.to_numeric(raw, errors='coerce').values
    else:
        vals = pd.to_numeric(actual[src_col], errors='coerce').values
    df[f"{tgt_name}_exp"] = vals

# ── Predictions ──
for tgt in ["LOI", "Tg", "TENSILE", "UL94"]:
    df[f"{tgt}_pred"] = pred[f"{tgt}_pred"].values

# ── UL94 risk classification ──
def ul94_band(prob):
    if pd.isna(prob):
        return "N/A"
    if prob >= 0.60:
        return "High-confidence V-0"
    elif prob >= 0.40:
        return "Uncertain"
    else:
        return "Low V-0 probability"
df["UL94_Risk_Level"] = df["UL94_pred"].apply(ul94_band)

# ── Applicability Domain (kNN distance in training feature space) ──
MODEL_CONFIGS = {
    "LOI":      HERE.parent / "LOI/Result_Final_TabICL_Interpret",
    "Tg":       HERE.parent / "Tg/Result_Final_TabICL_Interpret",
    "TENSILE":  HERE.parent / "TENSILE/Result_Final_TabICL_Interpret",
    "UL94":     HERE.parent / "94/Result_Final_TabICL_Interpret_Cls",
}

# Load training data and compute AD for each target
ep_d = pd.read_csv(HERE.parent / "Experiment_EP_D.csv")
fr_d = pd.read_csv(HERE.parent / "Experiment_FR_D.csv")
cur_d = pd.read_csv(HERE.parent / "Experiment_Curing_D.csv")

# Handle FR_SMILES=0
zero_fr = exp["FR_SMILES"].astype(str) == "0"
if zero_fr.any():
    fr_d.loc[zero_fr.values] = 0

# Build experiment feature matrix (need formula features computed in exp)
# Re-compute formula features on exp
temp_cols = [f"Curing_Tem{i}" for i in range(1, 10)]
time_cols = [f"Curing_Time{i}" for i in range(1, 10)]
exp["T_max"] = exp[temp_cols].max(axis=1)
exp["t_total"] = exp[time_cols].sum(axis=1).fillna(0)
exp["Q_thermal"] = sum(exp[tc].fillna(0) * exp[tic].fillna(0) for tc, tic in zip(temp_cols, time_cols))
exp["EP_wt_fraction"] = (100.0 - exp["Flame_retardant_AdditionAmount(wt%)"].fillna(0) - exp["Curing_agent_AdditionAmount(wt%)"].fillna(0)) / 100.0

# Build experiment feature matrix
combined = {}
for col in ep_d.columns: combined[col] = ep_d[col].values
for col in fr_d.columns: combined[col] = fr_d[col].values
for col in cur_d.columns: combined[col] = cur_d[col].values
for col in ["EEW", "Flame_retardant_AdditionAmount(wt%)", "Curing_agent_AdditionAmount(wt%)",
            "T_max", "t_total", "Q_thermal", "EP_wt_fraction"]:
    combined[col] = exp[col].fillna(0).values
full_df = pd.DataFrame(combined).replace('na', np.nan).replace('NA', np.nan)

ad_distances = {}
for tgt, model_dir in MODEL_CONFIGS.items():
    sd = model_dir / "saved_data"
    if not sd.exists():
        subdirs = [d for d in model_dir.iterdir() if d.is_dir()]
        sd = subdirs[0] / "saved_data"
    with open(sd / "plot_data.pkl", "rb") as f:
        data = pickle.load(f)
    
    features = data["top_k_features"]
    X_train = data["X_train"]
    
    # Build experiment matrix in training order
    X_exp = np.zeros((len(full_df), len(features)), dtype=np.float32)
    for j, f in enumerate(features):
        if f in full_df.columns:
            X_exp[:, j] = pd.to_numeric(full_df[f], errors='coerce').fillna(0).values.astype(np.float32)
    
    # Scale both train and exp
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype(np.float32))
    X_exp_scaled = scaler.transform(X_exp)
    
    # kNN distance to training data (k=5, average distance)
    nn = NearestNeighbors(n_neighbors=5, metric='euclidean')
    nn.fit(X_train_scaled)
    distances, _ = nn.kneighbors(X_exp_scaled)
    avg_dist = distances.mean(axis=1)
    
    # AD threshold: 95th percentile of training distances to their own 5-NN
    train_dist, _ = nn.kneighbors(X_train_scaled)
    train_avg = train_dist.mean(axis=1)
    threshold = np.percentile(train_avg, 95)
    
    ad_distances[tgt] = {
        "distance": avg_dist,
        "threshold": threshold,
        "inside": avg_dist <= threshold,
    }

# Add AD info to table
# Use average AD across all 4 targets as composite AD
all_dists = np.column_stack([ad_distances[t]["distance"] for t in ["LOI", "Tg", "TENSILE", "UL94"]])
df["AD_Distance"] = all_dists.mean(axis=1)
# Inside if within threshold for ALL targets
inside_all = np.ones(len(df), dtype=bool)
for tgt in ["LOI", "Tg", "TENSILE", "UL94"]:
    inside_all &= ad_distances[tgt]["inside"]
df["Inside_AD"] = inside_all
df["Inside_AD_Label"] = df["Inside_AD"].map({True: "Inside AD", False: "Outside AD"})

# AD distances per target
for tgt in ["LOI", "Tg", "TENSILE", "UL94"]:
    df[f"AD_{tgt}_dist"] = ad_distances[tgt]["distance"]
    df[f"AD_{tgt}_threshold"] = ad_distances[tgt]["threshold"]

# ── Prediction errors (where experimental data exists) ──
error_targets = [
    ("LOI", "LOI"),
    ("Tg", "Tg"),
    ("TENSILE", "Tensile"),
    ("UL94", "UL94"),
]
for tgt_pred, tgt_exp in error_targets:
    exp_col = f"{tgt_exp}_exp"
    pred_col = f"{tgt_pred}_pred"
    if exp_col in df.columns and pred_col in df.columns:
        if tgt_pred != "UL94":
            df[f"{tgt_pred}_Error"] = df[pred_col] - df[exp_col]
            df[f"{tgt_pred}_AbsError"] = df[f"{tgt_pred}_Error"].abs()
        else:
            # UL94: classification comparison
            df["UL94_pred_class"] = (df[pred_col] >= 0.5).astype(int)
            df["UL94_Correct"] = np.where(
                df[exp_col].notna() & df[pred_col].notna(),
                df["UL94_pred_class"].values == df[exp_col].fillna(0).astype(int).values,
                np.nan
            )

# ── Save ──
out_path = OUT_DIR / "experiment_validation_full.csv"
df.to_csv(out_path, index=False)
print(f"Saved {out_path}")
print(f"\nRows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"\nExperimental data available:")
for tgt in ["LOI", "UL94", "Tensile", "Tg"]:
    n = df[f"{tgt}_exp"].notna().sum()
    print(f"  {tgt}_exp: {n}/{len(df)} rows")

print(f"\nAD Summary:")
print(f"  Inside AD: {df['Inside_AD'].sum()}/{len(df)}")
print(f"  Outside AD: {(~df['Inside_AD']).sum()}/{len(df)}")

print(f"\nUL94 Risk Distribution:")
print(df["UL94_Risk_Level"].value_counts().to_string())

# Print errors for rows with experimental data
print(f"\n=== Prediction Errors (rows with experimental data) ===")
for tgt_pred, tgt_exp in error_targets:
    exp_col = f"{tgt_exp}_exp"
    pred_col = f"{tgt_pred}_pred"
    if exp_col not in df.columns:
        continue
    valid = df[exp_col].notna()
    if valid.sum() == 0:
        print(f"  {tgt_pred}: no experimental data")
        continue
    if tgt_pred != "UL94":
        err_col = f"{tgt_pred}_AbsError"
        mae = df.loc[valid, err_col].mean()
        max_err = df.loc[valid, err_col].max()
        print(f"  {tgt_pred}: MAE={mae:.2f}, MaxAE={max_err:.2f} (n={valid.sum()})")
        for i in df[valid].index:
            print(f"    {df.loc[i, 'Sample_ID']}: exp={df.loc[i, exp_col]:.1f}, pred={df.loc[i, pred_col]:.1f}, err={df.loc[i, f'{tgt_pred}_Error']:+.1f}, AD={'IN' if df.loc[i, 'Inside_AD'] else 'OUT'}")
    else:
        correct = df.loc[valid, "UL94_Correct"].sum()
        print(f"    UL94 Accuracy: {correct}/{valid.sum()} = {correct/valid.sum():.1%}")
        for i in df[valid].index:
            exp_val = df.loc[i, exp_col]
            print(f"    {df.loc[i, 'Sample_ID']}: exp={exp_val:.0f}, pred_prob={df.loc[i, pred_col]:.3f} → class={int(df.loc[i, 'UL94_pred_class'])}, AD={'IN' if df.loc[i, 'Inside_AD'] else 'OUT'}")
