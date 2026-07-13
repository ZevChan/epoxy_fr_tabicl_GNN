"""
PUBCHEM prediction pipeline: SMILES batch (no 3D) + median imputation.
Filters: no halogens + contains P/S/B/Si.
"""
import os, sys, json, pickle, re, warnings, time, subprocess, tempfile
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
ALVADESC_CLI = r"D:\Program Files\Alvascience\alvaDesc\alvaDescCLI.exe"
PUBCHEM_DIR = Path(r"D:\translated_text\translated_text\translated_text\27.SCFC\sdf_files\translated_textSDF_20250328\smiles_chunks")

# Formula
EP_SMILES = "CC(C)(C1=CC=C(OCC2CO2)C=C1)C1=CC=C(OCC2CO2)C=C1"  # DGEBA E51
CURING_SMILES = "NC1=CC=C(CC2=CC=C(N)C=C2)C=C1"  # DDM
EEW = 189
FR_AMOUNTS = [1, 2, 5, 10, 15, 20]
CURE_SCHEDULE = [(80, 2), (120, 2), (150, 2)]

MODEL_CONFIGS = {
    "LOI":      {"dir": HERE.parent / "LOI/Result_Final_TabICL_Interpret",      "type": "regression", "params": {"n_estimators": 8, "feat_shuffle_method": "latin", "outlier_threshold": 3.0}},
    "Tg":       {"dir": HERE.parent / "Tg/Result_Final_TabICL_Interpret",       "type": "regression", "params": {"n_estimators": 4, "feat_shuffle_method": "latin", "outlier_threshold": 3.0}},
    "TENSILE":  {"dir": HERE.parent / "TENSILE/Result_Final_TabICL_Interpret",  "type": "regression", "params": {"n_estimators": 8, "feat_shuffle_method": "latin", "outlier_threshold": 3.0}},
    "UL94":     {"dir": HERE.parent / "94/Result_Final_TabICL_Interpret_Cls",   "type": "classification", "params": {"n_estimators": 8, "feat_shuffle_method": "latin", "outlier_threshold": 3.0}},
}

OUT_DIR = HERE.parent / "pubchem_predictions"
OUT_DIR.mkdir(exist_ok=True)

# SMILES filters
HALOGEN_RE = re.compile(r'(?<![A-Za-z\[0-9])(?:Cl|Br|\[F\]|\[I\]|(?<![A-Z])F(?!e|m|r)|(?<![A-Z])I(?!r|n))(?![a-z])')
TARGET_RE = re.compile(r'(?<![A-Za-z\[0-9])(?:\[Si\]|(?<![A-Z])P(?!b|d|t|m|o|a|u)|(?<![A-Z])S(?!c|e|r|g|n|b|m|i)|(?<![A-Z])B(?!e|i|r|h))(?![a-z])')


def load_model_data(config):
    """Load training data + feature list from saved_data."""
    result_dir = config["dir"]
    sd = result_dir / "saved_data"
    if not sd.exists():
        subdirs = [d for d in result_dir.iterdir() if d.is_dir()]
        sd = subdirs[0] / "saved_data"
    with open(sd / "plot_data.pkl", "rb") as f:
        data = pickle.load(f)
    return data["top_k_features"], data["X_train"], data["y_train"]


def get_all_descriptor_names():
    """Union of all descriptor names needed across 4 targets (strip prefix)."""
    all_names = set()
    for tgt, cfg in MODEL_CONFIGS.items():
        features, _, _ = load_model_data(cfg)
        for feat in features:
            for prefix in ["FR_", "CURING_", "EP_"]:
                if feat.startswith(prefix):
                    name = feat[len(prefix):]
                    if prefix == "EP_" and name == "wt_fraction":
                        continue
                    all_names.add(name)
                    break
    return sorted(all_names)


def compute_training_medians(desc_names):
    """Compute median of each descriptor from training data (FR descriptors only)."""
    # Read FR training descriptors
    fr_csv = HERE.parent / "94/FR_SMILES_D.txt"
    df = pd.read_csv(fr_csv, sep="\t")
    medians = {}
    for name in desc_names:
        col = f"FR_{name}"
        if col in df.columns:
            medians[name] = df[col].median()
        else:
            medians[name] = 0.0  # fallback
    # Also read CURING descriptors
    cur_csv = HERE.parent / "94/CURING_SMILES_D.txt"
    df_c = pd.read_csv(cur_csv, sep="\t")
    for name in desc_names:
        col = f"CURING_{name}"
        if col in df_c.columns:
            medians[f"CURING_{name}"] = df_c[col].median()
    # EP descriptors
    ep_csv = HERE.parent / "94/EP_SMILES_D.txt"
    df_e = pd.read_csv(ep_csv, sep="\t")
    for name in desc_names:
        col = f"EP_{name}"
        if col in df_e.columns:
            medians[f"EP_{name}"] = df_e[col].median()
    return medians


def compute_formula_params(fr_wt_pct):
    ep_plus_ddm = 100.0 - fr_wt_pct
    ep_wt = 0.8 * ep_plus_ddm
    ddm_wt = 0.2 * ep_plus_ddm
    t_total = sum(t for _, t in CURE_SCHEDULE)
    q_thermal = sum(temp * t for temp, t in CURE_SCHEDULE)
    t_max = max(temp for temp, _ in CURE_SCHEDULE)
    return {
        "Flame_retardant_AdditionAmount(wt%)": fr_wt_pct,
        "Curing_agent_AdditionAmount(wt%)": ddm_wt,
        "EP_wt_fraction": ep_wt / 100.0,
        "EEW": EEW,
        "t_total": t_total,
        "Q_thermal": q_thermal,
        "T_max": t_max,
    }


def build_feature_vector(fr_descs, ep_descs, cur_descs, formula, feature_order, medians):
    row = []
    for feat in feature_order:
        if feat.startswith("FR_"):
            val = fr_descs.get(feat[3:], np.nan)
        elif feat.startswith("EP_"):
            name = feat[3:]
            if name == "wt_fraction":
                val = formula["EP_wt_fraction"]
            else:
                val = ep_descs.get(name, np.nan)
        elif feat.startswith("CURING_"):
            val = cur_descs.get(feat[7:], np.nan)
        else:
            val = formula.get(feat, np.nan)
        if pd.isna(val):
            val = medians.get(feat, medians.get(feat.replace("FR_", "").replace("CURING_", "").replace("EP_", ""), 0.0))
        row.append(float(val))
    return np.array(row, dtype=np.float32)


def batch_compute_descriptors(smiles_list, desc_names):
    """Write SMILES to temp file, run alvaDesc once, parse output."""
    if not smiles_list:
        return []
    # Write temp SMILES file
    tmp_smi = str(OUT_DIR / f"tmp_batch_{os.getpid()}.smi")
    with open(tmp_smi, "w") as f:
        for smi in smiles_list:
            f.write(smi + " mol\n")
    
    desc_str = ",".join(desc_names)
    tmp_out = str(OUT_DIR / f"tmp_batch_{os.getpid()}_out.txt")
    
    try:
        subprocess.run(
            [ALVADESC_CLI, f"--input={tmp_smi}", "--inputtype=SMILES",
             f"--descriptors={desc_str}", f"--output={tmp_out}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120
        )
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"  WARNING: alvaDesc failed/timeout for {len(smiles_list)} mols, skipping")
        for p in [tmp_smi, tmp_out]:
            if os.path.exists(p):
                os.unlink(p)
        return []
    
    results = []
    if os.path.exists(tmp_out):
        with open(tmp_out) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                vals = line.split("\t")
                d = {}
                for name, val in zip(desc_names, vals):
                    try:
                        d[name] = float(val) if val != "na" else np.nan
                    except ValueError:
                        d[name] = np.nan
                results.append(d)
    
    # Cleanup
    for p in [tmp_smi, tmp_out]:
        if os.path.exists(p):
            os.unlink(p)
    return results


def main():
    from xgboost import XGBRegressor, XGBClassifier
    from sklearn.preprocessing import StandardScaler
    
    print("=" * 60)
    print("PUBLICHEM PREDICTION PIPELINE (SMILES batch + median imputation)")
    print("=" * 60)
    
    # ── Step 1: Pre-compute EP/CURING descriptors ──
    print("\n[1/5] Computing EP & CURING descriptors...")
    desc_names = get_all_descriptor_names()
    print(f"  {len(desc_names)} unique descriptor names")
    
    # EP (DGEBA) - compute once via batch
    ep_results = batch_compute_descriptors([EP_SMILES], desc_names)
    ep_descs = ep_results[0] if ep_results else {}
    ep_nan = sum(1 for v in ep_descs.values() if pd.isna(v))
    print(f"  EP: {len(ep_descs)} computed, {ep_nan} NaN")
    
    # CURING (DDM)
    cur_results = batch_compute_descriptors([CURING_SMILES], desc_names)
    cur_descs = cur_results[0] if cur_results else {}
    cur_nan = sum(1 for v in cur_descs.values() if pd.isna(v))
    print(f"  CURING: {len(cur_descs)} computed, {cur_nan} NaN")
    
    # ── Step 2: Training data medians ──
    print("\n[2/5] Computing training data medians for imputation...")
    medians = compute_training_medians(desc_names)
    print(f"  {len(medians)} median values")
    
    # ── Step 3: Train models ──
    print("\n[3/5] Training TabICL models...")
    models = {}
    scalers = {}
    feature_sets = {}
    for tgt, cfg in MODEL_CONFIGS.items():
        features, X_train, y_train = load_model_data(cfg)
        feature_sets[tgt] = features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        scalers[tgt] = scaler
        
        ModelClass = XGBClassifier if cfg["type"] == "classification" else XGBRegressor
        model = ModelClass(n_estimators=200, max_depth=6, learning_rate=0.1,
                          subsample=0.8, colsample_bytree=0.8, random_state=42,
                          n_jobs=-1)
        model.fit(X_scaled, y_train)
        models[tgt] = model
        print(f"  {tgt}: {len(features)} features, {len(X_train)} samples - trained")
    
    # Save models
    with open(OUT_DIR / "trained_models_v2.pkl", "wb") as f:
        pickle.dump({"models": models, "scalers": scalers, "feature_sets": feature_sets,
                     "ep_descs": ep_descs, "cur_descs": cur_descs, "medians": medians}, f)
    
    # ── Step 4: Process PUBCHEM ──
    print("\n[4/5] Processing PUBCHEM chunks...")
    chunk_files = sorted(PUBCHEM_DIR.glob("smiles_chunk_*.txt"))
    print(f"  {len(chunk_files)} chunk files")
    
    all_predictions = []
    total_processed = 0
    total_kept = 0
    t_start = time.time()
    
    for fi, chunk_path in enumerate(chunk_files):
        chunk_name = chunk_path.stem
        
        # Skip already processed chunks
        out_csv = OUT_DIR / f"predictions_{chunk_name}.csv"
        if out_csv.exists():
            print(f"  [{fi+1}/{len(chunk_files)}] {chunk_name}: already done, skipping", flush=True)
            continue
        
        # Read + filter
        with open(chunk_path) as f:
            all_smiles = [line.strip() for line in f if line.strip()]
        
        filtered = [s for s in all_smiles if not HALOGEN_RE.search(s) and TARGET_RE.search(s)]
        total_processed += len(all_smiles)
        total_kept += len(filtered)
        
        if not filtered:
            print(f"  [{fi+1}/{len(chunk_files)}] {chunk_name}: 0 kept (skip)")
            continue
        
        # Batch alvaDesc
        print(f"  [{fi+1}/{len(chunk_files)}] {chunk_name}: {len(filtered)} mols, computing descriptors...", flush=True)
        batch_results = batch_compute_descriptors(filtered, desc_names)
        if not batch_results:
            print(f"    alvaDesc failed, skipping chunk", flush=True)
            continue
        print(f"    descriptors done, building features...", flush=True)

        # Build all feature vectors at once
        n_mols = len(filtered)
        n_rows = n_mols * len(FR_AMOUNTS)
        all_rows = []
        for smi, fr_descs in zip(filtered, batch_results):
            for fr_amt in FR_AMOUNTS:
                formula = compute_formula_params(fr_amt)
                row = {"SMILES": smi, "Chunk": chunk_name, "FR_wt_pct": fr_amt}
                # Pre-build feature vectors for each target
                row_feats = {}
                for tgt in ["LOI", "Tg", "TENSILE", "UL94"]:
                    feat_vec = build_feature_vector(fr_descs, ep_descs, cur_descs,
                                                    formula, feature_sets[tgt], medians)
                    row_feats[tgt] = np.nan_to_num(feat_vec, nan=0.0)
                row["_feats"] = row_feats
                all_rows.append(row)

        # Batch predict per target (XGBoost handles large batches natively)
        print(f"    predicting {n_rows} samples...", flush=True)
        for tgt in ["LOI", "Tg", "TENSILE", "UL94"]:
            X_batch = np.stack([r["_feats"][tgt] for r in all_rows])
            if MODEL_CONFIGS[tgt]["type"] == "classification":
                preds = models[tgt].predict_proba(X_batch)[:, 1]
            else:
                preds = models[tgt].predict(X_batch)
            for i in range(n_rows):
                all_rows[i][f"{tgt}_pred"] = float(preds[i])

        # Clean up temp feature arrays
        for r in all_rows:
            del r["_feats"]
        
        # Save chunk
        pd.DataFrame(all_rows).drop(columns=["_feats"], errors='ignore').to_csv(
            OUT_DIR / f"predictions_{chunk_name}.csv", index=False)
        all_predictions.extend(all_rows)
        
        elapsed = time.time() - t_start
        rate = (fi + 1) / max(elapsed, 1) * 3600
        eta = (len(chunk_files) - fi - 1) / max(rate, 0.01)
        print(f"    done ({len(all_rows)} preds) | {rate:.0f} chunks/h, ETA {eta:.1f}h", flush=True)
    
    # ── Step 5: Final merge ──
    print(f"\n[5/5] Saving combined results...")
    all_df = pd.DataFrame(all_predictions)
    all_df.to_csv(OUT_DIR / "predictions_all_pubchem.csv", index=False)
    
    total_time = (time.time() - t_start) / 3600
    print(f"\nDone! {total_kept} molecules × 6 FR amounts = {len(all_predictions)} predictions")
    print(f"Total SMILES scanned: {total_processed:,}")
    print(f"Total kept: {total_kept:,} ({total_kept/max(total_processed,1)*100:.1f}%)")
    print(f"Time: {total_time:.1f} hours")
    print(f"Output: {OUT_DIR / 'predictions_all_pubchem.csv'}")


if __name__ == "__main__":
    main()
