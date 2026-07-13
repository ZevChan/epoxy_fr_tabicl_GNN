from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent

FORMULA_PATH = ROOT / "Experiment.csv"
EP_DESC_PATH = ROOT / "Experiment_EP_D.csv"
FR_DESC_PATH = ROOT / "Experiment_FR_D.csv"
CURING_DESC_PATH = ROOT / "Experiment_Curing_D.csv"

TRAIN_PATH = ROOT / "94" / "EP+FR+CURING_SMILES+translated_text_DATASET_20260414.csv"
SHAP_PATH = ROOT / "94" / "Result_TabICL_Elbow_Cls_20260415_1720" / "Global_SHAP_Features_All.csv"

TARGET_COL = "UL94"
BEST_K = 70
BEST_PARAMS = {
    "n_estimators": 8,
    "feat_shuffle_method": "latin",
    "outlier_threshold": 3.0,
}

SMILES_COLS = ["EP_SMILES", "FR_SMILES", "CURING_SMILES"]


def is_fr_zero(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    return text.isin(["0", "0.0", "nan", "none", ""])


def to_numeric_frame(df: pd.DataFrame, skip: set[str] | None = None) -> pd.DataFrame:
    skip = skip or set()
    out = df.copy()
    for col in out.columns:
        if col not in skip:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def add_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    temp_cols = [f"Curing_Tem{i}" for i in range(1, 10)]
    time_cols = [f"Curing_Time{i}" for i in range(1, 10)]
    existing_temp = [c for c in temp_cols if c in out.columns]
    existing_time = [c for c in time_cols if c in out.columns]

    if existing_temp:
        out["T_max"] = out[existing_temp].max(axis=1).fillna(0)
    if existing_time:
        out["t_total"] = out[existing_time].sum(axis=1).fillna(0)
    if existing_temp and existing_time:
        thermal = 0
        for t_col, h_col in zip(temp_cols, time_cols):
            if t_col in out.columns and h_col in out.columns:
                thermal = thermal + out[t_col].fillna(0) * out[h_col].fillna(0)
        out["Q_thermal"] = thermal

    drop_cols = [c for c in temp_cols + time_cols if c in out.columns]
    out = out.drop(columns=drop_cols, errors="ignore")

    fr_col = "Flame_retardant_AdditionAmount(wt%)"
    cur_col = "Curing_agent_AdditionAmount(wt%)"
    if fr_col in out.columns and cur_col in out.columns:
        fr = out[fr_col].fillna(0)
        cur = out[cur_col].fillna(0)
        out["EP_wt_fraction"] = (100.0 - fr - cur) / 100.0
        out["FR_wt_fraction"] = fr / 100.0
        out["CURING_wt_fraction"] = cur / 100.0
    return out


def combined_median(train_df: pd.DataFrame, experiment_df: pd.DataFrame, feature: str) -> float:
    values = []
    if feature in train_df.columns:
        values.append(pd.to_numeric(train_df[feature], errors="coerce"))
    if feature in experiment_df.columns:
        values.append(pd.to_numeric(experiment_df[feature], errors="coerce"))
    if not values:
        return 0.0
    merged = pd.concat(values, ignore_index=True).replace([np.inf, -np.inf], np.nan).dropna()
    if merged.empty:
        return 0.0
    return float(merged.median())


def descriptor_cols_for_file(top_features: list[str], prefix: str) -> list[str]:
    pseudo_features = {
        "EP_wt_fraction",
        "FR_wt_fraction",
        "CURING_wt_fraction",
        "T_max",
        "t_total",
        "Q_thermal",
    }
    return [
        f
        for f in top_features
        if f.startswith(prefix)
        and f not in pseudo_features
    ]


def read_selected_descriptor(path: Path, selected_cols: list[str], header_cols: list[str], row_count: int) -> pd.DataFrame:
    existing = [c for c in selected_cols if c in header_cols]
    if not existing:
        return pd.DataFrame(index=range(row_count))
    return pd.read_csv(path, usecols=existing)


def load_inputs(top_features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    formula = pd.read_csv(FORMULA_PATH)
    ep_cols = descriptor_cols_for_file(top_features, "EP_")
    fr_cols = descriptor_cols_for_file(top_features, "FR_")
    curing_cols = descriptor_cols_for_file(top_features, "CURING_")

    ep_header = pd.read_csv(EP_DESC_PATH, nrows=0).columns.tolist()
    fr_header = pd.read_csv(FR_DESC_PATH, nrows=0).columns.tolist()
    curing_header = pd.read_csv(CURING_DESC_PATH, nrows=0).columns.tolist()

    ep_desc = read_selected_descriptor(EP_DESC_PATH, ep_cols, ep_header, len(formula))
    fr_desc = read_selected_descriptor(FR_DESC_PATH, fr_cols, fr_header, len(formula))
    curing_desc = read_selected_descriptor(CURING_DESC_PATH, curing_cols, curing_header, len(formula))

    row_counts = {len(formula), len(ep_desc), len(fr_desc), len(curing_desc)}
    if len(row_counts) != 1:
        raise ValueError(
            "Input row counts do not match: "
            f"Experiment={len(formula)}, EP={len(ep_desc)}, FR={len(fr_desc)}, CURING={len(curing_desc)}"
        )
    missing_formula_cols = [c for c in SMILES_COLS if c not in formula.columns]
    if missing_formula_cols:
        raise ValueError(f"Experiment.csv is missing columns: {missing_formula_cols}")
    total_descriptor_cols = len(ep_header) + len(fr_header) + len(curing_header)
    return formula, ep_desc, fr_desc, curing_desc, total_descriptor_cols


def build_experiment_features(
    top_features: list[str],
    train_processed: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    formula, ep_desc, fr_desc, curing_desc, total_descriptor_cols = load_inputs(top_features)
    fr_zero = is_fr_zero(formula["FR_SMILES"])

    ep_desc = to_numeric_frame(ep_desc)
    fr_desc = to_numeric_frame(fr_desc)
    curing_desc = to_numeric_frame(curing_desc)

    descriptor_cols = list(ep_desc.columns) + list(fr_desc.columns) + list(curing_desc.columns)

    combined = pd.concat(
        [
            formula.reset_index(drop=True),
            ep_desc.reset_index(drop=True),
            fr_desc.reset_index(drop=True),
            curing_desc.reset_index(drop=True),
        ],
        axis=1,
    )
    combined = to_numeric_frame(combined, skip=set(SMILES_COLS))

    # User rule: descriptor NAs on rows with FR_SMILES == 0 represent absent FR and become 0.
    zero_descriptor_na_count = int(combined.loc[fr_zero, descriptor_cols].isna().sum().sum())
    combined.loc[fr_zero, descriptor_cols] = combined.loc[fr_zero, descriptor_cols].fillna(0)

    combined = add_physical_features(combined)
    train_processed_for_median = train_processed.copy()

    log_rows = []
    feature_matrix = pd.DataFrame(index=combined.index)

    for feature in top_features:
        if feature not in combined.columns:
            fill_value = combined_median(train_processed_for_median, combined, feature)
            feature_matrix[feature] = fill_value
            log_rows.append(
                {
                    "feature": feature,
                    "status": "missing_in_experiment_filled",
                    "fr_zero_na_filled": 0,
                    "nonzero_na_filled": len(combined),
                    "fill_value": fill_value,
                }
            )
            continue

        values = pd.to_numeric(combined[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        zero_na = int(values.loc[fr_zero].isna().sum())
        if zero_na:
            values.loc[fr_zero] = values.loc[fr_zero].fillna(0)

        nonzero_na = int(values.loc[~fr_zero].isna().sum())
        fill_value = np.nan
        if nonzero_na:
            fill_value = combined_median(train_processed_for_median, combined, feature)
            values.loc[~fr_zero] = values.loc[~fr_zero].fillna(fill_value)

        # Final guard for features whose median cannot be computed.
        final_na = int(values.isna().sum())
        if final_na:
            fallback = combined_median(train_processed_for_median, combined, feature)
            values = values.fillna(fallback)
            fill_value = fallback

        feature_matrix[feature] = values.astype(float)
        log_rows.append(
            {
                "feature": feature,
                "status": "ok" if zero_na == 0 and nonzero_na == 0 else "filled",
                "fr_zero_na_filled": zero_na,
                "nonzero_na_filled": nonzero_na,
                "fill_value": fill_value,
            }
        )

    summary = pd.DataFrame(
        [
            {"item": "rows", "value": len(combined)},
            {"item": "fr_smiles_zero_rows", "value": int(fr_zero.sum())},
            {"item": "descriptor_na_filled_on_fr_zero_rows", "value": zero_descriptor_na_count},
            {"item": "top_features", "value": len(top_features)},
            {"item": "ignored_descriptor_columns_not_used_by_model", "value": total_descriptor_cols - len(descriptor_cols)},
        ]
    )
    return formula, combined, feature_matrix, pd.DataFrame(log_rows), summary


def main() -> int:
    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / f"Experiment_TabICL_UL94_Predictions_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    shap_df = pd.read_csv(SHAP_PATH)
    top_features = shap_df["Feature_Name"].tolist()[:BEST_K]

    train_raw = pd.read_csv(TRAIN_PATH)
    train_raw = to_numeric_frame(train_raw, skip=set(SMILES_COLS))
    train_processed = add_physical_features(train_raw)
    train_valid = train_processed.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    train_valid = train_valid[train_valid[TARGET_COL].isin([0, 1])].reset_index(drop=True)

    missing_train_features = [f for f in top_features if f not in train_valid.columns]
    if missing_train_features:
        raise ValueError(f"Training dataset is missing top features: {missing_train_features}")

    formula, combined, feature_matrix, preprocess_log, summary = build_experiment_features(
        top_features, train_valid
    )

    X_train_raw = train_valid[top_features].apply(pd.to_numeric, errors="coerce")
    train_fill_values = {}
    for feature in top_features:
        fill_value = combined_median(train_valid, combined, feature)
        train_fill_values[feature] = fill_value
        X_train_raw[feature] = X_train_raw[feature].replace([np.inf, -np.inf], np.nan).fillna(fill_value)

    X_new_raw = feature_matrix[top_features].replace([np.inf, -np.inf], np.nan)
    for feature in top_features:
        X_new_raw[feature] = X_new_raw[feature].fillna(train_fill_values[feature])

    y_train = train_valid[TARGET_COL].astype(int).values
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw.values.astype(float))
    X_new_scaled = scaler.transform(X_new_raw.values.astype(float))

    combined.to_csv(out_dir / "Experiment_TabICL_UL94_Combined_Processed.csv", index=False)
    X_new_raw.to_csv(out_dir / "Experiment_TabICL_UL94_Feature_Matrix.csv", index=False)
    preprocess_log.to_csv(out_dir / "Experiment_TabICL_UL94_Preprocess_Log.csv", index=False)
    summary.to_csv(out_dir / "Experiment_TabICL_UL94_Preprocess_Summary.csv", index=False)

    try:
        import torch
        from tabicl import TabICLClassifier
    except Exception as exc:
        error_path = out_dir / "TabICL_IMPORT_ERROR.txt"
        error_path.write_text(
            "Feature preprocessing completed, but TabICL could not be imported.\n"
            f"Python: {sys.executable}\n"
            f"Error: {exc!r}\n",
            encoding="utf-8",
        )
        print(f"ERROR: TabICL import failed. Preprocessed files are in: {out_dir}")
        print(f"ERROR_DETAIL: {exc!r}")
        return 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TabICLClassifier(
        **BEST_PARAMS,
        device=device,
        kv_cache=True,
        batch_size=8,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)
    probs = model.predict_proba(X_new_scaled)[:, 1]
    preds = model.predict(X_new_scaled)

    results = formula.copy()
    results.insert(0, "Sample_Index", np.arange(len(results)))
    results.insert(1, "Predicted_UL94_Class", preds.astype(int))
    results.insert(2, "Predicted_UL94_Probability", probs.astype(float))
    results.insert(3, "TabICL_Model", "UL94_K70_n8_latin_outlier3.0")
    results.to_csv(out_dir / "Experiment_TabICL_UL94_Predictions.csv", index=False)

    report = [
        "Experiment TabICL UL94 prediction report",
        f"timestamp: {timestamp}",
        f"python: {sys.executable}",
        f"device: {device}",
        f"training_rows: {len(train_valid)}",
        f"experiment_rows: {len(results)}",
        f"fr_smiles_zero_rows: {int(is_fr_zero(formula['FR_SMILES']).sum())}",
        f"features: {BEST_K}",
        f"best_params: {BEST_PARAMS}",
        f"predicted_positive_count: {int(np.sum(preds == 1))}",
        f"predicted_probability_min: {float(np.min(probs)):.6f}",
        f"predicted_probability_median: {float(np.median(probs)):.6f}",
        f"predicted_probability_max: {float(np.max(probs)):.6f}",
    ]
    (out_dir / "Experiment_TabICL_UL94_Report.txt").write_text("\n".join(report), encoding="utf-8")

    print(f"OK: predictions written to {out_dir}")
    print(results[["Sample_Index", "Predicted_UL94_Class", "Predicted_UL94_Probability"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
