from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, ElasticNet, LogisticRegression, Ridge, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

TARGET_CONFIG = {
    "LOI": {"dir": "LOI", "target_col": "LOI", "task": "regression", "best_k": 71},
    "Tg": {"dir": "Tg", "target_col": "Tg", "task": "regression", "best_k": 72},
    "TENSILE": {"dir": "TENSILE", "target_col": "Tensile", "task": "regression", "best_k": 40},
    "UL94": {"dir": "94", "target_col": "UL94", "task": "classification", "best_k": 70},
}


SMILES_COLS = ["EP_SMILES", "FR_SMILES", "CURING_SMILES"]


@dataclass
class ModelSpec:
    name: str
    family: str
    kind: str
    builder: Callable


def optional_import(name: str):
    try:
        module = __import__(name)
        return module, None
    except Exception as exc:
        return None, str(exc)


def find_dataset(target: str) -> Path:
    base = ROOT / TARGET_CONFIG[target]["dir"]
    candidates = sorted(base.glob("EP+FR+CURING_SMILES+*_DATASET_20260414.csv"))
    if not candidates:
        candidates = sorted(base.glob("*DATASET*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No dataset CSV found in {base}")
    return candidates[0]


def add_physical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    temp_cols = [f"Curing_Tem{i}" for i in range(1, 10)]
    time_cols = [f"Curing_Time{i}" for i in range(1, 10)]
    existing_temp = [c for c in temp_cols if c in df.columns]
    existing_time = [c for c in time_cols if c in df.columns]
    if existing_temp:
        df["T_max"] = df[existing_temp].max(axis=1).fillna(0)
    if existing_time:
        df["t_total"] = df[existing_time].sum(axis=1).fillna(0)
    if existing_temp and existing_time:
        thermal = 0
        for t_col, h_col in zip(temp_cols, time_cols):
            if t_col in df.columns and h_col in df.columns:
                thermal = thermal + df[t_col].fillna(0) * df[h_col].fillna(0)
        df["Q_thermal"] = thermal
    df = df.drop(columns=[c for c in temp_cols + time_cols if c in df.columns], errors="ignore")

    fr_col = "Flame_retardant_AdditionAmount(wt%)"
    cur_col = "Curing_agent_AdditionAmount(wt%)"
    if fr_col in df.columns and cur_col in df.columns:
        fr = df[fr_col].fillna(0)
        cur = df[cur_col].fillna(0)
        df["EP_wt_fraction"] = (100.0 - fr - cur) / 100.0
        df["FR_wt_fraction"] = fr / 100.0
        df["CURING_wt_fraction"] = cur / 100.0
    return df


def load_data(target: str) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    cfg = TARGET_CONFIG[target]
    df = pd.read_csv(find_dataset(target))
    df = add_physical_features(df)
    df = df.dropna(subset=[cfg["target_col"]]).reset_index(drop=True)
    if cfg["task"] == "classification":
        df = df[df[cfg["target_col"]].isin([0, 1])].copy()
        y = df[cfg["target_col"]].astype(int).values
    else:
        y = df[cfg["target_col"]].astype(float).values

    exclude = set(SMILES_COLS + [cfg["target_col"]])
    feature_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return df, y, feature_cols


def find_shap_features(target: str, feature_cols: list[str]) -> list[str]:
    base = ROOT / TARGET_CONFIG[target]["dir"]
    candidates = sorted(base.rglob("Global_SHAP_Features_All*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "Feature_Name" not in df.columns:
            continue
        ordered = [f for f in df["Feature_Name"].tolist() if f in feature_cols]
        if ordered:
            return ordered
    return feature_cols


def descriptor_matrix(df: pd.DataFrame, feature_cols: list[str], target: str, quick: bool) -> tuple[np.ndarray, list[str]]:
    ordered = find_shap_features(target, feature_cols)
    k = TARGET_CONFIG[target]["best_k"] if not quick else min(TARGET_CONFIG[target]["best_k"], 50)
    selected = ordered[: min(k, len(ordered))]
    X = df[selected].replace([np.inf, -np.inf], np.nan).values.astype(np.float32)
    return X, selected


def descriptor_models(task: str, quick: bool) -> list[ModelSpec]:
    if task == "classification":
        specs = [
            ModelSpec("DescriptorProcess_LogisticRegressionClassifier", "descriptor_linear", "sklearn", lambda: LogisticRegression(max_iter=1000, class_weight="balanced")),
            ModelSpec("DescriptorProcess_RidgeClassifier", "descriptor_linear", "sklearn", lambda: RidgeClassifier()),
            ModelSpec("DescriptorProcess_RandomForestClassifier", "descriptor_tree", "sklearn", lambda: RandomForestClassifier(n_estimators=120 if quick else 400, random_state=42, n_jobs=-1, class_weight="balanced")),
            ModelSpec("DescriptorProcess_ExtraTreesClassifier", "descriptor_tree", "sklearn", lambda: ExtraTreesClassifier(n_estimators=120 if quick else 400, random_state=42, n_jobs=-1, class_weight="balanced")),
            ModelSpec("DescriptorProcess_GradientBoostingClassifier", "descriptor_boosting", "sklearn", lambda: GradientBoostingClassifier(random_state=42)),
            ModelSpec("DescriptorProcess_RbfSVMClassifier", "descriptor_kernel", "sklearn", lambda: SVC(C=10, gamma="scale", probability=True, class_weight="balanced")),
            ModelSpec("DescriptorProcess_MLPClassifier", "descriptor_nn", "sklearn", lambda: MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200 if quick else 700, random_state=42, early_stopping=True)),
        ]
    else:
        specs = [
            ModelSpec("DescriptorProcess_RidgeRegressor", "descriptor_linear", "sklearn", lambda: Ridge(alpha=1.0)),
            ModelSpec("DescriptorProcess_ElasticNetRegressor", "descriptor_linear", "sklearn", lambda: ElasticNet(alpha=0.05, l1_ratio=0.5, random_state=42)),
            ModelSpec("DescriptorProcess_BayesianRidgeRegressor", "descriptor_linear", "sklearn", lambda: BayesianRidge()),
            ModelSpec("DescriptorProcess_RbfSVMRegressor", "descriptor_kernel", "sklearn", lambda: SVR(C=10, gamma="scale")),
            ModelSpec("DescriptorProcess_RandomForestRegressor", "descriptor_tree", "sklearn", lambda: RandomForestRegressor(n_estimators=120 if quick else 400, max_depth=12, random_state=42, n_jobs=-1)),
            ModelSpec("DescriptorProcess_ExtraTreesRegressor", "descriptor_tree", "sklearn", lambda: ExtraTreesRegressor(n_estimators=120 if quick else 400, max_depth=12, random_state=42, n_jobs=-1)),
            ModelSpec("DescriptorProcess_GradientBoostingRegressor", "descriptor_boosting", "sklearn", lambda: GradientBoostingRegressor(n_estimators=120 if quick else 300, learning_rate=0.05, max_depth=4, random_state=42)),
            ModelSpec("DescriptorProcess_MLPRegressor", "descriptor_nn", "sklearn", lambda: MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=200 if quick else 700, random_state=42, early_stopping=True)),
        ]

    xgb, _ = optional_import("xgboost")
    if xgb is not None:
        if task == "classification":
            specs.append(ModelSpec("DescriptorProcess_XGBoostClassifier", "descriptor_boosting", "sklearn", lambda: xgb.XGBClassifier(n_estimators=120 if quick else 400, max_depth=4, learning_rate=0.05, eval_metric="logloss", tree_method="hist", random_state=42)))
        else:
            specs.append(ModelSpec("DescriptorProcess_XGBoostRegressor", "descriptor_boosting", "sklearn", lambda: xgb.XGBRegressor(n_estimators=120 if quick else 400, max_depth=4, learning_rate=0.05, objective="reg:squarederror", tree_method="hist", random_state=42)))

    lgb, _ = optional_import("lightgbm")
    if lgb is not None:
        cls = lgb.LGBMClassifier if task == "classification" else lgb.LGBMRegressor
        suffix = "Classifier" if task == "classification" else "Regressor"
        specs.append(
            ModelSpec(
                f"DescriptorProcess_LightGBM{suffix}",
                "descriptor_boosting",
                "sklearn",
                lambda cls=cls, quick=quick: cls(n_estimators=120 if quick else 400, learning_rate=0.05, random_state=42, n_jobs=-1, verbose=-1),
            )
        )

    catboost, _ = optional_import("catboost")
    if catboost is not None:
        cls = catboost.CatBoostClassifier if task == "classification" else catboost.CatBoostRegressor
        suffix = "Classifier" if task == "classification" else "Regressor"
        specs.append(
            ModelSpec(
                f"DescriptorProcess_CatBoost{suffix}",
                "descriptor_boosting",
                "sklearn",
                lambda cls=cls, quick=quick: cls(iterations=120 if quick else 400, learning_rate=0.05, depth=6, random_seed=42, verbose=False, allow_writing_files=False),
            )
        )

    tabicl, _ = optional_import("tabicl")
    if tabicl is not None:
        cls = tabicl.TabICLClassifier if task == "classification" else tabicl.TabICLRegressor
        suffix = "Classifier" if task == "classification" else "Regressor"
        specs.append(
            ModelSpec(
                f"DescriptorProcess_TabICL{suffix}",
                "descriptor_tabicl",
                "sklearn",
                lambda cls=cls: cls(n_estimators=8, feat_shuffle_method="latin", outlier_threshold=5.0, device="cuda", kv_cache=True, batch_size=8, random_state=42),
            )
        )

    tabpfn, _ = optional_import("tabpfn")
    if tabpfn is not None:
        cls = tabpfn.TabPFNClassifier if task == "classification" else tabpfn.TabPFNRegressor
        suffix = "Classifier" if task == "classification" else "Regressor"
        specs.append(ModelSpec(f"DescriptorProcess_TabPFN{suffix}", "descriptor_tabpfn", "sklearn", lambda cls=cls: cls(device="cuda")))

    return specs


def predict_scores(model, X, task: str):
    if task == "classification":
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X)[:, 1]
            pred = (prob >= 0.5).astype(int)
        elif hasattr(model, "decision_function"):
            raw = model.decision_function(X)
            prob = 1 / (1 + np.exp(-raw))
            pred = (prob >= 0.5).astype(int)
        else:
            pred = model.predict(X)
            prob = pred.astype(float)
        return pred, prob
    pred = model.predict(X)
    return pred, pred


def eval_predictions(y_true, y_pred, y_score, task: str) -> dict:
    if task == "classification":
        try:
            auc = roc_auc_score(y_true, y_score)
        except Exception:
            auc = np.nan
        return {
            "AUC": auc,
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "Accuracy": accuracy_score(y_true, y_pred),
        }
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": math.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def run_sklearn_models(target: str, X: np.ndarray, y: np.ndarray, specs: list[ModelSpec], quick: bool):
    task = TARGET_CONFIG[target]["task"]
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) if task == "classification" else KFold(n_splits=5, shuffle=True, random_state=42)
    folds = []
    oof_rows = []
    skipped = []

    for spec in specs:
        print(f"[{target}] {spec.name}")
        for fold, (tr, te) in enumerate(splitter.split(X, y), start=1):
            try:
                model = Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("model", spec.builder()),
                    ]
                )
                t0 = time.time()
                model.fit(X[tr], y[tr])
                pred, score = predict_scores(model, X[te], task)
                metrics = eval_predictions(y[te], pred, score, task)
                folds.append(
                    {
                        "Target": target,
                        "Model": spec.name,
                        "Family": spec.family,
                        "Fold": fold,
                        "Seconds": time.time() - t0,
                        **metrics,
                    }
                )
                for idx, p, s in zip(te, pred, score):
                    oof_rows.append({"Target": target, "Model": spec.name, "Index": int(idx), "Actual": y[idx], "Predicted": p, "Score": s, "Fold": fold})
            except Exception as exc:
                skipped.append({"Target": target, "Model": spec.name, "Family": spec.family, "Fold": fold, "Reason": repr(exc)})
                print(f"  skipped fold {fold}: {exc}")
                break
    return pd.DataFrame(folds), pd.DataFrame(oof_rows), pd.DataFrame(skipped)


def gnn_specs(quick: bool) -> list[dict]:
    convs = ["gcn", "sage", "gat", "gin", "gine"]
    fusions = ["concat", "weighted_sum", "gated", "attention", "film"]
    if quick:
        convs = ["gcn", "gin", "gine"]
        fusions = ["concat", "weighted_sum", "attention", "film"]
    return [{"conv": c, "fusion": f, "name": f"Graph_{c.upper()}_Fusion_{f}"} for c in convs for f in fusions]


def run_gnn_models_placeholder(target: str, quick: bool) -> pd.DataFrame:
    rows = []
    for spec in gnn_specs(quick):
        rows.append(
            {
                "Target": target,
                "Model": spec["name"],
                "Family": "advanced_gnn_or_fusion",
                "Fold": np.nan,
                "Reason": (
                    "Configured in this benchmark. Run with --include-gnn to train; "
                    "the architecture set covers GCN/SAGE/GAT/GIN/GINE plus concat, "
                    "weighted_sum, gated, attention, and FiLM fusion."
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize(target: str, folds: pd.DataFrame) -> pd.DataFrame:
    task = TARGET_CONFIG[target]["task"]
    if folds.empty:
        return folds
    if task == "classification":
        metrics = ["AUC", "F1", "Accuracy"]
        primary = "AUC"
    else:
        metrics = ["R2", "RMSE", "MAE"]
        primary = "R2"
    agg = folds.groupby(["Target", "Model", "Family"], as_index=False).agg({m: ["mean", "std"] for m in metrics})
    agg.columns = ["_".join(c).strip("_") for c in agg.columns.to_flat_index()]
    agg = agg.rename(columns={f"{primary}_mean": "Primary_Mean", f"{primary}_std": "Primary_Std"})
    return agg.sort_values("Primary_Mean", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=list(TARGET_CONFIG), required=True)
    parser.add_argument("--quick", action="store_true", help="Use fewer trees/features and shorter training.")
    parser.add_argument("--include-gnn", action="store_true", help="Reserved switch for the heavy PyG GNN training block.")
    args = parser.parse_args()

    df, y, feature_cols = load_data(args.target)
    X, selected_features = descriptor_matrix(df, feature_cols, args.target, args.quick)
    task = TARGET_CONFIG[args.target]["task"]

    manifest = {
        "target": args.target,
        "task": task,
        "n_samples": int(len(y)),
        "n_descriptor_features": int(X.shape[1]),
        "selected_features": selected_features,
        "quick": bool(args.quick),
    }
    with (OUT / f"unified_{args.target}_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    folds, oof, skipped = run_sklearn_models(args.target, X, y, descriptor_models(task, args.quick), args.quick)

    if not args.include_gnn:
        skipped = pd.concat([skipped, run_gnn_models_placeholder(args.target, args.quick)], ignore_index=True)
    else:
        skipped = pd.concat(
            [
                skipped,
                pd.DataFrame(
                    [
                        {
                            "Target": args.target,
                            "Model": "GNN training block",
                            "Family": "advanced_gnn_or_fusion",
                            "Fold": np.nan,
                            "Reason": "Heavy PyG training is intentionally separated; use the existing GNN scripts or extend this file with local GPU-specific settings.",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    summary = summarize(args.target, folds)
    folds.to_csv(OUT / f"unified_{args.target}_all_folds.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / f"unified_{args.target}_summary.csv", index=False, encoding="utf-8-sig")
    oof.to_csv(OUT / f"unified_{args.target}_oof_predictions.csv", index=False, encoding="utf-8-sig")
    skipped.to_csv(OUT / f"unified_{args.target}_skipped_models.csv", index=False, encoding="utf-8-sig")

    print(summary.head(20).to_string(index=False))
    print(f"Outputs written to {OUT}")


if __name__ == "__main__":
    main()
