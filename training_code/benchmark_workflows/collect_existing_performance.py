from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

TARGETS = {
    "LOI": {"dir": "LOI", "task": "regression", "primary_metric": "R2", "higher_is_better": True},
    "Tg": {"dir": "Tg", "task": "regression", "primary_metric": "R2", "higher_is_better": True},
    "TENSILE": {"dir": "TENSILE", "task": "regression", "primary_metric": "R2", "higher_is_better": True},
    "UL94": {"dir": "94", "task": "classification", "primary_metric": "AUC", "higher_is_better": True},
}


FAMILY_RULES = [
    ("descriptor_tabicl", re.compile(r"\bTabICL\b", re.I)),
    ("descriptor_tabpfn", re.compile(r"\bTabPFN\b", re.I)),
    ("descriptor_tree_boosting", re.compile(r"XGBoost|LightGBM|CatBoost|GBR", re.I)),
    ("descriptor_tree_bagging", re.compile(r"RandomForest|ExtraTrees", re.I)),
    ("descriptor_linear_kernel_nn", re.compile(r"Ridge|Lasso|ElasticNet|Bayesian|SVR|Kernel|PLS|MLP|ELM", re.I)),
    ("plain_gnn", re.compile(r"PureGNN|PlainGNN|translated_textGNN|GNN$", re.I)),
    ("fusion_sysfilm", re.compile(r"SysFiLM|System.?FiLM|Fusion|translated_text", re.I)),
    ("residual_gnn", re.compile(r"Residual|translated_text", re.I)),
]


def family_for(model: str, source: str) -> str:
    if re.search(r"SystemFiLM|System.?FiLM|Fusion|translated_text", source, re.I):
        return "fusion_sysfilm"
    if re.search(r"PureGNN|PlainGNN|translated_textGNN", source, re.I):
        return "plain_gnn"
    if re.search(r"Residual|translated_text", source, re.I):
        return "residual_gnn"

    text = f"{model} {source}"
    for family, pattern in FAMILY_RULES:
        if pattern.search(text):
            return family
    return "other"


def latest_file(base: Path, pattern: str) -> list[Path]:
    files = list(base.rglob(pattern))
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


def metric_cols(task: str) -> dict[str, str]:
    if task == "classification":
        return {
            "mean_primary": "Mean_Test_AUC",
            "std_primary": "Std_Test_AUC",
            "fold_primary": "Test_AUC",
            "mean_secondary": "Mean_Test_F1",
            "mean_error": "",
        }
    return {
        "mean_primary": "Mean_Test_R2",
        "std_primary": "Std_Test_R2",
        "fold_primary": "Test_R2",
        "mean_secondary": "Mean_Test_RMSE",
        "mean_error": "Mean_Test_RMSE",
    }


def add_summary_rows(rows: list[dict], target: str, source: str, csv_path: Path, df: pd.DataFrame) -> None:
    task = TARGETS[target]["task"]
    cols = metric_cols(task)
    model_col = "Model" if "Model" in df.columns else None
    if model_col is None:
        return

    for _, row in df.iterrows():
        model = str(row[model_col])
        mean_primary = row.get(cols["mean_primary"], np.nan)
        std_primary = row.get(cols["std_primary"], np.nan)
        mean_secondary = row.get(cols["mean_secondary"], np.nan)
        if pd.isna(mean_primary):
            continue
        rows.append(
            {
                "Target": target,
                "Task": task,
                "Model": model,
                "Family": family_for(model, source),
                "Source": source,
                "Result_File": str(csv_path.relative_to(ROOT)),
                "Primary_Metric": TARGETS[target]["primary_metric"],
                "Mean_Primary": float(mean_primary),
                "Std_Primary": float(std_primary) if not pd.isna(std_primary) else np.nan,
                "Mean_Secondary": float(mean_secondary) if not pd.isna(mean_secondary) else np.nan,
                "Higher_Is_Better": True,
            }
        )


def add_pure_gnn_rows(rows: list[dict], target: str, csv_path: Path) -> None:
    task = TARGETS[target]["task"]
    df = pd.read_csv(csv_path)
    if df.empty:
        return
    if task == "classification":
        primary_candidates = ["AUC", "Test_AUC", "ROC_AUC", "Accuracy"]
        secondary_candidates = ["F1", "Test_F1", "Accuracy"]
    else:
        primary_candidates = ["R2", "Test_R2"]
        secondary_candidates = ["RMSE", "Test_RMSE"]
    primary = next((c for c in primary_candidates if c in df.columns), None)
    secondary = next((c for c in secondary_candidates if c in df.columns), None)
    if primary is None:
        return
    rows.append(
        {
            "Target": target,
            "Task": task,
            "Model": "PureGNN",
            "Family": "plain_gnn",
            "Source": "PureGNN_5CV",
            "Result_File": str(csv_path.relative_to(ROOT)),
            "Primary_Metric": TARGETS[target]["primary_metric"],
            "Mean_Primary": float(df[primary].mean()),
            "Std_Primary": float(df[primary].std(ddof=0)),
            "Mean_Secondary": float(df[secondary].mean()) if secondary else np.nan,
            "Higher_Is_Better": True,
        }
    )


def add_tabicl_hpo_rows(rows: list[dict], target: str, csv_path: Path) -> None:
    task = TARGETS[target]["task"]
    df = pd.read_csv(csv_path)
    if "value" not in df.columns or df.empty:
        return
    best = df.sort_values("value", ascending=False).iloc[0]
    rows.append(
        {
            "Target": target,
            "Task": task,
            "Model": "TabICL-HPO-best",
            "Family": "descriptor_tabicl",
            "Source": "TabICL_HPO",
            "Result_File": str(csv_path.relative_to(ROOT)),
            "Primary_Metric": TARGETS[target]["primary_metric"],
            "Mean_Primary": float(best["value"]),
            "Std_Primary": np.nan,
            "Mean_Secondary": np.nan,
            "Higher_Is_Better": True,
        }
    )


def add_final_interpret_rows(rows: list[dict], target: str, metrics_path: Path) -> None:
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    task = TARGETS[target]["task"]
    if task == "classification":
        primary = metrics.get("AUC", metrics.get("auc", metrics.get("ROC_AUC", np.nan)))
        secondary = metrics.get("F1", metrics.get("Accuracy", np.nan))
    else:
        primary = metrics.get("R2", metrics.get("r2", metrics.get("test_r2", np.nan)))
        secondary = metrics.get("RMSE", metrics.get("rmse", metrics.get("test_rmse", np.nan)))
    if pd.isna(primary):
        return
    rows.append(
        {
            "Target": target,
            "Task": task,
            "Model": "TabICL-final-interpret",
            "Family": "descriptor_tabicl",
            "Source": "Final_TabICL_Interpret",
            "Result_File": str(metrics_path.relative_to(ROOT)),
            "Primary_Metric": TARGETS[target]["primary_metric"],
            "Mean_Primary": float(primary),
            "Std_Primary": np.nan,
            "Mean_Secondary": float(secondary) if not pd.isna(secondary) else np.nan,
            "Higher_Is_Better": True,
        }
    )


def collect() -> pd.DataFrame:
    rows: list[dict] = []
    for target, cfg in TARGETS.items():
        base = ROOT / cfg["dir"]
        if not base.exists():
            continue

        for path in latest_file(base, "CV_Statistical_Summary_NoGNN*.csv"):
            add_summary_rows(rows, target, "NoGNN_descriptor_benchmark", path, pd.read_csv(path))

        for path in latest_file(base, "CV_Statistical_Summary_Strict*.csv"):
            add_summary_rows(rows, target, "SystemFiLM_or_fusion_benchmark", path, pd.read_csv(path))

        for path in latest_file(base, "*GNN*5Fold*Metrics*.csv"):
            add_pure_gnn_rows(rows, target, path)

        for path in latest_file(base, "Optuna_HPO_Results*.csv"):
            add_tabicl_hpo_rows(rows, target, path)

        for path in latest_file(base, "metrics.json"):
            add_final_interpret_rows(rows, target, path)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["Target", "Mean_Primary"], ascending=[True, False]).reset_index(drop=True)
    return df


def build_tables(df: pd.DataFrame) -> None:
    df.to_csv(OUT / "all_existing_model_performance.csv", index=False, encoding="utf-8-sig")

    best_by_family = (
        df.sort_values("Mean_Primary", ascending=False)
        .groupby(["Target", "Family"], as_index=False)
        .first()
        .sort_values(["Target", "Mean_Primary"], ascending=[True, False])
    )
    best_by_family.to_csv(OUT / "best_by_family.csv", index=False, encoding="utf-8-sig")

    rows = []
    for target, g in best_by_family.groupby("Target"):
        tabicl = g[g["Family"].eq("descriptor_tabicl")]
        if tabicl.empty:
            continue
        tabicl_best = tabicl.iloc[0]
        for _, row in g.iterrows():
            rows.append(
                {
                    "Target": target,
                    "Reference_Model": tabicl_best["Model"],
                    "Compared_Family": row["Family"],
                    "Compared_Model": row["Model"],
                    "TabICL_Primary": tabicl_best["Mean_Primary"],
                    "Compared_Primary": row["Mean_Primary"],
                    "Absolute_Gain_TabICL_minus_Compared": tabicl_best["Mean_Primary"] - row["Mean_Primary"],
                    "TabICL_Wins": tabicl_best["Mean_Primary"] >= row["Mean_Primary"],
                    "Interpretation": (
                        "TabICL is better or tied"
                        if tabicl_best["Mean_Primary"] >= row["Mean_Primary"]
                        else "This family has a stronger existing run; inspect complexity and leakage controls"
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(OUT / "tabicl_superiority_table.csv", index=False, encoding="utf-8-sig")

    manifest = pd.DataFrame(
        [
            ["descriptor_tabicl", "Pure descriptors + processing parameters + TabICL"],
            ["descriptor_tabpfn", "Pure descriptors + processing parameters + TabPFN"],
            ["descriptor_tree_boosting", "XGBoost / LightGBM / CatBoost / GBR descriptor baselines"],
            ["descriptor_tree_bagging", "RandomForest / ExtraTrees descriptor baselines"],
            ["descriptor_linear_kernel_nn", "Linear, kernel, PLS, SVR, MLP and ELM descriptor baselines"],
            ["plain_gnn", "Molecular graph encoders without descriptor fusion"],
            ["fusion_sysfilm", "GNN-system embedding or descriptor-GNN fusion methods"],
            ["residual_gnn", "Residual learning after descriptor model"],
            ["other", "Unclassified or auxiliary model"],
        ],
        columns=["Family", "Meaning"],
    )
    manifest.to_csv(OUT / "model_family_manifest.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    df = collect()
    if df.empty:
        print("No existing result files were found.")
        return
    build_tables(df)
    print(f"Collected {len(df)} rows.")
    print(f"Outputs written to: {OUT}")
    print(df.groupby(["Target", "Family"])["Mean_Primary"].max().round(4).to_string())


if __name__ == "__main__":
    main()
