from __future__ import annotations

from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
TARGETS = ["LOI", "Tg", "TENSILE", "UL94"]


def load_descriptor_summary(target: str) -> pd.DataFrame:
    path = OUT / f"unified_{target}_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.rename(columns={"Primary_Mean": "Mean_Primary", "Primary_Std": "Std_Primary"})
    df["Input_Block"] = "DescriptorProcess"
    df["Benchmark_Source"] = "from_scratch_descriptor_process"
    df["Primary_Metric"] = "AUC" if target == "UL94" else "R2"
    keep = ["Target", "Model", "Family", "Input_Block", "Benchmark_Source", "Primary_Metric", "Mean_Primary", "Std_Primary"]
    return df[keep]


def load_gnn_summary(target: str) -> pd.DataFrame:
    path = OUT / f"gnn_fusion_{target}_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    metric = "AUC" if target == "UL94" else "R2"
    df = df[df["Model"].astype(str).str.startswith("Graph_")].copy()
    df["Mean_Primary"] = df[f"{metric}_mean"]
    df["Std_Primary"] = df[f"{metric}_std"]
    df["Input_Block"] = "GraphAndProcess"
    df["Benchmark_Source"] = "from_scratch_gnn_fusion"
    df["Primary_Metric"] = metric
    keep = ["Target", "Model", "Family", "Input_Block", "Benchmark_Source", "Primary_Metric", "Mean_Primary", "Std_Primary"]
    return df[keep]


def method_group(model: str) -> str:
    if model.startswith("DescriptorProcess_TabICL"):
        return "DescriptorProcess_TabICL"
    if model.startswith("DescriptorProcess_TabPFN"):
        return "DescriptorProcess_TabPFN"
    if any(x in model for x in ["XGBoost", "LightGBM", "CatBoost", "GradientBoosting"]):
        return "DescriptorProcess_Boosting"
    if any(x in model for x in ["RandomForest", "ExtraTrees"]):
        return "DescriptorProcess_Bagging"
    if model.startswith("DescriptorProcess_"):
        return "DescriptorProcess_OtherML"
    if model.startswith("Graph_") and model.endswith("_Fusion_graph_only"):
        return "Plain_GNN"
    if model.startswith("Graph_"):
        return "GraphDescriptor_Fusion"
    return "Other"


def main() -> None:
    parts = []
    for target in TARGETS:
        parts.append(load_descriptor_summary(target))
        parts.append(load_gnn_summary(target))
    df = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    df["Method_Group"] = df["Model"].map(method_group)
    df = df.sort_values(["Target", "Mean_Primary"], ascending=[True, False]).reset_index(drop=True)
    df.to_csv(OUT / "from_scratch_all_model_summary.csv", index=False, encoding="utf-8-sig")

    best_by_group = (
        df.sort_values("Mean_Primary", ascending=False)
        .groupby(["Target", "Method_Group"], as_index=False)
        .first()
        .sort_values(["Target", "Mean_Primary"], ascending=[True, False])
    )
    best_by_group.to_csv(OUT / "from_scratch_best_by_method_group.csv", index=False, encoding="utf-8-sig")

    rows = []
    for target, sub in best_by_group.groupby("Target"):
        tabicl = sub[sub["Method_Group"].eq("DescriptorProcess_TabICL")]
        gnn = sub[sub["Method_Group"].eq("GraphMolecular_FusionProcess")]
        if tabicl.empty or gnn.empty:
            continue
        t = tabicl.iloc[0]
        g = gnn.iloc[0]
        rows.append(
            {
                "Target": target,
                "TabICL_Model": t["Model"],
                "TabICL_Mean": t["Mean_Primary"],
                "Best_GNN_Fusion_Model": g["Model"],
                "Best_GNN_Fusion_Mean": g["Mean_Primary"],
                "TabICL_minus_Best_GNN_Fusion": t["Mean_Primary"] - g["Mean_Primary"],
                "TabICL_Wins": t["Mean_Primary"] >= g["Mean_Primary"],
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "from_scratch_tabicl_vs_best_gnn_fusion.csv", index=False, encoding="utf-8-sig")
    print(df.groupby(["Target", "Method_Group"])["Mean_Primary"].max().round(4).to_string())
    print(f"From-scratch summary written to {OUT}")


if __name__ == "__main__":
    main()
