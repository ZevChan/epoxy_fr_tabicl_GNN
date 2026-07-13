from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

TARGET_ORDER = ["LOI", "Tg", "TENSILE", "UL94"]
CONV_ORDER = ["GCN", "SAGE", "GAT", "GIN", "GINE"]
FUSION_ORDER = ["concat", "weighted_sum", "gated", "attention", "film"]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def metric_col(df: pd.DataFrame) -> str | None:
    for col in ["R2_mean", "AUC_mean", "Primary_Mean"]:
        if col in df.columns:
            return col
    return None


def split_gnn_name(model: str) -> tuple[str | None, str | None]:
    text = str(model)
    if text.startswith("Graph_") and "_Fusion_" in text:
        left, fusion = text.split("_Fusion_", 1)
        return left.replace("Graph_", "").upper(), fusion
    parts = text.split("_", 1)
    if len(parts) != 2:
        return None, None
    conv = parts[0].upper()
    fusion = parts[1]
    return conv, fusion


def plot_gnn_fusion_matrix() -> None:
    files = sorted(OUT.glob("gnn_fusion_*_summary.csv"))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5), dpi=300)
    axes = axes.ravel()

    for ax, target in zip(axes, TARGET_ORDER):
        path = OUT / f"gnn_fusion_{target}_summary.csv"
        matrix = pd.DataFrame(np.nan, index=CONV_ORDER, columns=FUSION_ORDER)
        metric_name = "R2/AUC"
        if path.exists():
            df = pd.read_csv(path)
            col = metric_col(df)
            if col is not None:
                metric_name = "AUC" if col.startswith("AUC") else "R2"
                for _, row in df.iterrows():
                    conv, fusion = split_gnn_name(row["Model"])
                    if conv in matrix.index and fusion in matrix.columns:
                        matrix.loc[conv, fusion] = row[col]

        im = ax.imshow(matrix.values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        ax.set_title(f"{target}: GNN encoder x fusion ({metric_name})", fontweight="bold")
        ax.set_xticks(np.arange(len(FUSION_ORDER)))
        ax.set_xticklabels(FUSION_ORDER, rotation=30, ha="right")
        ax.set_yticks(np.arange(len(CONV_ORDER)))
        ax.set_yticklabels(CONV_ORDER)

        for i, conv in enumerate(CONV_ORDER):
            for j, fusion in enumerate(FUSION_ORDER):
                value = matrix.loc[conv, fusion]
                if np.isfinite(value):
                    ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=9, fontweight="bold")
                else:
                    ax.text(j, i, "not run", ha="center", va="center", fontsize=6.5, color="#666666")

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)

    fig.suptitle("Detailed GNN/Fusion Method Matrix", fontsize=15, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.10, top=0.90, wspace=0.35, hspace=0.42)
    cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.03)
    cbar.set_label("R2 or AUC")
    fig.savefig(OUT / "fig5_gnn_fusion_method_matrix.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / "fig5_gnn_fusion_method_matrix.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_existing_fusion_individual_models() -> None:
    path = OUT / "all_existing_model_performance.csv"
    if not path.exists():
        raise FileNotFoundError("Run collect_existing_performance.py first.")
    df = pd.read_csv(path)
    df = df[df["Source"].eq("SystemFiLM_or_fusion_benchmark")].copy()
    if df.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=300)
    axes = axes.ravel()
    for ax, target in zip(axes, TARGET_ORDER):
        sub = df[df["Target"].eq(target)].sort_values("Mean_Primary", ascending=True).tail(12)
        metric = sub["Primary_Metric"].iloc[0] if not sub.empty else "Score"
        colors = ["#009E73" if "TabICL" in str(m) else "#8491B4" for m in sub["Model"]]
        bars = ax.barh(sub["Model"], sub["Mean_Primary"], color=colors, edgecolor="black", linewidth=0.7)
        ax.set_xlim(0, 1.0)
        ax.set_title(f"{target}: individual models on fusion/system features", fontweight="bold")
        ax.set_xlabel(metric)
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        for bar, val in zip(bars, sub["Mean_Primary"]):
            ax.text(min(val + 0.012, 0.98), bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=8)

    fig.suptitle("Fusion/System-Feature Experiments: Individual Learners", fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_fusion_individual_models.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / "fig6_fusion_individual_models.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_existing_gnn_vs_new_gnn() -> None:
    existing_path = OUT / "best_by_family.csv"
    if not existing_path.exists():
        raise FileNotFoundError("Run collect_existing_performance.py first.")
    existing = pd.read_csv(existing_path)
    plain = existing[existing["Family"].eq("plain_gnn")][["Target", "Model", "Mean_Primary", "Primary_Metric"]].copy()
    plain["Detailed_Method"] = "Existing PureGNN"

    rows = []
    for target in TARGET_ORDER:
        path = OUT / f"gnn_fusion_{target}_summary.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        col = metric_col(df)
        if col is None:
            continue
        for _, row in df.iterrows():
            rows.append(
                {
                    "Target": target,
                    "Model": row["Model"],
                    "Mean_Primary": row[col],
                    "Primary_Metric": "AUC" if col.startswith("AUC") else "R2",
                    "Detailed_Method": row["Model"],
                }
            )
    detailed = pd.DataFrame(rows)
    combined = pd.concat([plain, detailed], ignore_index=True)
    if combined.empty:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=300)
    axes = axes.ravel()
    for ax, target in zip(axes, TARGET_ORDER):
        sub = combined[combined["Target"].eq(target)].sort_values("Mean_Primary", ascending=True)
        if sub.empty:
            ax.text(0.5, 0.5, "No GNN detail result yet", ha="center", va="center")
            ax.set_axis_off()
            continue
        colors = ["#CC79A7" if m == "Existing PureGNN" else "#332288" for m in sub["Detailed_Method"]]
        bars = ax.barh(sub["Detailed_Method"], sub["Mean_Primary"], color=colors, edgecolor="black", linewidth=0.7)
        metric = sub["Primary_Metric"].iloc[0]
        ax.set_xlim(0, 1.0)
        ax.set_title(f"{target}: GNN method detail", fontweight="bold")
        ax.set_xlabel(metric)
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        for bar, val in zip(bars, sub["Mean_Primary"]):
            ax.text(min(val + 0.012, 0.98), bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=8)

    fig.suptitle("GNN Method Detail: Existing PureGNN and Newly Run Fusion-GNN", fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(OUT / "fig7_gnn_individual_methods.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / "fig7_gnn_individual_methods.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_style()
    plot_gnn_fusion_matrix()
    plot_existing_fusion_individual_models()
    plot_existing_gnn_vs_new_gnn()
    print(f"Detailed figures written to: {OUT}")


if __name__ == "__main__":
    main()
