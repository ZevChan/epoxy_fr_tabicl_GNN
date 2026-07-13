from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
TARGETS = ["LOI", "Tg", "TENSILE", "UL94"]

GROUP_ORDER = [
    "DescriptorProcess_TabICL",
    "DescriptorProcess_TabPFN",
    "DescriptorProcess_Boosting",
    "DescriptorProcess_Bagging",
    "DescriptorProcess_OtherML",
    "GraphMolecular_FusionProcess",
]
GROUP_LABEL = {
    "DescriptorProcess_TabICL": "Descriptor/process TabICL",
    "DescriptorProcess_TabPFN": "Descriptor/process TabPFN",
    "DescriptorProcess_Boosting": "Descriptor/process boosting",
    "DescriptorProcess_Bagging": "Descriptor/process bagging",
    "DescriptorProcess_OtherML": "Descriptor/process other ML",
    "GraphMolecular_FusionProcess": "Graph + process fusion",
}
COLORS = {
    "DescriptorProcess_TabICL": "#D55E00",
    "DescriptorProcess_TabPFN": "#E69F00",
    "DescriptorProcess_Boosting": "#0072B2",
    "DescriptorProcess_Bagging": "#56B4E9",
    "DescriptorProcess_OtherML": "#999999",
    "GraphMolecular_FusionProcess": "#009E73",
}
CONVS = ["GCN", "SAGE", "GAT", "GIN", "GINE"]
FUSIONS = ["concat", "weighted_sum", "gated", "attention", "film"]


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
        }
    )


def split_graph_name(model: str):
    if model.startswith("Graph_") and "_Fusion_" in model:
        left, fusion = model.split("_Fusion_", 1)
        return left.replace("Graph_", ""), fusion
    return None, None


def plot_best_group_bars(df: pd.DataFrame) -> None:
    best = (
        df.sort_values("Mean_Primary", ascending=False)
        .groupby(["Target", "Method_Group"], as_index=False)
        .first()
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=300)
    axes = axes.ravel()
    for ax, target in zip(axes, TARGETS):
        sub = best[best["Target"].eq(target)].copy()
        sub["Method_Group"] = pd.Categorical(sub["Method_Group"], GROUP_ORDER, ordered=True)
        sub = sub.sort_values("Mean_Primary", ascending=True)
        labels = [GROUP_LABEL.get(x, x) for x in sub["Method_Group"].astype(str)]
        colors = [COLORS.get(x, "#BBBBBB") for x in sub["Method_Group"].astype(str)]
        bars = ax.barh(labels, sub["Mean_Primary"], color=colors, edgecolor="black", linewidth=0.7)
        metric = sub["Primary_Metric"].iloc[0]
        ax.set_title(f"{target} ({metric})", fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_xlabel(metric)
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        for bar, value in zip(bars, sub["Mean_Primary"]):
            ax.text(min(value + 0.012, 0.98), bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=8)
    fig.suptitle("From-Scratch Benchmark: Best Model in Each Method Group", fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(OUT / "from_scratch_fig1_best_method_group.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / "from_scratch_fig1_best_method_group.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_individual_model_rankings(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
    axes = axes.ravel()
    for ax, target in zip(axes, TARGETS):
        sub = df[df["Target"].eq(target)].sort_values("Mean_Primary", ascending=True).tail(16)
        colors = [COLORS.get(x, "#BBBBBB") for x in sub["Method_Group"]]
        bars = ax.barh(sub["Model"], sub["Mean_Primary"], color=colors, edgecolor="black", linewidth=0.7)
        metric = sub["Primary_Metric"].iloc[0]
        ax.set_xlim(0, 1)
        ax.set_title(f"{target}: top individual models", fontweight="bold")
        ax.set_xlabel(metric)
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        for bar, value in zip(bars, sub["Mean_Primary"]):
            ax.text(min(value + 0.012, 0.98), bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=7.5)
    fig.suptitle("From-Scratch Benchmark: Self-Describing Model Names", fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(OUT / "from_scratch_fig2_individual_model_ranking.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / "from_scratch_fig2_individual_model_ranking.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_gnn_matrix(df: pd.DataFrame) -> None:
    gnn = df[df["Model"].str.startswith("Graph_", na=False)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5), dpi=300)
    axes = axes.ravel()
    for ax, target in zip(axes, TARGETS):
        sub = gnn[gnn["Target"].eq(target)]
        matrix = pd.DataFrame(np.nan, index=CONVS, columns=FUSIONS)
        for _, row in sub.iterrows():
            conv, fusion = split_graph_name(row["Model"])
            if conv in matrix.index and fusion in matrix.columns:
                matrix.loc[conv, fusion] = row["Mean_Primary"]
        im = ax.imshow(matrix.values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
        metric = sub["Primary_Metric"].iloc[0] if not sub.empty else "R2/AUC"
        ax.set_title(f"{target}: graph encoder x fusion ({metric})", fontweight="bold")
        ax.set_xticks(np.arange(len(FUSIONS)))
        ax.set_xticklabels(FUSIONS, rotation=30, ha="right")
        ax.set_yticks(np.arange(len(CONVS)))
        ax.set_yticklabels(CONVS)
        for i, conv in enumerate(CONVS):
            for j, fusion in enumerate(FUSIONS):
                value = matrix.loc[conv, fusion]
                if np.isfinite(value):
                    ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8.5, fontweight="bold")
    fig.suptitle("From-Scratch Graph/Fusion Grid", fontsize=15, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.10, top=0.90, wspace=0.35, hspace=0.42)
    cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.03)
    cbar.set_label("R2 or AUC")
    fig.savefig(OUT / "from_scratch_fig3_graph_fusion_matrix.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / "from_scratch_fig3_graph_fusion_matrix.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_tabicl_vs_gnn() -> None:
    path = OUT / "from_scratch_tabicl_vs_best_gnn_fusion.csv"
    df = pd.read_csv(path)
    x = np.arange(len(df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
    ax.bar(x - width / 2, df["TabICL_Mean"], width, label="DescriptorProcess_TabICL", color="#D55E00", edgecolor="black", linewidth=0.7)
    ax.bar(x + width / 2, df["Best_GNN_Fusion_Mean"], width, label="Best Graph_Fusion", color="#009E73", edgecolor="black", linewidth=0.7)
    for i, row in df.iterrows():
        ax.text(i, max(row["TabICL_Mean"], row["Best_GNN_Fusion_Mean"]) + 0.025, f"+{row['TabICL_minus_Best_GNN_Fusion']:.3f}", ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Target"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("R2 or AUC")
    ax.set_title("TabICL vs. Best Graph/Fusion Model", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "from_scratch_fig4_tabicl_vs_best_graph_fusion.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / "from_scratch_fig4_tabicl_vs_best_graph_fusion.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure()
    path = OUT / "from_scratch_all_model_summary.csv"
    if not path.exists():
        raise FileNotFoundError("Run collect_from_scratch_performance.py first.")
    df = pd.read_csv(path)
    plot_best_group_bars(df)
    plot_individual_model_rankings(df)
    plot_gnn_matrix(df)
    plot_tabicl_vs_gnn()
    print(f"From-scratch figures written to {OUT}")


if __name__ == "__main__":
    main()

