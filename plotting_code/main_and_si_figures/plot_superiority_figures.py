from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

BEST_BY_FAMILY = OUT / "best_by_family.csv"
SUPERIORITY = OUT / "tabicl_superiority_table.csv"

TARGET_ORDER = ["LOI", "Tg", "TENSILE", "UL94"]
FAMILY_LABELS = {
    "descriptor_tabicl": "Descriptors + process + TabICL",
    "descriptor_tabpfn": "Descriptors + process + TabPFN",
    "descriptor_tree_boosting": "Boosting baselines",
    "descriptor_tree_bagging": "Bagging baselines",
    "descriptor_linear_kernel_nn": "Linear/kernel/MLP",
    "fusion_sysfilm": "GNN-descriptor fusion",
    "plain_gnn": "GNN only",
    "residual_gnn": "Residual GNN",
    "other": "Other",
}
FAMILY_ORDER = [
    "descriptor_tabicl",
    "descriptor_tabpfn",
    "descriptor_tree_boosting",
    "descriptor_tree_bagging",
    "descriptor_linear_kernel_nn",
    "fusion_sysfilm",
    "plain_gnn",
    "residual_gnn",
    "other",
]
COLORS = {
    "descriptor_tabicl": "#D55E00",
    "descriptor_tabpfn": "#E69F00",
    "descriptor_tree_boosting": "#0072B2",
    "descriptor_tree_bagging": "#56B4E9",
    "descriptor_linear_kernel_nn": "#999999",
    "fusion_sysfilm": "#009E73",
    "plain_gnn": "#CC79A7",
    "residual_gnn": "#332288",
    "other": "#BBBBBB",
}


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


def load_best() -> pd.DataFrame:
    df = pd.read_csv(BEST_BY_FAMILY)
    df = df[df["Family"].isin(FAMILY_LABELS)].copy()
    df["Target"] = pd.Categorical(df["Target"], TARGET_ORDER, ordered=True)
    df["Family"] = pd.Categorical(df["Family"], FAMILY_ORDER, ordered=True)
    return df.sort_values(["Target", "Family"])


def plot_grouped_bars(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=300)
    axes = axes.ravel()

    for ax, target in zip(axes, TARGET_ORDER):
        sub = df[df["Target"].astype(str) == target].copy()
        sub = sub.sort_values("Mean_Primary", ascending=True)
        colors = [COLORS.get(f, "#BBBBBB") for f in sub["Family"].astype(str)]
        labels = [FAMILY_LABELS.get(f, f) for f in sub["Family"].astype(str)]

        bars = ax.barh(labels, sub["Mean_Primary"], color=colors, edgecolor="black", linewidth=0.7)
        metric = sub["Primary_Metric"].iloc[0] if not sub.empty else "Score"
        ax.set_title(f"{target} ({metric})", fontweight="bold")
        ax.set_xlim(0, 1.0)
        ax.set_xlabel(metric)
        ax.grid(axis="x", linestyle="--", alpha=0.25)

        for bar, val in zip(bars, sub["Mean_Primary"]):
            ax.text(min(val + 0.015, 0.98), bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center", fontsize=8)

    fig.suptitle("Best Existing Performance by Model Family", fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_best_by_family_bar.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / "fig1_best_by_family_bar.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_gain_bars() -> None:
    df = pd.read_csv(SUPERIORITY)
    keep = ["descriptor_tree_boosting", "descriptor_tabpfn", "fusion_sysfilm", "plain_gnn"]
    df = df[df["Compared_Family"].isin(keep)].copy()
    df["Compared_Family"] = pd.Categorical(df["Compared_Family"], keep, ordered=True)
    df["Target"] = pd.Categorical(df["Target"], TARGET_ORDER, ordered=True)
    df = df.sort_values(["Target", "Compared_Family"])

    x = np.arange(len(TARGET_ORDER))
    width = 0.19
    fig, ax = plt.subplots(figsize=(10.5, 5), dpi=300)

    for i, fam in enumerate(keep):
        sub = df[df["Compared_Family"].astype(str) == fam]
        gains = [sub[sub["Target"].astype(str) == t]["Absolute_Gain_TabICL_minus_Compared"].iloc[0] if len(sub[sub["Target"].astype(str) == t]) else np.nan for t in TARGET_ORDER]
        ax.bar(x + (i - 1.5) * width, gains, width, label=FAMILY_LABELS[fam], color=COLORS[fam], edgecolor="black", linewidth=0.7)

    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(TARGET_ORDER)
    ax.set_ylabel("Absolute gain of TabICL")
    ax.set_title("How Much TabICL Outperforms Alternative Families", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_tabicl_absolute_gain.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / "fig2_tabicl_absolute_gain.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame) -> None:
    pivot = df.pivot_table(index="Family", columns="Target", values="Mean_Primary", aggfunc="max", observed=False)
    pivot = pivot.reindex(FAMILY_ORDER).dropna(how="all")
    pivot = pivot.reindex(columns=TARGET_ORDER)

    fig, ax = plt.subplots(figsize=(8, 5.8), dpi=300)
    im = ax.imshow(pivot.values, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([FAMILY_LABELS.get(f, f) for f in pivot.index])

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8, color="black")

    ax.set_title("Performance Landscape Across Tasks", fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("R2 or AUC")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_performance_heatmap.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / "fig3_performance_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_complexity_tradeoff(df: pd.DataFrame) -> None:
    complexity = {
        "descriptor_tabicl": 2,
        "descriptor_tabpfn": 2,
        "descriptor_tree_boosting": 2,
        "descriptor_tree_bagging": 2,
        "descriptor_linear_kernel_nn": 1,
        "plain_gnn": 4,
        "fusion_sysfilm": 5,
        "residual_gnn": 5,
        "other": 3,
    }
    display = df[df["Family"].astype(str).isin(complexity)].copy()
    display["Complexity"] = display["Family"].astype(str).map(complexity)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=300)
    axes = axes.ravel()
    for ax, target in zip(axes, TARGET_ORDER):
        sub = display[display["Target"].astype(str) == target]
        for _, row in sub.iterrows():
            fam = str(row["Family"])
            ax.scatter(row["Complexity"], row["Mean_Primary"], s=95, color=COLORS.get(fam, "#BBBBBB"), edgecolor="black", linewidth=0.8)
            if fam in ["descriptor_tabicl", "fusion_sysfilm", "plain_gnn", "descriptor_tree_boosting"]:
                ax.text(row["Complexity"] + 0.04, row["Mean_Primary"], FAMILY_LABELS[fam].split(" + ")[-1].replace(" baselines", ""), fontsize=7, va="center")
        ax.set_xlim(0.5, 5.7)
        ax.set_ylim(0, 1.0)
        ax.set_title(target, fontweight="bold")
        ax.set_xlabel("Implementation complexity")
        ax.set_ylabel("R2 or AUC")
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.grid(True, linestyle="--", alpha=0.22)

    fig.suptitle("Accuracy vs. Complexity: GNN/Fusion Adds Cost Without Winning", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_accuracy_complexity_tradeoff.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / "fig4_accuracy_complexity_tradeoff.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_style()
    if not BEST_BY_FAMILY.exists() or not SUPERIORITY.exists():
        raise FileNotFoundError("Run collect_existing_performance.py first.")
    df = load_best()
    plot_grouped_bars(df)
    plot_gain_bars()
    plot_heatmap(df)
    plot_complexity_tradeoff(df)
    print(f"Figures written to: {OUT}")


if __name__ == "__main__":
    main()

