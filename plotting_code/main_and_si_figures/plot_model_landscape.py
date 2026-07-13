"""Model landscape figures from all_model_performance_summary.csv."""
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_submission_figures_v2 import pick_five_models, load_benchmark_summary, add_full_box as _afb, bold_axes as _ba

DEST = HERE.parent / "translated_text2"
CSV = DEST / "all_model_performance_summary.csv"
df_all = pd.read_csv(CSV)

# --------------- style ---------------
INK = "#333333"
GRID = "#E6E6E6"
EXPLICIT = "#1F4E79"
GRAPH = "#C0504D"
SLATE = "#808080"
SILVER = "#D9D9D9"
RESIN = "#4A7B9D"
FR = "#E08E36"

GROUP_ORDER = [
    "DescriptorProcess_OtherML",
    "DescriptorProcess_Bagging",
    "DescriptorProcess_Boosting",
    "DescriptorProcess_TabPFN",
    "DescriptorProcess_TabICL",
    "Plain_GNN",
    "GraphDescriptor_Fusion",
]
GROUP_LABELS = {
    "DescriptorProcess_OtherML": "Traditional ML",
    "DescriptorProcess_Bagging": "Bagging",
    "DescriptorProcess_Boosting": "Boosting",
    "DescriptorProcess_TabPFN": "TabPFN",
    "DescriptorProcess_TabICL": "TabICL",
    "Plain_GNN": "Plain GNN",
    "GraphDescriptor_Fusion": "Graph Fusion",
}
XGB_LIGHT = "#29B6F6"
GROUP_COLORS = {
    "Traditional ML": SLATE,
    "Bagging": RESIN,
    "Boosting": XGB_LIGHT,
    "TabPFN": RESIN,
    "TabICL": EXPLICIT,
    "Plain GNN": FR,
    "Graph Fusion": GRAPH,
}

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 1.05, "axes.labelsize": 11, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "legend.fontsize": 9.5,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
})


def add_full_box(ax, lw=1.05):
    for s in ["top", "right", "bottom", "left"]:
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(lw)
        ax.spines[s].set_color(INK)


def bold_axes(ax, tick=9.5, label=11):
    ax.tick_params(width=1.05, labelsize=tick, colors=INK)
    for it in ax.get_xticklabels() + ax.get_yticklabels():
        it.set_fontweight("bold")
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    ax.xaxis.label.set_fontsize(label)
    ax.yaxis.label.set_fontsize(label)


def prepare_df():
    df = df_all.copy()
    df["Group_Label"] = df["Method_Group"].map(GROUP_LABELS).fillna("Other")
    # map Target to display
    tmap = {"LOI": "LOI", "Tg": "Tg", "TENSILE": "Tensile", "UL94": "UL-94"}
    df["TargetDisplay"] = df["Target"].map(tmap)
    # Normalize Mean_Primary to 0-1 for cross-task comparison (R2 and AUC both 0-1 ideally)
    # Some R2 can be negative — keep raw for scatter, clip violin to reasonable range
    return df


# ============================================================
# Figure 1: Violin + Stripplot — monochromatic gradient by performance rank
# ============================================================
def fig1_violin():
    df = prepare_df()
    order = ["Plain GNN", "Traditional ML", "Graph Fusion", "Bagging", "Boosting", "TabPFN", "TabICL"]
    # Gradient: light→deep blue along the fixed order
    n = len(order)
    tcmap = mcolors.LinearSegmentedColormap.from_list("violin_grad", ["#D6E4F0", EXPLICIT])
    palette = {lab: tcmap(order.index(lab) / max(n - 1, 1)) for lab in order}

    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=300)
    plot_df = df.copy()
    plot_df["Mean_Primary"] = plot_df["Mean_Primary"].clip(-0.5, 1.0)

    sns.violinplot(
        data=plot_df, x="Group_Label", y="Mean_Primary", order=order,
        palette=palette, inner=None, linewidth=0.8, cut=0, ax=ax,
    )
    sns.stripplot(
        data=plot_df, x="Group_Label", y="Mean_Primary", order=order,
        color=INK, size=4.5, alpha=0.55, jitter=0.22, ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Performance (R² / AUC)")
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", color=GRID, linestyle="--", linewidth=0.65)
    add_full_box(ax)
    bold_axes(ax, tick=9.5, label=11.5)
    fig.tight_layout()
    fig.savefig(DEST / "Figure_model_violin_landscape.png", dpi=600, bbox_inches="tight")
    fig.savefig(DEST / "Figure_model_violin_landscape.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure_model_violin_landscape")


# ============================================================
# Figure 2: Mean-Variance Pareto scatter (1×4) — improved
# ============================================================
def categorize_family(family_str):
    """translated_text，translated_text"""
    family_str = str(family_str).lower()
    if "tabicl" in family_str or "tabpfn" in family_str:
        return "Advanced Tabular"
    elif "boosting" in family_str or "bagging" in family_str or "forest" in family_str or "tree" in family_str:
        return "Tree Ensemble"
    elif "graph" in family_str or "gnn" in family_str or "gcn" in family_str or "gat" in family_str or "gine" in family_str:
        return "Graph / Fusion"
    else:
        return "Traditional ML"


def fig2_pareto():
    df = df_all.copy()
    df["Broad_Family"] = df["Family"].apply(categorize_family)

    XGB_LIGHT = "#29B6F6"
    palette = {
        "Advanced Tabular": EXPLICIT,
        "Tree Ensemble": XGB_LIGHT,
        "Graph / Fusion": GRAPH,
        "Traditional ML": SLATE,
    }

    targets = [("LOI", "LOI"), ("Tg", "Tg"), ("TENSILE", "Tensile"), ("UL94", "UL-94")]

    fig, axes = plt.subplots(1, 4, figsize=(18.0, 4.8), dpi=300)
    for i, ((target_key, display), ax) in enumerate(zip(targets, axes)):
        sub_df = df[df["Target"].eq(target_key)].copy()
        sub_df = sub_df.dropna(subset=["Mean_Primary", "Std_Primary"])

        sns.scatterplot(
            data=sub_df,
            x="Std_Primary",
            y="Mean_Primary",
            hue="Broad_Family",
            palette=palette,
            s=45,
            alpha=0.75,
            edgecolor=INK,
            linewidth=0.5,
            ax=ax,
            legend=(i == 0),
        )

        ax.invert_xaxis()
        ax.set_title(display, loc="center", fontsize=13, fontweight="bold", color=INK, pad=8)
        ax.set_ylim(0, None)

        if i == 0:
            ax.set_ylabel("Mean Performance (R² / AUC)", fontsize=11, fontweight="bold")
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles, labels=labels, frameon=True, facecolor="white",
                      edgecolor=GRID, fontsize=8, loc="lower left")
        else:
            ax.set_ylabel("")
        ax.set_xlabel("Std (CV)", fontsize=11, fontweight="bold")

        # Annotate best TabICL model
        tabicl_models = sub_df[sub_df["Model"].str.contains("TabICL", case=False, na=False)]
        if not tabicl_models.empty:
            best_tab = tabicl_models.sort_values("Mean_Primary", ascending=False).iloc[0]
            ax.annotate("TabICL",
                        xy=(best_tab["Std_Primary"], best_tab["Mean_Primary"]),
                        xytext=(best_tab["Std_Primary"] - 0.01, best_tab["Mean_Primary"] - 0.06),
                        arrowprops=dict(facecolor=INK, arrowstyle="->", lw=1.2),
                        fontsize=9, fontweight="bold", color=EXPLICIT, ha="right")

        ax.grid(color=GRID, linestyle="--", linewidth=0.65)
        add_full_box(ax)
        bold_axes(ax, tick=9, label=11)

    fig.tight_layout()
    fig.savefig(DEST / "Figure_model_pareto_scatter.png", dpi=600, bbox_inches="tight")
    fig.savefig(DEST / "Figure_model_pareto_scatter.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure_model_pareto_scatter")


# ============================================================
# Figure 3: Top-model heatmap (SI)
# ============================================================
def fig3_heatmap():
    df = prepare_df()

    # Build a multi-metric pivot: pick top 35 models by average Mean_Primary across tasks
    metrics = []
    for tkey, tdisp in [("LOI", "LOI"), ("Tg", "Tg"), ("TENSILE", "Tensile"), ("UL94", "UL94")]:
        sub = df[df["Target"] == tkey].copy()
        sub = sub[["Model", "Mean_Primary"]].rename(columns={"Mean_Primary": f"{tdisp}_Primary"})
        metrics.append(sub)

        # Add RMSE for regression, F1 for classification
        if tkey == "UL94":
            if "F1_mean" in df.columns:
                f1 = df[df["Target"] == tkey][["Model", "F1_mean"]].rename(columns={"F1_mean": f"{tdisp}_F1"})
                metrics.append(f1)
            if "AUC_mean" in df.columns:
                auc = df[df["Target"] == tkey][["Model", "AUC_mean"]].rename(columns={"AUC_mean": f"{tdisp}_AUC"})
                metrics.append(auc)
        else:
            if "RMSE_mean" in df.columns:
                rmse = df[df["Target"] == tkey][["Model", "RMSE_mean"]].rename(columns={"RMSE_mean": f"{tdisp}_RMSE"})
                metrics.append(rmse)

    # Merge all
    pivot = metrics[0]
    for m in metrics[1:]:
        pivot = pivot.merge(m, on="Model", how="outer")
    pivot = pivot.set_index("Model")

    # Rank by average primary (higher better)
    primary_cols = [c for c in pivot.columns if "_Primary" in c]
    pivot["avg_primary"] = pivot[primary_cols].mean(axis=1)
    pivot = pivot.sort_values("avg_primary", ascending=False).head(35)
    pivot = pivot.drop(columns=["avg_primary"])

    # Normalize each column to 0-1 for heatmap (RMSE reversed)
    normed = pivot.copy()
    for col in normed.columns:
        if "_RMSE" in col:
            normed[col] = 1 - (normed[col] - normed[col].min()) / (normed[col].max() - normed[col].min() + 1e-9)
        else:
            normed[col] = (normed[col] - normed[col].min()) / (normed[col].max() - normed[col].min() + 1e-9)

    # Shorten model names
    def short(s):
        replacements = {
            "DescriptorProcess_": "", "Regressor": "", "Classifier": "",
            "Graph_": "", "_Fusion_": "-",
        }
        for old, new in replacements.items():
            s = s.replace(old, new)
        return s[:40]

    normed.index = [short(m) for m in normed.index]

    fig, ax = plt.subplots(figsize=(16, 12), dpi=300)
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    sns.heatmap(
        normed, annot=pivot.round(3).values, fmt="", cmap="YlOrRd",
        linewidths=0.6, linecolor="white", ax=ax,
        annot_kws={"fontsize": 6.5, "fontweight": "bold"},
        cbar_kws={"label": "Normalized score", "shrink": 0.6},
    )
    ax.set_title("Top 35 models — multi-metric heatmap (SI)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7.5)
    fig.tight_layout()
    fig.savefig(DEST / "Figure_SI_model_heatmap.png", dpi=600, bbox_inches="tight")
    fig.savefig(DEST / "Figure_SI_model_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure_SI_model_heatmap")


# ============================================================
# Combined Figure: Benchmark bars (top) + Violin (bottom)
# ============================================================
def fig_combined_benchmark_violin():
    df_bench = load_benchmark_summary()
    violin_df = prepare_df()

    fig = plt.figure(figsize=(18.0, 9.5), dpi=300)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.15], hspace=0.35)

    # --- Row 1: 1×4 panoramic bars ---
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    sub_gs = GridSpecFromSubplotSpec(1, 4, gs[0, 0], wspace=0.28)
    axes_bar = [fig.add_subplot(sub_gs[0, i]) for i in range(4)]

    targets = [("LOI", "LOI"), ("UL94", "UL-94"), ("Tg", "Tg"), ("TENSILE", "Tensile")]
    target_base_color = {"LOI": EXPLICIT, "Tg": RESIN, "TENSILE": FR, "UL94": GRAPH}
    for i, ((target_key, display), ax) in enumerate(zip(targets, axes_bar)):
        if i == 0:
            ax.text(-0.15, 1.08, "a", transform=ax.transAxes, fontsize=20, fontweight="bold", va="bottom", color=INK)
        models = pick_five_models(df_bench, target_key)
        xs = np.arange(len(models))
        means = [m["mean"] for m in models]
        stds = [m["std"] for m in models]
        labels = [m["label"] for m in models]
        base = target_base_color.get(target_key, EXPLICIT)
        tcmap = mcolors.LinearSegmentedColormap.from_list(f"cb_{target_key}", ["#FFFFFF", base])
        colors = [tcmap(0.2 + 0.8 * j / max(len(models) - 1, 1)) for j in range(len(models))]
        ax.bar(xs, means, yerr=stds, capsize=4.5, color=colors, edgecolor="none", width=0.6,
               error_kw={"linewidth": 1.2, "ecolor": INK})
        for x_pos, mv, sv in zip(xs, means, stds):
            ly = mv + sv + 0.02
            if mv < 0:
                ly = mv - sv - 0.06
            ax.text(x_pos, ly, f"{mv:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=12, fontweight="bold", rotation=25, ha="right", rotation_mode="anchor")
        y_max = max([m + s for m, s in zip(means, stds)]) if means else 1.0
        y_min = min(0, min([m - s for m, s in zip(means, stds)])) if means else 0.0
        ax.set_ylim(y_min * 1.15 if y_min < 0 else 0, max(1.02, y_max * 1.15))
        metric = "AUC" if target_key == "UL94" else "$R^2$"
        ax.set_ylabel(f"{display} ({metric})", fontsize=14.5, fontweight="bold")
        ax.axhline(0, color=INK, linewidth=1.0, linestyle="-")
        ax.grid(axis="y", color=GRID, linestyle="--", linewidth=0.7)
        _afb(ax)
        _ba(ax, tick=12, label=14.5)

    # --- Row 2: Violin ---
    ax_v = fig.add_subplot(gs[1, 0])
    ax_v.text(-0.06, 1.04, "b", transform=ax_v.transAxes, fontsize=20, fontweight="bold", va="bottom", color=INK)
    order_v = ["Plain GNN", "Traditional ML", "Graph Fusion", "Bagging", "Boosting", "TabPFN", "TabICL"]
    n_v = len(order_v)
    tcmap_v = mcolors.LinearSegmentedColormap.from_list("vg", ["#D6E4F0", EXPLICIT])
    palette_v = {lab: tcmap_v(order_v.index(lab) / max(n_v - 1, 1)) for lab in order_v}
    plot_df = violin_df.copy()
    plot_df["Mean_Primary"] = plot_df["Mean_Primary"].clip(-0.5, 1.0)
    sns.violinplot(data=plot_df, x="Group_Label", y="Mean_Primary", hue="Group_Label",
                   order=order_v, hue_order=order_v, palette=palette_v, inner=None,
                   linewidth=0.8, cut=0, ax=ax_v, legend=False)
    sns.stripplot(data=plot_df, x="Group_Label", y="Mean_Primary", order=order_v,
                  color=INK, size=4.5, alpha=0.55, jitter=0.22, ax=ax_v)
    ax_v.set_xlabel("")
    ax_v.set_ylabel("Performance (R² / AUC)", fontsize=14.5, fontweight="bold")
    ax_v.set_ylim(0, 1.08)
    ax_v.grid(axis="y", color=GRID, linestyle="--", linewidth=0.65)
    _afb(ax_v)
    _ba(ax_v, tick=12, label=14.5)
    for item in ax_v.get_xticklabels():
        item.set_fontsize(12.5)
        item.set_fontweight("bold")

    fig.tight_layout()
    fig.savefig(DEST / "Figure3_combined_benchmark_violin.png", dpi=600, bbox_inches="tight")
    fig.savefig(DEST / "Figure3_combined_benchmark_violin.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Saved Figure3_combined_benchmark_violin")


# ============================================================
if __name__ == "__main__":
    print("Generating model landscape figures ...")
    fig1_violin()
    fig2_pareto()
    fig3_heatmap()
    fig_combined_benchmark_violin()
    print("All done.")
