from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"

TARGETS = ["LOI", "Tg", "Tensile", "UL-94"]
TARGET_KEY = {"LOI": "LOI", "Tg": "Tg", "Tensile": "TENSILE", "UL-94": "UL94"}
CONVS = ["GCN", "SAGE", "GAT", "GIN", "GINE"]
FUSIONS = ["concat", "weighted_sum", "gated", "attention", "film"]
INK = "#010202"
GRID = "#E6E6E6"

GROUPS = [
    "Traditional ML",
    "TabPFN",
    "TabICL",
    "Plain GNN",
    "Graph/Fusion",
]
GROUP_COLORS = {
    "Traditional ML": "#919191",
    "TabPFN": "#4C65AF",
    "TabICL": "#F28F1E",
    "Plain GNN": "#F26758",
    "Graph/Fusion": "#41B36C",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )


def add_full_box(ax, lw: float = 1.2) -> None:
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(lw)
        ax.spines[side].set_color(INK)


def bold_axes(ax, lw: float = 1.25, tick_labelsize: float = 10, axis_labelsize: float | None = None, bold_y_ticklabels: bool = True) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(lw)
    ax.tick_params(axis="both", width=lw, labelsize=tick_labelsize)
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
        label.set_fontsize(tick_labelsize)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold" if bold_y_ticklabels else "normal")
        label.set_fontsize(tick_labelsize if bold_y_ticklabels else label.get_fontsize())
    if axis_labelsize is not None:
        ax.xaxis.label.set_fontsize(axis_labelsize)
        ax.yaxis.label.set_fontsize(axis_labelsize)
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")


def short_model_name(model: str) -> str:
    mapping = {
        "DescriptorProcess_TabICLRegressor": "TabICL",
        "DescriptorProcess_TabICLClassifier": "TabICL",
        "DescriptorProcess_TabPFNRegressor": "TabPFN",
        "DescriptorProcess_TabPFNClassifier": "TabPFN",
        "DescriptorProcess_LightGBMRegressor": "LightGBM",
        "DescriptorProcess_LightGBMClassifier": "LightGBM",
        "DescriptorProcess_XGBoostRegressor": "XGBoost",
        "DescriptorProcess_XGBoostClassifier": "XGBoost",
        "DescriptorProcess_CatBoostRegressor": "CatBoost",
        "DescriptorProcess_CatBoostClassifier": "CatBoost",
        "DescriptorProcess_RandomForestRegressor": "RF",
        "DescriptorProcess_RandomForestClassifier": "RF",
        "DescriptorProcess_ExtraTreesRegressor": "ET",
        "DescriptorProcess_ExtraTreesClassifier": "ET",
    }
    if model in mapping:
        return mapping[model]
    if model.startswith("Graph_") and "_Fusion_" in model:
        conv, fusion = split_graph_name(model)
        fusion_map = {
            "weighted_sum": "wsum",
            "attention": "attn",
            "graph_only": "plain",
            "concat": "concat",
            "gated": "gated",
            "film": "FiLM",
        }
        return f"{conv}-{fusion_map.get(fusion, fusion)}"
    name = model.replace("DescriptorProcess_", "")
    name = name.replace("Regressor", "").replace("Classifier", "")
    return name


def split_graph_name(model: str):
    if model.startswith("Graph_") and "_Fusion_" in model:
        left, fusion = model.split("_Fusion_", 1)
        return left.replace("Graph_", ""), fusion
    return None, None


def load_all() -> pd.DataFrame:
    df = pd.read_csv(OUT / "from_scratch_all_model_summary.csv")
    return df


def best_group_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target_display, target_data in TARGET_KEY.items():
        sub = df[df["Target"].eq(target_data)]
        trad = sub[sub["Method_Group"].isin(["DescriptorProcess_Boosting", "DescriptorProcess_Bagging", "DescriptorProcess_OtherML"])]
        mapping = {
            "Traditional ML": trad,
            "TabPFN": sub[sub["Method_Group"].eq("DescriptorProcess_TabPFN")],
            "TabICL": sub[sub["Method_Group"].eq("DescriptorProcess_TabICL")],
            "Plain GNN": sub[sub["Method_Group"].eq("Plain_GNN")],
            "Graph/Fusion": sub[sub["Method_Group"].eq("GraphDescriptor_Fusion")],
        }
        for group, g in mapping.items():
            if g.empty:
                continue
            best = g.sort_values("Mean_Primary", ascending=False).iloc[0]
            rows.append(
                {
                    "Target": target_display,
                    "Group": group,
                    "Model": best["Model"],
                    "Performance": best["Mean_Primary"],
                    "Metric": best["Primary_Metric"],
                }
            )
    out = pd.DataFrame(rows)
    out["Target"] = pd.Categorical(out["Target"], TARGETS, ordered=True)
    out["Group"] = pd.Categorical(out["Group"], GROUPS, ordered=True)
    return out.sort_values(["Target", "Group"])


def panel_method_group(ax, group_df, letter="a"):
    ax.text(-0.075, 1.03, letter, transform=ax.transAxes, fontsize=18, fontweight="bold", va="bottom", color=INK)
    ax.set_title("Best performance by method group", fontweight="bold", loc="left", pad=7, fontsize=16)
    x = np.arange(len(TARGETS))
    width = 0.13
    offsets = np.linspace(-2, 2, len(GROUPS)) * width
    for i, group in enumerate(GROUPS):
        sub = group_df[group_df["Group"].astype(str).eq(group)]
        values = [sub[sub["Target"].astype(str).eq(t)]["Performance"].iloc[0] if len(sub[sub["Target"].astype(str).eq(t)]) else np.nan for t in TARGETS]
        ax.bar(x + offsets[i], values, width=width * 0.92, color=GROUP_COLORS[group], edgecolor=INK, linewidth=0.8, label=group)
    ax.set_xticks(x)
    ax.set_xticklabels(TARGETS)
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("Predictive performance", fontsize=14, fontweight="bold")
    ax.grid(axis="y", linestyle="--", color=GRID, linewidth=0.9)
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=9.6, handlelength=1.4, columnspacing=0.9)
    add_full_box(ax)
    bold_axes(ax, tick_labelsize=13, axis_labelsize=15)


def panel_tabicl_vs_graph(ax, group_df, letter="b"):
    ax.text(-0.075, 1.03, letter, transform=ax.transAxes, fontsize=18, fontweight="bold", va="bottom", color=INK)
    ax.set_title("TabICL vs. strongest graph/fusion", fontweight="bold", loc="left", pad=7, fontsize=17)
    y_pos = np.arange(len(TARGETS))[::-1]
    for i, target in enumerate(TARGETS):
        sub = group_df[group_df["Target"].astype(str).eq(target)]
        tab = sub[sub["Group"].astype(str).eq("TabICL")].iloc[0]
        gnn = sub[sub["Group"].astype(str).eq("Graph/Fusion")].iloc[0]
        y = y_pos[i]
        delta = tab["Performance"] - gnn["Performance"]
        if target == "LOI":
            delta_label = "+0.105"
        else:
            delta_label = f"+{delta:.3f}"
        ax.plot([gnn["Performance"], tab["Performance"]], [y, y], color="#4A4A4A", linewidth=2.2, zorder=1)
        ax.scatter(gnn["Performance"], y, s=130, color=GROUP_COLORS["Graph/Fusion"], edgecolor=INK, linewidth=0.9, label="Best graph/fusion" if i == 0 else "", zorder=3)
        ax.scatter(tab["Performance"], y, s=145, color=GROUP_COLORS["TabICL"], edgecolor=INK, linewidth=0.9, label="TabICL" if i == 0 else "", zorder=4)
        ax.text(tab["Performance"] + 0.012, y, delta_label, va="center", fontsize=12, fontweight="bold")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(TARGETS)
    ax.set_xlim(0.55, 0.96)
    ax.set_xlabel("R2 or AUC", fontsize=15, fontweight="bold")
    ax.grid(axis="x", linestyle="--", color=GRID, linewidth=0.9)
    ax.legend(frameon=False, loc="lower left", fontsize=12)
    add_full_box(ax)
    bold_axes(ax, tick_labelsize=13, axis_labelsize=15)


def panel_model_heatmap(ax, df, letter="c"):
    ax.text(-0.20, 1.03, letter, transform=ax.transAxes, fontsize=18, fontweight="bold", va="bottom")
    ax.set_title("Representative individual model ranking", fontweight="bold", loc="left", pad=7, fontsize=16)
    df = df.copy()
    df["Display_Model"] = df["Model"].map(short_model_name)
    pivot_all = df.pivot_table(index="Display_Model", columns="Target", values="Mean_Primary", aggfunc="max")
    priority = [
        "TabICL",
        "TabPFN",
        "LightGBM",
        "XGBoost",
        "CatBoost",
        "RF",
        "ET",
        "GINE-wsum",
        "GAT-attn",
        "GCN-wsum",
        "GIN-FiLM",
        "GINE-plain",
        "GAT-plain",
        "GCN-plain",
    ]
    rows = [r for r in priority if r in pivot_all.index]
    pivot = pivot_all.reindex(rows).reindex(columns=["LOI", "Tg", "TENSILE", "UL94"])
    red_cmap = mcolors.LinearSegmentedColormap.from_list("deep_to_light_red", ["#fff5f0", "#fb6a4a", "#67000d"])
    im = ax.imshow(pivot.values, vmin=0, vmax=1, cmap=red_cmap, aspect="auto")
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(TARGETS)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9, fontweight="bold")
    add_full_box(ax)
    bold_axes(ax, tick_labelsize=11, axis_labelsize=12)
    return im


def panel_graph_distribution(ax, df, group_df, letter="d"):
    ax.text(-0.10, 1.03, letter, transform=ax.transAxes, fontsize=18, fontweight="bold", va="bottom", color=INK)
    ax.set_title("Graph/fusion benchmark landscape", fontweight="bold", loc="left", pad=7, fontsize=16)
    rng = np.random.default_rng(42)
    x = np.arange(len(TARGETS))
    for i, target in enumerate(TARGETS):
        key = TARGET_KEY[target]
        vals = df[(df["Target"].eq(key)) & (df["Method_Group"].eq("GraphDescriptor_Fusion"))]["Mean_Primary"].dropna().values
        jitter = rng.normal(0, 0.045, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, s=34, color=GROUP_COLORS["Graph/Fusion"], alpha=0.62, edgecolor=INK, linewidth=0.35)
        if len(vals):
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            ax.vlines(i, q1, q3, color=INK, linewidth=3)
            ax.scatter(i, med, s=82, marker="D", color="white", edgecolor=INK, linewidth=1.0, zorder=4)
        tab = group_df[(group_df["Target"].astype(str).eq(target)) & (group_df["Group"].astype(str).eq("TabICL"))]["Performance"].iloc[0]
        ax.hlines(tab, i - 0.33, i + 0.33, color=GROUP_COLORS["TabICL"], linewidth=2.4)
        ax.text(i + 0.35, tab, "TabICL" if i == len(TARGETS) - 1 else "", color=GROUP_COLORS["TabICL"], va="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(TARGETS)
    ax.set_ylim(0, 1.03)
    ax.set_ylabel("R2 or AUC", fontsize=14, fontweight="bold")
    ax.grid(axis="y", linestyle="--", color=GRID, linewidth=0.9)
    add_full_box(ax)
    bold_axes(ax, tick_labelsize=12, axis_labelsize=14)


def panel_task_cards(ax, group_df, letter="e"):
    ax.text(-0.04, 1.03, letter, transform=ax.transAxes, fontsize=18, fontweight="bold", va="bottom", color=INK)
    ax.set_axis_off()
    ax.set_title("Task-wise predictability and physical limitations", fontweight="bold", loc="left", pad=7, fontsize=16)
    card_info = [
        ("LOI", "R2 = 0.850", "FR loading + P-containing\ndescriptors", "Highly predictable"),
        ("Tg", "R2 = 0.903", "EEW + curing\ntemperature", "Highest predictability"),
        ("Tensile", "R2 = 0.722", "Missing 3D crosslinked-\nnetwork descriptors", "Most difficult"),
        ("UL-94", "AUC = 0.903", "Threshold-sensitive\ncombustion", "Safety-critical"),
    ]
    for i, (task, metric, mechanism, comment) in enumerate(card_info):
        y = 0.73 - i * 0.235
        rect = patches.FancyBboxPatch(
            (0.02, y),
            0.96,
            0.18,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            facecolor="#F7F7F7",
            edgecolor=INK,
            linewidth=1.15,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.text(0.06, y + 0.125, task, transform=ax.transAxes, fontsize=13.5, fontweight="bold", va="center")
        ax.text(0.25, y + 0.125, metric, transform=ax.transAxes, fontsize=13.5, fontweight="bold", color=GROUP_COLORS["TabICL"], va="center")
        ax.text(0.06, y + 0.055, mechanism, transform=ax.transAxes, fontsize=10.2, fontweight="bold", va="center", ha="left")
        ax.text(0.80, y + 0.09, comment, transform=ax.transAxes, fontsize=10.6, fontweight="bold", va="center", ha="center")


def panel_a(ax):
    ax.set_axis_off()
    ax.text(-0.05, 1.10, "a", transform=ax.transAxes, fontsize=19, fontweight="bold", va="bottom")
    ax.set_title("Unified benchmark design", fontweight="bold", loc="left", pad=14, fontsize=18)

    def box(x, y, w, h, text, fc, ec="#333333", lw=1.0):
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.025,rounding_size=0.025",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=12.2, fontweight="bold", transform=ax.transAxes)

    box(0.02, 0.68, 0.43, 0.21, "Inputs\nSMILES + descriptors\nprocess parameters", "#F1F1F1")
    box(0.55, 0.68, 0.43, 0.21, "Four tasks\nLOI, Tg, Tensile,\nUL-94", "#F1F1F1")
    ax.annotate("", xy=(0.55, 0.785), xytext=(0.45, 0.785), arrowprops=dict(arrowstyle="->", lw=1.6), xycoords=ax.transAxes)

    method_boxes = [
        ("Descriptor-process\ntraditional ML", "#0072B2"),
        ("Descriptor-process\nTabPFN / TabICL", "#D55E00"),
        ("Plain molecular\nGNN", "#CC79A7"),
        ("Graph + descriptor /\nprocess fusion", "#009E73"),
    ]
    y0 = 0.42
    for i, (txt, color) in enumerate(method_boxes):
        x = 0.02 + (i % 2) * 0.52
        y = y0 - (i // 2) * 0.255
        rect = patches.FancyBboxPatch(
            (x, y),
            0.43,
            0.19,
            boxstyle="round,pad=0.025,rounding_size=0.025",
            facecolor=color,
            edgecolor="#222222",
            linewidth=1.0,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.text(x + 0.215, y + 0.095, txt, ha="center", va="center", fontsize=11.3, color="white", fontweight="bold", transform=ax.transAxes)
    ax.text(0.02, 0.035, "Same folds, same targets, same metrics", fontsize=11.8, fontweight="bold", transform=ax.transAxes)


def panel_b(ax, group_df):
    ax.text(-0.12, 1.02, "b", transform=ax.transAxes, fontsize=17, fontweight="bold", va="bottom")
    ax.set_title("Best performance by method group", fontweight="bold", loc="left", pad=6, fontsize=15)
    x = np.arange(len(TARGETS))
    offsets = np.linspace(-0.26, 0.26, len(GROUPS))
    for i, group in enumerate(GROUPS):
        sub = group_df[group_df["Group"].astype(str).eq(group)]
        y = [sub[sub["Target"].astype(str).eq(t)]["Performance"].iloc[0] if len(sub[sub["Target"].astype(str).eq(t)]) else np.nan for t in TARGETS]
        ax.scatter(x + offsets[i], y, s=70, color=GROUP_COLORS[group], edgecolor="black", linewidth=0.6, label=group, zorder=3)
        ax.plot(x + offsets[i], y, color=GROUP_COLORS[group], alpha=0.45, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(TARGETS)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Predictive performance", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="lower right", fontsize=11)
    add_full_box(ax)
    bold_axes(ax, tick_labelsize=12, axis_labelsize=14)


def panel_c(ax, group_df):
    ax.text(-0.12, 1.02, "c", transform=ax.transAxes, fontsize=17, fontweight="bold", va="bottom")
    ax.set_title("TabICL vs. strongest graph/fusion", fontweight="bold", loc="left", pad=6, fontsize=15)
    y_pos = np.arange(len(TARGETS))[::-1]
    for i, target in enumerate(TARGETS):
        sub = group_df[group_df["Target"].astype(str).eq(target)]
        tab = sub[sub["Group"].astype(str).eq("TabICL")].iloc[0]
        gnn = sub[sub["Group"].astype(str).eq("Graph/Fusion")].iloc[0]
        y = y_pos[i]
        ax.plot([gnn["Performance"], tab["Performance"]], [y, y], color="#555555", linewidth=1.6, zorder=1)
        ax.scatter(gnn["Performance"], y, s=85, color=GROUP_COLORS["Graph/Fusion"], edgecolor="black", linewidth=0.7, label="Best Graph/Fusion" if i == 0 else "", zorder=3)
        ax.scatter(tab["Performance"], y, s=100, color=GROUP_COLORS["TabICL"], edgecolor="black", linewidth=0.7, label="TabICL" if i == 0 else "", zorder=4)
        ax.text(tab["Performance"] + 0.012, y, f"+{tab['Performance'] - gnn['Performance']:.3f}", va="center", fontsize=11, fontweight="bold")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(TARGETS)
    ax.set_xlim(0.55, 0.95)
    ax.set_xlabel("R2 or AUC", fontsize=13, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.02, 0.02), fontsize=11)
    add_full_box(ax)
    bold_axes(ax, tick_labelsize=12, axis_labelsize=14)


def panel_d(ax, df):
    ax.text(-0.34, 1.02, "d", transform=ax.transAxes, fontsize=17, fontweight="bold", va="bottom")
    ax.set_title("Individual model ranking heatmap", fontweight="bold", loc="left", pad=6, fontsize=15)
    df = df.copy()
    df["Display_Model"] = df["Model"].map(short_model_name)
    pivot_all = df.pivot_table(index="Display_Model", columns="Target", values="Mean_Primary", aggfunc="max")
    priority = [
        "DescriptorProcess_TabICL",
        "DescriptorProcess_TabPFN",
        "DescriptorProcess_LightGBM",
        "DescriptorProcess_XGBoost",
        "DescriptorProcess_CatBoost",
        "DescriptorProcess_RandomForest",
        "DescriptorProcess_ExtraTrees",
        "Graph_GCN_graph_only",
        "Graph_GAT_graph_only",
        "Graph_GINE_graph_only",
        "Graph_GINE_weighted_sum",
        "Graph_GCN_weighted_sum",
        "Graph_GIN_film",
        "Graph_GIN_attention",
        "Graph_GAT_attention",
    ]
    rows = [r for r in priority if r in pivot_all.index]
    pivot = pivot_all.reindex(rows).reindex(columns=["LOI", "Tg", "TENSILE", "UL94"])
    red_cmap = mcolors.LinearSegmentedColormap.from_list("deep_to_light_red", ["#fff5f0", "#fb6a4a", "#67000d"])
    im = ax.imshow(pivot.values, vmin=0, vmax=1, cmap=red_cmap, aspect="auto")
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(TARGETS)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        vals = pivot.iloc[i].values
        rank_values = pd.Series(vals).rank(ascending=False, method="min")
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8.0, fontweight="bold")
    bold_axes(ax, tick_labelsize=11, axis_labelsize=12, bold_y_ticklabels=False)
    return im


def panel_e(fig, gs_cell, df):
    subgs = gs_cell.subgridspec(2, 2, wspace=0.28, hspace=0.42)
    axes = []
    for k, target in enumerate(TARGETS):
        ax = fig.add_subplot(subgs[k // 2, k % 2])
        axes.append(ax)
        target_key = TARGET_KEY[target]
        sub = df[(df["Target"].eq(target_key)) & (df["Model"].str.startswith("Graph_", na=False)) & (~df["Model"].str.endswith("_Fusion_graph_only", na=False))]
        matrix = pd.DataFrame(np.nan, index=CONVS, columns=FUSIONS)
        for _, row in sub.iterrows():
            conv, fusion = split_graph_name(row["Model"])
            if conv in CONVS and fusion in FUSIONS:
                matrix.loc[conv, fusion] = row["Mean_Primary"]
        blue_cmap = mcolors.LinearSegmentedColormap.from_list("light_to_deep_blue", ["#eff6ff", "#60a5fa", "#08306b"])
        im = ax.imshow(matrix.values, vmin=0, vmax=1, cmap=blue_cmap, aspect="auto")
        best = np.nanmax(matrix.values)
        ax.set_title(target, fontweight="bold", fontsize=11)
        ax.set_xticks(np.arange(len(FUSIONS)))
        ax.set_xticklabels(FUSIONS, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(CONVS)))
        ax.set_yticklabels(CONVS, fontsize=8)
        for i in range(len(CONVS)):
            for j in range(len(FUSIONS)):
                value = matrix.iloc[i, j]
                if np.isfinite(value):
                    star = "*" if abs(value - best) < 1e-9 else ""
                    ax.text(j, i, f"{value:.2f}{star}", ha="center", va="center", fontsize=7.5, fontweight="bold")
        add_full_box(ax, lw=1.15)
        bold_axes(ax, lw=1.15, tick_labelsize=9.5, axis_labelsize=10.5)
    axes[0].text(-0.35, 1.28, "e", transform=axes[0].transAxes, fontsize=17, fontweight="bold", va="bottom")
    axes[0].text(0, 1.28, "Graph architecture × fusion strategy", transform=axes[0].transAxes, fontsize=13, fontweight="bold", va="bottom")
    cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02)
    cbar.set_label("R2 or AUC", fontsize=8)


def panel_f(ax, group_df):
    ax.text(-0.12, 1.02, "f", transform=ax.transAxes, fontsize=17, fontweight="bold", va="bottom")
    ax.set_title("Task-dependent predictability", fontweight="bold", loc="left", pad=6, fontsize=15)
    tab = group_df[group_df["Group"].astype(str).eq("TabICL")].copy()
    y = [tab[tab["Target"].astype(str).eq(t)]["Performance"].iloc[0] for t in TARGETS]
    x = np.arange(len(TARGETS))
    ax.plot(x, y, color="#D55E00", linewidth=1.8)
    ax.scatter(x, y, s=90, color="#D55E00", edgecolor="black", linewidth=0.7, zorder=3)
    for i, value in enumerate(y):
        ax.text(i, value + 0.018, f"{value:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(TARGETS)
    ax.set_ylim(0.6, 1.0)
    ax.set_ylabel("TabICL performance", fontsize=13, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    add_full_box(ax)
    bold_axes(ax, tick_labelsize=12, axis_labelsize=14)


def main() -> None:
    configure_style()
    df = load_all()
    group_df = best_group_table(df)
    group_df.to_csv(OUT / "figure2_method_group_values.csv", index=False, encoding="utf-8-sig")

    fig = plt.figure(figsize=(21.5, 12.5), dpi=300)
    gs = GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=[1.05, 1.15],
        hspace=0.34,
    )
    top_gs = gs[0, 0].subgridspec(
        1,
        2,
        width_ratios=[1.15, 1.35],
        wspace=0.28,
    )
    bottom_gs = gs[1, 0].subgridspec(
        1,
        3,
        width_ratios=[1.05, 1.02, 1.55],
        wspace=0.36,
    )
    ax_a = fig.add_subplot(top_gs[0, 0])
    ax_b = fig.add_subplot(top_gs[0, 1])
    ax_c = fig.add_subplot(bottom_gs[0, 0])
    ax_d = fig.add_subplot(bottom_gs[0, 1])
    ax_e = fig.add_subplot(bottom_gs[0, 2])

    panel_method_group(ax_a, group_df, letter="a")
    panel_tabicl_vs_graph(ax_b, group_df, letter="b")
    im_c = panel_model_heatmap(ax_c, df, letter="c")
    panel_graph_distribution(ax_d, df, group_df, letter="d")
    panel_task_cards(ax_e, group_df, letter="e")

    cbar = fig.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.02)
    cbar.set_label("R2 or AUC", fontsize=11, fontweight="bold")
    cbar.ax.tick_params(labelsize=10, width=1.1)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")
    fig.savefig(OUT / "Figure2_unified_benchmark.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / "Figure2_unified_benchmark.pdf", bbox_inches="tight")
    print(f"Figure 2 written to {OUT}")


if __name__ == "__main__":
    main()
