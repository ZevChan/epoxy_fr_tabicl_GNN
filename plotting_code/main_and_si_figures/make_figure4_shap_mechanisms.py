from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "translated_text"
FIG_DIR.mkdir(exist_ok=True)

INK = "#010202"
GRID = "#E6E6E6"
TABICL = "#F28F1E"
BLUE = "#4C65AF"
GREEN = "#41B36C"
RED = "#F26758"
GREY = "#919191"

TASKS = [
    ("LOI", "LOI", ROOT / "LOI" / "Result_Final_TabICL_Interpret" / "saved_data"),
    ("UL94", "UL-94", ROOT / "94" / "Result_Final_TabICL_Interpret_Cls" / "saved_data"),
    ("Tg", "Tg", ROOT / "Tg" / "Result_Final_TabICL_Interpret" / "saved_data"),
    ("TENSILE", "Tensile", ROOT / "TENSILE" / "Result_Final_TabICL_Interpret" / "saved_data"),
]

CATEGORY_COLORS = {
    "Formulation": "#F28F1E",
    "Process": "#41B36C",
    "EP descriptors": "#4C65AF",
    "FR descriptors": "#F26758",
    "CURING descriptors": "#8E62AA",
    "Other": "#919191",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 1.2,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )


def add_full_box(ax, lw: float = 1.15) -> None:
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(lw)
        ax.spines[side].set_color(INK)


def bold_axes(ax, tick: float = 9.5, label: float = 10.5) -> None:
    ax.tick_params(axis="both", width=1.15, labelsize=tick)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    ax.xaxis.label.set_fontsize(label)
    ax.yaxis.label.set_fontsize(label)


def panel_label(ax, letter: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=18, fontweight="bold", va="bottom", color=INK)


def category(feature: str) -> str:
    formulation = {
        "Flame_retardant_AdditionAmount(wt%)",
        "Curing_agent_AdditionAmount(wt%)",
        "EP_wt_fraction",
        "FR_wt_fraction",
        "CURING_wt_fraction",
    }
    process = {"T_max", "t_total", "Q_thermal"}
    if feature in formulation:
        return "Formulation"
    if feature in process:
        return "Process"
    if feature.startswith("EP_") or feature in {"EEW"}:
        return "EP descriptors"
    if feature.startswith("FR_"):
        return "FR descriptors"
    if feature.startswith("CURING_"):
        return "CURING descriptors"
    return "Other"


def load_task(base: Path) -> dict:
    with open(base / "plot_data.pkl", "rb") as f:
        obj = pickle.load(f)
    features = list(obj["top_k_features"])
    shap_values = np.asarray(obj["shap_values"], dtype=float)
    x_test = np.asarray(obj["X_test"], dtype=float)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]
    return {"features": features, "shap": shap_values, "x": x_test}


def load_all() -> dict[str, dict]:
    return {key: load_task(base) for key, _, base in TASKS}


def mean_abs_by_feature(task_data: dict) -> pd.Series:
    return pd.Series(np.nanmean(np.abs(task_data["shap"]), axis=0), index=task_data["features"])


def short_feature(name: str) -> str:
    return name.replace("Flame_retardant_AdditionAmount(wt%)", "FR loading").replace("Curing_agent_AdditionAmount(wt%)", "Curing amount")


def select_heatmap_features(all_data: dict[str, dict], n: int = 30) -> list[str]:
    scores: dict[str, float] = {}
    for data in all_data.values():
        s = mean_abs_by_feature(data)
        top = s.sort_values(ascending=False).head(18)
        for feat, val in top.items():
            scores[feat] = max(scores.get(feat, 0.0), float(val))
    priority = {
        "Formulation": 0,
        "Process": 1,
        "EP descriptors": 2,
        "FR descriptors": 3,
        "CURING descriptors": 4,
        "Other": 5,
    }
    feats = sorted(scores, key=lambda f: (priority[category(f)], -scores[f], f))
    return feats[:n]


def panel_heatmap(ax, all_data: dict[str, dict]) -> None:
    panel_label(ax, "a", x=-0.16)
    feats = select_heatmap_features(all_data, 30)
    mat = []
    for key, _, _ in TASKS:
        s = mean_abs_by_feature(all_data[key])
        mat.append([s.get(f, 0.0) for f in feats])
    mat = np.array(mat).T
    row_max = np.nanmax(mat, axis=1)
    norm = np.divide(mat, row_max[:, None], out=np.zeros_like(mat), where=row_max[:, None] > 0)
    cmap = mcolors.LinearSegmentedColormap.from_list("white_orange", ["#fff7ec", "#fdb863", "#b35806"])
    ax.imshow(norm, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(TASKS)))
    ax.set_xticklabels([display for _, display, _ in TASKS])
    ax.set_yticks(np.arange(len(feats)))
    ax.set_yticklabels([short_feature(f) for f in feats], fontsize=7.4)
    ax.set_title("Cross-task SHAP importance", fontsize=15, fontweight="bold", loc="left", pad=7)
    for i, feat in enumerate(feats):
        ax.add_patch(
            patches.Rectangle(
                (-0.86, i - 0.5),
                0.17,
                1.0,
                facecolor=CATEGORY_COLORS[category(feat)],
                edgecolor="none",
                clip_on=False,
            )
        )
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] > 0:
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6.6, fontweight="bold")
    add_full_box(ax)
    bold_axes(ax, tick=8.6, label=10)
    handles = [patches.Patch(color=color, label=label) for label, color in CATEGORY_COLORS.items() if label != "Other"]
    ax.legend(handles=handles, frameon=False, fontsize=7.6, loc="lower center", bbox_to_anchor=(0.52, -0.18), ncol=2)


def compact_beeswarm(ax, data: dict, title: str, letter: str, callout: str, preferred: list[str]) -> None:
    panel_label(ax, letter, x=-0.12)
    features = data["features"]
    shap_values = data["shap"]
    x_test = data["x"]
    importance = mean_abs_by_feature(data)
    ordered = [f for f in preferred if f in features]
    for f in importance.sort_values(ascending=False).index:
        if f not in ordered:
            ordered.append(f)
        if len(ordered) >= 8:
            break
    ordered = ordered[:8]
    rng = np.random.default_rng(7)
    max_abs = max(float(np.nanmax(np.abs(shap_values[:, [features.index(f) for f in ordered]]))), 1e-6)
    for row, feat in enumerate(ordered[::-1]):
        idx = features.index(feat)
        sv = shap_values[:, idx]
        vals = x_test[:, idx]
        if np.nanmax(vals) > np.nanmin(vals):
            color_val = (vals - np.nanmin(vals)) / (np.nanmax(vals) - np.nanmin(vals))
        else:
            color_val = np.full_like(vals, 0.5)
        y = np.full(len(sv), row) + rng.normal(0, 0.08, len(sv))
        ax.scatter(sv, y, c=color_val, cmap="coolwarm", s=15, alpha=0.82, edgecolor=INK, linewidth=0.08)
    ax.axvline(0, color=INK, linewidth=1.0)
    ax.set_yticks(np.arange(len(ordered)))
    ax.set_yticklabels([short_feature(f) for f in ordered[::-1]], fontsize=8.3)
    ax.set_xlim(-max_abs * 1.18, max_abs * 1.18)
    ax.set_xlabel("SHAP value")
    ax.set_title(title, fontsize=15, fontweight="bold", loc="left", pad=7)
    ax.grid(axis="x", color=GRID, linestyle="--", linewidth=0.8)
    ax.text(
        0.99,
        0.04,
        callout,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.6,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor=INK, boxstyle="round,pad=0.28", linewidth=0.85),
    )
    add_full_box(ax)
    bold_axes(ax, tick=8.7, label=10.5)


def panel_summary(ax) -> None:
    panel_label(ax, "f", x=-0.10)
    ax.set_axis_off()
    ax.set_title("Property-specific mechanism summary", fontsize=15, fontweight="bold", loc="left", pad=7)
    cards = [
        ("LOI", "Formulation loading\n+ P-containing groups", TABICL),
        ("UL-94", "Loading + conjugated\ncharring structure\n+ compatibility", GREEN),
        ("Tg", "EEW + T_max\n+ crosslinking potential", BLUE),
        ("Tensile", "Q_thermal + steric/polar\ndescriptors + network\nimperfection", RED),
    ]
    for i, (title, body, color) in enumerate(cards):
        y = 0.73 - i * 0.235
        rect = patches.FancyBboxPatch(
            (0.04, y),
            0.92,
            0.18,
            boxstyle="round,pad=0.018,rounding_size=0.024",
            facecolor="#F7F7F7",
            edgecolor=INK,
            linewidth=1.1,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.add_patch(patches.Rectangle((0.04, y), 0.025, 0.18, transform=ax.transAxes, facecolor=color, edgecolor="none"))
        ax.text(0.10, y + 0.095, title, transform=ax.transAxes, fontsize=13, fontweight="bold", va="center")
        ax.text(0.39, y + 0.095, body, transform=ax.transAxes, fontsize=10.5, fontweight="bold", va="center")


def main() -> None:
    configure_style()
    all_data = load_all()
    fig = plt.figure(figsize=(21.0, 12.5), dpi=300)
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.35, 1.0, 1.0], height_ratios=[1.0, 1.0], wspace=0.34, hspace=0.35)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    panel_heatmap(ax_a, all_data)
    compact_beeswarm(
        ax_b,
        all_data["LOI"],
        "LOI mechanism attribution",
        "b",
        "P-containing FR\n-> radical quenching /\nchar promotion\n-> higher LOI",
        ["Flame_retardant_AdditionAmount(wt%)", "FR_F01[C-P]", "FR_B01[C-P]", "FR_Eta_betaP_A", "EP_wt_fraction"],
    )
    compact_beeswarm(
        ax_c,
        all_data["UL94"],
        "UL-94 mechanism attribution",
        "c",
        "Conjugated/aromatic FR\n-> condensed-phase charring /\ndripping suppression\n-> V-0 probability",
        ["Flame_retardant_AdditionAmount(wt%)", "FR_max_conj_path", "CURING_Mor30p", "EP_wt_fraction", "Curing_agent_AdditionAmount(wt%)"],
    )
    compact_beeswarm(
        ax_d,
        all_data["Tg"],
        "Tg mechanism attribution",
        "d",
        "Low EEW + high T_max\n-> higher crosslink density\n-> restricted segmental motion\n-> higher Tg",
        ["EEW", "T_max", "Q_thermal", "CURING_RBF", "CURING_SssCH2", "EP_stdMW"],
    )
    compact_beeswarm(
        ax_e,
        all_data["TENSILE"],
        "Tensile-strength mechanism attribution",
        "e",
        "Curing history\n-> network perfection /\ninternal stress\nBulky/polar FR\n-> network disruption",
        ["Q_thermal", "FR_E1m", "FR_TPSA_efficiency", "Flame_retardant_AdditionAmount(wt%)", "EP_SIC5"],
    )
    panel_summary(ax_f)

    fig.savefig(FIG_DIR / "Figure4_SHAP_mechanisms.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG_DIR / "Figure4_SHAP_mechanisms.pdf", bbox_inches="tight")
    print(f"Figure 4 written to {FIG_DIR}")


if __name__ == "__main__":
    main()
