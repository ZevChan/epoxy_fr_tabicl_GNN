from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
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

TASK_DIRS = {
    "LOI": ROOT / "LOI" / "Result_Final_TabICL_Interpret" / "saved_data",
    "UL94": ROOT / "94" / "Result_Final_TabICL_Interpret_Cls" / "saved_data",
    "Tg": ROOT / "Tg" / "Result_Final_TabICL_Interpret" / "saved_data",
    "TENSILE": ROOT / "TENSILE" / "Result_Final_TabICL_Interpret" / "saved_data",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 1.15,
            "axes.labelsize": 9.8,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )


def add_full_box(ax, lw: float = 1.1) -> None:
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(lw)
        ax.spines[side].set_color(INK)


def bold_axes(ax, tick: float = 8.8, label: float = 9.8) -> None:
    ax.tick_params(axis="both", width=1.1, labelsize=tick)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontweight("bold")
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    ax.xaxis.label.set_fontsize(label)
    ax.yaxis.label.set_fontsize(label)


def panel_label(ax, letter: str, x: float = -0.16, y: float = 1.04) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=16, fontweight="bold", va="bottom", color=INK)


def load_task(key: str) -> dict:
    with open(TASK_DIRS[key] / "plot_data.pkl", "rb") as f:
        obj = pickle.load(f)
    shap_values = np.asarray(obj["shap_values"], dtype=float)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, -1]
    return {
        "features": list(obj["top_k_features"]),
        "x": np.asarray(obj["X_test"], dtype=float),
        "shap": shap_values,
    }


def feature_arrays(data: dict, feature: str) -> tuple[np.ndarray, np.ndarray]:
    idx = data["features"].index(feature)
    return data["x"][:, idx].astype(float), data["shap"][:, idx].astype(float)


def values(data: dict, feature: str) -> np.ndarray:
    return data["x"][:, data["features"].index(feature)].astype(float)


def robust_limits(x: np.ndarray, low: float = 1, high: float = 99) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 0, 1
    lo, hi = np.nanpercentile(x, [low, high])
    if lo == hi:
        lo, hi = np.nanmin(x), np.nanmax(x)
    pad = (hi - lo) * 0.04 if hi > lo else 1
    return lo - pad, hi + pad


def binned_line(x: np.ndarray, y: np.ndarray, n_bins: int = 18) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    bins = np.array_split(np.arange(len(xs)), min(n_bins, len(xs)))
    return np.array([np.nanmedian(xs[b]) for b in bins if len(b)]), np.array([np.nanmedian(ys[b]) for b in bins if len(b)])


def dependence_panel(ax, data: dict, main_feature: str, color_feature: str, title: str, letter: str, threshold_lines: list[tuple[float, str]], x_label: str) -> None:
    panel_label(ax, letter)
    x, shap = feature_arrays(data, main_feature)
    c = values(data, color_feature)
    sc = ax.scatter(x, shap, c=c, cmap="coolwarm", s=24, alpha=0.84, edgecolor=INK, linewidth=0.12)
    bx, by = binned_line(x, shap)
    ax.plot(bx, by, color=INK, linewidth=2.0)
    ax.axhline(0, color=GREY, linestyle="--", linewidth=1.1)
    for val, text in threshold_lines:
        ax.axvline(val, color=RED, linestyle="--", linewidth=1.25)
        ax.text(val, 0.96, text, transform=ax.get_xaxis_transform(), rotation=90, va="top", ha="right", fontsize=8.2, fontweight="bold", color=RED)
    ax.set_xlim(*robust_limits(x, 0.5, 99.5))
    ax.set_title(title, fontsize=13.5, fontweight="bold", loc="left", pad=6)
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"SHAP for {short_name(main_feature)}")
    ax.grid(color=GRID, linestyle="--", linewidth=0.75)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label(short_name(color_feature), fontsize=8.6, fontweight="bold")
    cbar.ax.tick_params(labelsize=7.6, width=0.8)
    add_full_box(ax)
    bold_axes(ax)


def short_name(feature: str) -> str:
    return (
        feature.replace("Flame_retardant_AdditionAmount(wt%)", "FR loading")
        .replace("Curing_agent_AdditionAmount(wt%)", "Curing amount")
        .replace("FR_TPSA_efficiency", "FR TPSA efficiency")
    )


def surface_panel(ax, data: dict, x_feature: str, y_feature: str, shap_feature: str, title: str, letter: str, x_label: str, y_label: str, markers: list[tuple[str, float, str]] = []) -> None:
    panel_label(ax, letter)
    x = values(data, x_feature)
    y = values(data, y_feature)
    _, shap = feature_arrays(data, shap_feature)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(shap)
    x, y, shap = x[mask], y[mask], shap[mask]
    x_lo, x_hi = np.nanpercentile(x, [2, 98])
    y_lo, y_hi = np.nanpercentile(y, [2, 98])
    x_bins = np.linspace(x_lo, x_hi, 22)
    y_bins = np.linspace(y_lo, y_hi, 22)
    grid = np.full((len(y_bins) - 1, len(x_bins) - 1), np.nan)
    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            inside = (x >= x_bins[i]) & (x < x_bins[i + 1]) & (y >= y_bins[j]) & (y < y_bins[j + 1])
            if inside.sum() >= 2:
                grid[j, i] = np.nanmean(shap[inside])
    vmax = np.nanpercentile(np.abs(grid), 95) if np.isfinite(grid).any() else 1
    im = ax.imshow(
        grid,
        origin="lower",
        extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]],
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.scatter(x, y, s=6, color=INK, alpha=0.18, edgecolor="none")
    for axis, val, text in markers:
        if axis == "x":
            ax.axvline(val, color=INK, linestyle="--", linewidth=1.1)
            ax.text(val, 0.98, text, transform=ax.get_xaxis_transform(), rotation=90, va="top", ha="right", fontsize=8.0, fontweight="bold")
        else:
            ax.axhline(val, color=INK, linestyle="--", linewidth=1.1)
            ax.text(0.02, val, text, transform=ax.get_yaxis_transform(), va="bottom", ha="left", fontsize=8.0, fontweight="bold")
    ax.set_title(title, fontsize=13.5, fontweight="bold", loc="left", pad=6)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(color=GRID, linestyle="--", linewidth=0.55, alpha=0.6)
    cbar = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label(f"Mean SHAP for {short_name(shap_feature)}", fontsize=8.2, fontweight="bold")
    cbar.ax.tick_params(labelsize=7.5, width=0.8)
    add_full_box(ax)
    bold_axes(ax)


def main() -> None:
    configure_style()
    data = {key: load_task(key) for key in TASK_DIRS}
    fig = plt.figure(figsize=(21.5, 11.5), dpi=300)
    gs = GridSpec(4, 2, figure=fig, hspace=0.48, wspace=0.30)

    dependence_panel(
        fig.add_subplot(gs[0, 0]),
        data["LOI"],
        "Flame_retardant_AdditionAmount(wt%)",
        "FR_F01[C-P]",
        "LOI dependence on FR loading",
        "a",
        [(15, "15 wt% threshold")],
        "FR loading (wt%)",
    )
    surface_panel(
        fig.add_subplot(gs[0, 1]),
        data["LOI"],
        "Flame_retardant_AdditionAmount(wt%)",
        "FR_F01[C-P]",
        "Flame_retardant_AdditionAmount(wt%)",
        "LOI interaction landscape",
        "b",
        "FR loading (wt%)",
        "FR_F01[C-P]",
        [("x", 15, "15 wt%")],
    )
    dependence_panel(
        fig.add_subplot(gs[1, 0]),
        data["UL94"],
        "FR_max_conj_path",
        "Flame_retardant_AdditionAmount(wt%)",
        "UL-94 dependence on FR conjugation",
        "c",
        [(15, "conj. path ~15")],
        "FR_max_conj_path",
    )
    surface_panel(
        fig.add_subplot(gs[1, 1]),
        data["UL94"],
        "FR_max_conj_path",
        "Flame_retardant_AdditionAmount(wt%)",
        "FR_max_conj_path",
        "UL-94 interaction landscape",
        "d",
        "FR_max_conj_path",
        "FR loading (wt%)",
        [("x", 15, "path ~15")],
    )
    dependence_panel(
        fig.add_subplot(gs[2, 0]),
        data["Tg"],
        "EEW",
        "T_max",
        "Tg dependence on EEW",
        "e",
        [(200, "EEW = 200")],
        "EEW",
    )
    surface_panel(
        fig.add_subplot(gs[2, 1]),
        data["Tg"],
        "EEW",
        "T_max",
        "EEW",
        "Tg interaction landscape",
        "f",
        "EEW",
        "T_max",
        [("x", 200, "EEW 200"), ("y", 160, "T_max 160 C")],
    )
    dependence_panel(
        fig.add_subplot(gs[3, 0]),
        data["TENSILE"],
        "Q_thermal",
        "FR_E1m",
        "Tensile dependence on Q_thermal",
        "g",
        [(600, "under-cured"), (800, "plateau")],
        "Q_thermal",
    )
    surface_panel(
        fig.add_subplot(gs[3, 1]),
        data["TENSILE"],
        "Q_thermal",
        "FR_E1m",
        "Q_thermal",
        "Tensile interaction landscape",
        "h",
        "Q_thermal",
        "FR_E1m",
        [("x", 600, "under-cured"), ("x", 800, "plateau")],
    )

    fig.savefig(FIG_DIR / "Figure5_SHAP_dependencies_boundaries.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG_DIR / "Figure5_SHAP_dependencies_boundaries.pdf", bbox_inches="tight")
    print(f"Figure 5 written to {FIG_DIR}")


if __name__ == "__main__":
    main()
