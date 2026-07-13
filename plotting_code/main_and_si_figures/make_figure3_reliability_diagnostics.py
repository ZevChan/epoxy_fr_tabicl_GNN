from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "outputs"
sys.path.insert(0, str(HERE))

from run_unified_benchmark import TARGET_CONFIG, descriptor_matrix, load_data  # noqa: E402


INK = "#010202"
GRID = "#E6E6E6"
TABICL = "#F28F1E"
BLUE = "#4C65AF"
GREEN = "#41B36C"
RED = "#F26758"
GREY = "#919191"

REG_TARGETS = [
    ("LOI", "LOI", "LOI", "R2", ""),
    ("Tg", "Tg", "Tg", "R2", "degC"),
    ("TENSILE", "Tensile", "Tensile", "R2", "MPa"),
]
TABICL_REG = "DescriptorProcess_TabICLRegressor"
TABICL_CLS = "DescriptorProcess_TabICLClassifier"


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 1.15,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
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


def bold_axes(ax, tick_labelsize: float = 9.5, axis_labelsize: float = 10.5) -> None:
    ax.tick_params(axis="both", width=1.15, labelsize=tick_labelsize)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    ax.xaxis.label.set_fontsize(axis_labelsize)
    ax.yaxis.label.set_fontsize(axis_labelsize)


def panel_label(ax, letter: str, x: float = -0.14, y: float = 1.04) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=17, fontweight="bold", va="bottom", color=INK)


def load_oof(target: str, model: str) -> pd.DataFrame:
    df = pd.read_csv(OUT / f"unified_{target}_oof_predictions.csv")
    df = df[df["Model"].eq(model)].copy()
    df["Residual"] = df["Predicted"] - df["Actual"]
    df["AbsError"] = df["Residual"].abs()
    return df


def regression_metrics(df: pd.DataFrame) -> tuple[float, float, float]:
    y = df["Actual"].values
    pred = df["Predicted"].values
    return (
        r2_score(y, pred),
        mean_squared_error(y, pred) ** 0.5,
        mean_absolute_error(y, pred),
    )


def panel_parity(ax, df: pd.DataFrame, display: str, letter: str, unit: str = "") -> None:
    panel_label(ax, letter)
    r2, rmse, mae = regression_metrics(df)
    x = df["Actual"].values
    y = df["Predicted"].values
    err = df["AbsError"].values
    lo = min(np.nanmin(x), np.nanmin(y))
    hi = max(np.nanmax(x), np.nanmax(y))
    pad = (hi - lo) * 0.06 if hi > lo else 1
    sc = ax.scatter(x, y, c=err, s=18, cmap="YlOrRd", edgecolor=INK, linewidth=0.15, alpha=0.78)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=GREY, linestyle="--", linewidth=1.3)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_title(f"{display} parity", fontsize=13.5, fontweight="bold", loc="left", pad=6)
    ax.set_xlabel("Experimental value")
    ax.set_ylabel("Predicted value")
    metric_text = f"R2 = {r2:.3f}\nRMSE = {rmse:.2f}{unit}\nMAE = {mae:.2f}{unit}"
    ax.text(
        0.04,
        0.96,
        metric_text,
        transform=ax.transAxes,
        va="top",
        fontsize=9.2,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor=INK, boxstyle="round,pad=0.25", linewidth=0.8),
    )
    ax.grid(color=GRID, linestyle="--", linewidth=0.8)
    add_full_box(ax)
    bold_axes(ax)
    return sc


def panel_error_cdf(ax, reg_data: dict[str, pd.DataFrame], letter: str = "d") -> None:
    panel_label(ax, letter)
    colors = {"LOI": BLUE, "Tg": TABICL, "Tensile": RED}
    for _, display, _, _, unit in REG_TARGETS:
        df = reg_data[display]
        errs = np.sort(df["AbsError"].values)
        cdf = np.arange(1, len(errs) + 1) / len(errs)
        q80 = np.quantile(errs, 0.80)
        ax.plot(errs, cdf, color=colors[display], linewidth=2.1, label=f"{display} ({q80:.2f}{unit})")
        ax.scatter([q80], [0.80], color=colors[display], edgecolor=INK, s=42, zorder=4)
    ax.axhline(0.80, color=GREY, linestyle="--", linewidth=1.2)
    ax.text(0.98, 0.82, "80% boundary", transform=ax.transAxes, ha="right", fontsize=9.5, fontweight="bold")
    ax.set_title("Absolute-error CDF", fontsize=13.5, fontweight="bold", loc="left", pad=6)
    ax.set_xlabel("Absolute error")
    ax.set_ylabel("Cumulative fraction")
    ax.set_ylim(0, 1.02)
    ax.grid(color=GRID, linestyle="--", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8.8, loc="lower right")
    add_full_box(ax)
    bold_axes(ax)


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (prob >= lo) & (prob < hi if hi < 1 else prob <= hi)
        if not np.any(mask):
            continue
        ece += mask.mean() * abs(prob[mask].mean() - y_true[mask].mean())
    return ece


def panel_calibration(ax, df: pd.DataFrame, letter: str = "e") -> None:
    panel_label(ax, letter)
    y = df["Actual"].astype(int).values
    prob = df["Score"].astype(float).clip(0, 1).values
    frac_pos, mean_pred = calibration_curve(y, prob, n_bins=10, strategy="quantile")
    ece = expected_calibration_error(y, prob)
    brier = brier_score_loss(y, prob)
    auc = roc_auc_score(y, prob)
    ax.plot([0, 1], [0, 1], color=GREY, linestyle="--", linewidth=1.25)
    ax.plot(mean_pred, frac_pos, color=TABICL, marker="o", markersize=5, linewidth=2.0)
    ax.axvline(0.5, color=INK, linestyle=":", linewidth=1.2)
    ax.text(0.52, 0.08, "threshold = 0.5", rotation=90, fontsize=8.8, fontweight="bold", va="bottom")
    ax.text(
        0.05,
        0.94,
        f"AUC = {auc:.3f}\nBrier = {brier:.3f}\nECE = {ece:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=9.4,
        fontweight="bold",
        bbox=dict(facecolor="white", edgecolor=INK, boxstyle="round,pad=0.25", linewidth=0.8),
    )
    ax.set_title("UL-94 probability calibration", fontsize=13.5, fontweight="bold", loc="left", pad=6)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed V-0 fraction")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(color=GRID, linestyle="--", linewidth=0.8)
    add_full_box(ax)
    bold_axes(ax)


def panel_threshold(ax, df: pd.DataFrame, letter: str = "f") -> None:
    panel_label(ax, letter)
    y = df["Actual"].astype(int).values
    prob = df["Score"].astype(float).clip(0, 1).values
    thresholds = np.linspace(0.02, 0.98, 97)
    fpr, fnr = [], []
    for t in thresholds:
        pred = prob >= t
        fp = np.sum((pred == 1) & (y == 0))
        fn = np.sum((pred == 0) & (y == 1))
        tn = np.sum((pred == 0) & (y == 0))
        tp = np.sum((pred == 1) & (y == 1))
        fpr.append(fp / max(fp + tn, 1))
        fnr.append(fn / max(fn + tp, 1))
    fpr = np.array(fpr)
    fnr = np.array(fnr)
    balanced = thresholds[np.argmin(np.abs(fpr - fnr))]
    safe_candidates = thresholds[fpr <= 0.10]
    safety = safe_candidates[0] if len(safe_candidates) else thresholds[np.argmin(fpr)]
    ax.plot(thresholds, fpr, color=RED, linewidth=2.0, label="False positive rate")
    ax.plot(thresholds, fnr, color=BLUE, linewidth=2.0, label="False negative rate")
    for t, label, color, y_text in [(0.5, "default", INK, 0.86), (safety, "safety-screening", RED, 0.74), (balanced, "balanced", BLUE, 0.62)]:
        ax.axvline(t, color=color, linestyle="--", linewidth=1.2)
        ax.text(t + 0.01, y_text, f"{label}\n{t:.2f}", fontsize=8.6, fontweight="bold", color=color)
    ax.set_title("UL-94 threshold-dependent errors", fontsize=13.5, fontweight="bold", loc="left", pad=6)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Error rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(color=GRID, linestyle="--", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8.6, loc="upper right")
    add_full_box(ax)
    bold_axes(ax)


def binned_smooth(x: np.ndarray, y: np.ndarray, n_bins: int = 18) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    bins = np.array_split(np.arange(len(xs)), min(n_bins, len(xs)))
    bx = np.array([np.nanmedian(xs[b]) for b in bins if len(b)])
    by = np.array([np.nanmedian(ys[b]) for b in bins if len(b)])
    return bx, by


def panel_residuals(fig, gs_cell, reg_data: dict[str, pd.DataFrame]) -> None:
    ax_outer = fig.add_subplot(gs_cell)
    panel_label(ax_outer, "g", x=-0.08)
    ax_outer.set_axis_off()
    ax_outer.set_title("Residual diagnostics", fontsize=13.5, fontweight="bold", loc="left", pad=6)
    specs = [
        ("LOI", "LOI", "Flame_retardant_AdditionAmount(wt%)", "FR loading (wt%)"),
        ("Tg", "Tg", "EEW", "EEW"),
        ("TENSILE", "Tensile", "Q_thermal", "Q_thermal"),
    ]
    for i, (target_key, display, feat, xlabel) in enumerate(specs):
        ax = ax_outer.inset_axes([0.03 + i * 0.325, 0.08, 0.29, 0.78])
        df_raw, _, feature_cols = load_data(target_key)
        pred = reg_data[display].copy()
        merged = pred.merge(df_raw[[feat]].reset_index().rename(columns={"index": "Index"}), on="Index", how="left")
        x = merged[feat].astype(float).values
        y = merged["Residual"].astype(float).values
        err = merged["AbsError"].astype(float).values
        mask = np.isfinite(x) & np.isfinite(y)
        x, y, err = x[mask], y[mask], err[mask]
        ax.scatter(x, y, c=err, cmap="YlOrRd", s=11, alpha=0.65, edgecolor=INK, linewidth=0.08)
        ax.axhline(0, color=GREY, linestyle="--", linewidth=1.0)
        if len(x) > 8:
            bx, by = binned_smooth(x, y)
            ax.plot(bx, by, color=INK, linewidth=1.6)
        ax.set_title(display, fontsize=10.5, fontweight="bold", pad=3)
        ax.set_xlabel(xlabel, fontsize=8.2, fontweight="bold")
        ax.set_ylabel("Residual" if i == 0 else "", fontsize=8.2, fontweight="bold")
        ax.grid(color=GRID, linestyle="--", linewidth=0.6)
        add_full_box(ax, lw=0.9)
        ax.tick_params(axis="both", labelsize=7.2, width=0.9)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")


def compute_ad_for_loi() -> pd.DataFrame:
    target = "LOI"
    df_raw, _, feature_cols = load_data(target)
    X, selected = descriptor_matrix(df_raw, feature_cols, target, quick=False)
    X = SimpleImputer(strategy="median").fit_transform(X)
    Xs = StandardScaler().fit_transform(X)
    oof = load_oof(target, TABICL_REG)
    folds = oof[["Index", "Fold", "AbsError"]].drop_duplicates("Index").set_index("Index")
    ad_distance = np.full(len(df_raw), np.nan)
    for fold in sorted(oof["Fold"].unique()):
        test_idx = oof[oof["Fold"].eq(fold)]["Index"].astype(int).values
        train_idx = np.setdiff1d(np.arange(len(df_raw)), test_idx)
        nn = NearestNeighbors(n_neighbors=min(5, len(train_idx))).fit(Xs[train_idx])
        dist, _ = nn.kneighbors(Xs[test_idx])
        ad_distance[test_idx] = dist.mean(axis=1)
    train_nn = NearestNeighbors(n_neighbors=6).fit(Xs)
    dist_all, _ = train_nn.kneighbors(Xs)
    boundary = np.quantile(dist_all[:, 1:].mean(axis=1), 0.95)
    try:
        embed = TSNE(n_components=2, perplexity=35, init="pca", learning_rate="auto", random_state=42).fit_transform(Xs)
    except Exception:
        embed = PCA(n_components=2, random_state=42).fit_transform(Xs)
    out = pd.DataFrame(
        {
            "UMAP1": embed[:, 0],
            "UMAP2": embed[:, 1],
            "AD_Distance": ad_distance,
            "Inside_AD": ad_distance <= boundary,
            "AbsError": folds.reindex(np.arange(len(df_raw)))["AbsError"].values,
        }
    )
    out["Boundary"] = boundary
    return out


def panel_ad(fig, gs_cell) -> None:
    ax = fig.add_subplot(gs_cell)
    panel_label(ax, "h", x=-0.12)
    ad = compute_ad_for_loi()
    ax.scatter(ad["UMAP1"], ad["UMAP2"], s=8, color="#D8D8D8", alpha=0.55, edgecolor="none", label="training manifold")
    outside = ad[~ad["Inside_AD"]]
    inside = ad[ad["Inside_AD"]]
    ax.scatter(inside["UMAP1"], inside["UMAP2"], c=inside["AbsError"], cmap="YlOrRd", s=18, alpha=0.82, edgecolor=INK, linewidth=0.08, label="test inside AD")
    ax.scatter(outside["UMAP1"], outside["UMAP2"], c=outside["AbsError"], cmap="YlOrRd", s=34, alpha=0.95, edgecolor=RED, linewidth=0.65, marker="^", label="test outside AD")
    ax.set_title("Applicability-domain map", fontsize=13.5, fontweight="bold", loc="left", pad=6)
    ax.set_xlabel("2D descriptor-process embedding")
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.grid(color=GRID, linestyle="--", linewidth=0.7)
    add_full_box(ax)
    bold_axes(ax, tick_labelsize=8.5, axis_labelsize=9.5)
    inset = ax.inset_axes([0.60, 0.10, 0.36, 0.34])
    inset.scatter(ad["AD_Distance"], ad["AbsError"], s=9, color=BLUE, alpha=0.45, edgecolor="none")
    bx, by = binned_smooth(ad["AD_Distance"].values, ad["AbsError"].values, n_bins=12)
    inset.plot(bx, by, color=INK, linewidth=1.4)
    inset.axvline(ad["Boundary"].iloc[0], color=RED, linestyle="--", linewidth=1.0)
    inset.set_xlabel("AD distance", fontsize=7.2, fontweight="bold")
    inset.set_ylabel("|error|", fontsize=7.2, fontweight="bold")
    inset.tick_params(axis="both", labelsize=6.5, width=0.8)
    add_full_box(inset, lw=0.8)
    ax.legend(frameon=False, fontsize=7.8, loc="upper right")


def main() -> None:
    configure_style()
    reg_data = {display: load_oof(target_key, TABICL_REG) for target_key, display, *_ in REG_TARGETS}
    ul94 = load_oof("UL94", TABICL_CLS)

    fig = plt.figure(figsize=(22.0, 10.4), dpi=300)
    gs = GridSpec(2, 4, figure=fig, wspace=0.42, hspace=0.48)
    axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    for ax, (target_key, display, _, _, unit), letter in zip(axes[:3], REG_TARGETS, ["a", "b", "c"]):
        panel_parity(ax, reg_data[display], display, letter, unit=f" {unit}" if unit else "")
    panel_error_cdf(axes[3], reg_data, letter="d")

    ax_e = fig.add_subplot(gs[1, 0])
    ax_f = fig.add_subplot(gs[1, 1])
    panel_calibration(ax_e, ul94, letter="e")
    panel_threshold(ax_f, ul94, letter="f")
    panel_residuals(fig, gs[1, 2], reg_data)
    panel_ad(fig, gs[1, 3])

    fig.savefig(OUT / "Figure3_reliability_diagnostics.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT / "Figure3_reliability_diagnostics.pdf", bbox_inches="tight")
    print(f"Figure 3 written to {OUT}")


if __name__ == "__main__":
    main()
