from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import auc, confusion_matrix, roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "outputs"
FIG_DIR = ROOT / "translated_text"
FIG_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))

from run_unified_benchmark import descriptor_matrix, load_data  # noqa: E402


INK = "#010202"
GRID = "#E6E6E6"
TABICL = "#F28F1E"
BLUE = "#4C65AF"
GREEN = "#41B36C"
RED = "#F26758"
GREY = "#919191"

REG_TARGETS = [
    ("LOI", "LOI", "LOI", "Flame_retardant_AdditionAmount(wt%)", "FR loading (wt%)", ""),
    ("Tg", "Tg", "Tg", "EEW", "EEW", "degC"),
    ("TENSILE", "Tensile", "Tensile", "Q_thermal", "Q_thermal", "MPa"),
]
ALL_TARGETS = [
    ("LOI", "LOI", "regression"),
    ("Tg", "Tg", "regression"),
    ("TENSILE", "Tensile", "regression"),
    ("UL94", "UL-94", "classification"),
]
TABICL_REG = "DescriptorProcess_TabICLRegressor"
TABICL_CLS = "DescriptorProcess_TabICLClassifier"
FINAL_DIR = {
    "LOI": ROOT / "LOI" / "Result_Final_TabICL_Interpret" / "saved_data",
    "Tg": ROOT / "Tg" / "Result_Final_TabICL_Interpret" / "saved_data",
    "TENSILE": ROOT / "TENSILE" / "Result_Final_TabICL_Interpret" / "saved_data",
    "UL94": ROOT / "94" / "Result_Final_TabICL_Interpret_Cls" / "saved_data",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 1.25,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )


def add_full_box(ax, lw: float = 1.25) -> None:
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(lw)
        ax.spines[side].set_color(INK)


def bold_axes(ax, tick: float = 11, label: float = 13) -> None:
    ax.tick_params(axis="both", width=1.25, labelsize=tick)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")
    ax.xaxis.label.set_fontsize(label)
    ax.yaxis.label.set_fontsize(label)
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")


def savefig(fig, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def load_oof(target: str, model: str) -> pd.DataFrame:
    df = pd.read_csv(OUT / f"unified_{target}_oof_predictions.csv")
    df = df[df["Model"].eq(model)].copy()
    df["Residual"] = df["Predicted"] - df["Actual"]
    df["AbsError"] = df["Residual"].abs()
    return df


def binned_smooth(x: np.ndarray, y: np.ndarray, n_bins: int = 24) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    bins = np.array_split(np.arange(len(xs)), min(n_bins, len(xs)))
    return np.array([np.nanmedian(xs[b]) for b in bins if len(b)]), np.array([np.nanmedian(ys[b]) for b in bins if len(b)])


def plot_full_residual(target: str, display: str, feat: str, xlabel: str) -> None:
    pred = load_oof(target, TABICL_REG)
    df_raw, _, _ = load_data(target)
    merged = pred.merge(df_raw[[feat]].reset_index().rename(columns={"index": "Index"}), on="Index", how="left")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    configs = [
        ("Actual", "Experimental value"),
        ("Predicted", "Predicted value"),
        (feat, xlabel),
    ]
    for ax, (col, lab) in zip(axes[:3], configs):
        x = merged[col].astype(float).values
        y = merged["Residual"].astype(float).values
        err = merged["AbsError"].astype(float).values
        sc = ax.scatter(x, y, c=err, cmap="YlOrRd", s=18, alpha=0.72, edgecolor=INK, linewidth=0.12)
        ax.axhline(0, color=GREY, linestyle="--", linewidth=1.25)
        bx, by = binned_smooth(x, y)
        ax.plot(bx, by, color=INK, linewidth=1.8)
        ax.set_xlabel(lab)
        ax.set_ylabel("Residual" if ax is axes[0] else "")
        ax.grid(color=GRID, linestyle="--", linewidth=0.8)
        add_full_box(ax)
        bold_axes(ax)
    axes[3].hist(merged["Residual"].dropna(), bins=32, color=TABICL, edgecolor=INK, alpha=0.82)
    axes[3].axvline(0, color=GREY, linestyle="--", linewidth=1.25)
    axes[3].set_xlabel("Residual")
    axes[3].set_ylabel("Count")
    axes[3].grid(axis="y", color=GRID, linestyle="--", linewidth=0.8)
    add_full_box(axes[3])
    bold_axes(axes[3])
    fig.suptitle(f"{display}: full residual diagnostics", fontsize=18, fontweight="bold", y=1.04)
    savefig(fig, f"Residual_full_{display}")


def plot_single_error_cdf(target: str, display: str, unit: str) -> None:
    df = load_oof(target, TABICL_REG)
    errs = np.sort(df["AbsError"].values)
    cdf = np.arange(1, len(errs) + 1) / len(errs)
    q80 = np.quantile(errs, 0.80)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot(errs, cdf, color=TABICL, linewidth=2.4)
    ax.axhline(0.80, color=GREY, linestyle="--", linewidth=1.2)
    ax.axvline(q80, color=RED, linestyle="--", linewidth=1.2)
    ax.scatter([q80], [0.80], s=70, color=TABICL, edgecolor=INK, zorder=4)
    ax.text(q80, 0.06, f"80% = {q80:.2f} {unit}".strip(), rotation=90, fontsize=11, fontweight="bold", va="bottom", ha="right")
    ax.set_title(f"{display}: absolute-error CDF", fontsize=16, fontweight="bold", loc="left")
    ax.set_xlabel("Absolute error")
    ax.set_ylabel("Cumulative fraction")
    ax.set_ylim(0, 1.02)
    ax.grid(color=GRID, linestyle="--", linewidth=0.8)
    add_full_box(ax)
    bold_axes(ax)
    savefig(fig, f"Error_CDF_{display}")


def plot_roc() -> None:
    df = load_oof("UL94", TABICL_CLS)
    y = df["Actual"].astype(int).values
    prob = df["Score"].astype(float).clip(0, 1).values
    fpr, tpr, _ = roc_curve(y, prob)
    auc_value = roc_auc_score(y, prob)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.plot(fpr, tpr, color=TABICL, linewidth=2.4, label=f"TabICL AUC = {auc_value:.3f}")
    ax.plot([0, 1], [0, 1], color=GREY, linestyle="--", linewidth=1.2)
    ax.set_title("UL-94 ROC curve", fontsize=16, fontweight="bold", loc="left")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(color=GRID, linestyle="--", linewidth=0.8)
    ax.legend(frameon=False, fontsize=11, loc="lower right")
    add_full_box(ax)
    bold_axes(ax)
    savefig(fig, "ROC_UL94_TabICL")


def ad_embedding(target: str, model: str, task: str) -> pd.DataFrame:
    df_raw, _, feature_cols = load_data(target)
    X, _ = descriptor_matrix(df_raw, feature_cols, target, quick=False)
    X = SimpleImputer(strategy="median").fit_transform(X)
    Xs = StandardScaler().fit_transform(X)
    pred = load_oof(target, model)
    error_col = "AbsError"
    if task == "classification":
        pred[error_col] = (pred["Actual"].astype(float) - pred["Score"].astype(float)).abs()
    lookup = pred[["Index", "Fold", error_col]].drop_duplicates("Index").set_index("Index")
    ad_distance = np.full(len(df_raw), np.nan)
    for fold in sorted(pred["Fold"].unique()):
        test_idx = pred[pred["Fold"].eq(fold)]["Index"].astype(int).values
        train_idx = np.setdiff1d(np.arange(len(df_raw)), test_idx)
        nn = NearestNeighbors(n_neighbors=min(5, len(train_idx))).fit(Xs[train_idx])
        dist, _ = nn.kneighbors(Xs[test_idx])
        ad_distance[test_idx] = dist.mean(axis=1)
    train_nn = NearestNeighbors(n_neighbors=6).fit(Xs)
    dist_all, _ = train_nn.kneighbors(Xs)
    boundary = np.quantile(dist_all[:, 1:].mean(axis=1), 0.95)
    try:
        emb = TSNE(n_components=2, perplexity=35, init="pca", learning_rate="auto", random_state=42).fit_transform(Xs)
    except Exception:
        emb = PCA(n_components=2, random_state=42).fit_transform(Xs)
    out = pd.DataFrame(
        {
            "Dim1": emb[:, 0],
            "Dim2": emb[:, 1],
            "AD_Distance": ad_distance,
            "Inside_AD": ad_distance <= boundary,
            "Error": lookup.reindex(np.arange(len(df_raw)))[error_col].values,
            "Boundary": boundary,
        }
    )
    return out


def plot_ad_map(target: str, display: str, task: str) -> None:
    model = TABICL_CLS if task == "classification" else TABICL_REG
    ad = ad_embedding(target, model, task)
    outside = ad[~ad["Inside_AD"]]
    inside = ad[ad["Inside_AD"]]
    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    ax.scatter(ad["Dim1"], ad["Dim2"], s=9, color="#D8D8D8", alpha=0.52, edgecolor="none", label="descriptor-process manifold")
    ax.scatter(inside["Dim1"], inside["Dim2"], c=inside["Error"], cmap="YlOrRd", s=22, alpha=0.82, edgecolor=INK, linewidth=0.08, label="inside AD")
    ax.scatter(outside["Dim1"], outside["Dim2"], c=outside["Error"], cmap="YlOrRd", s=42, alpha=0.96, edgecolor=RED, linewidth=0.7, marker="^", label="outside AD")
    ax.set_title(f"{display}: applicability-domain map", fontsize=16, fontweight="bold", loc="left")
    ax.set_xlabel("2D descriptor-process embedding")
    ax.set_ylabel("2D descriptor-process embedding")
    ax.grid(color=GRID, linestyle="--", linewidth=0.8)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    add_full_box(ax)
    bold_axes(ax)
    savefig(fig, f"AD_map_{display}")
    ad.to_csv(FIG_DIR / f"AD_map_{display}_data.csv", index=False, encoding="utf-8-sig")


def threshold_values() -> dict[str, float]:
    df = load_oof("UL94", TABICL_CLS)
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
    fpr, fnr = np.array(fpr), np.array(fnr)
    safe_candidates = thresholds[fpr <= 0.10]
    return {
        "balanced": float(thresholds[np.argmin(np.abs(fpr - fnr))]),
        "default_0p50": 0.50,
        "safety_screening": float(safe_candidates[0] if len(safe_candidates) else thresholds[np.argmin(fpr)]),
    }


def plot_confusion_matrices() -> None:
    df = load_oof("UL94", TABICL_CLS)
    y = df["Actual"].astype(int).values
    prob = df["Score"].astype(float).clip(0, 1).values
    thresholds = threshold_values()
    fig, axes = plt.subplots(1, len(thresholds), figsize=(12.6, 3.8))
    rows = []
    for ax, (name, t) in zip(axes, thresholds.items()):
        pred = (prob >= t).astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        rows.append({"Threshold_Name": name, "Threshold": t, "TN": cm[0, 0], "FP": cm[0, 1], "FN": cm[1, 0], "TP": cm[1, 1]})
        im = ax.imshow(cm, cmap="Oranges", vmin=0)
        ax.set_title(f"{name}\nt = {t:.2f}", fontsize=13, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["True 0", "True 1"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16, fontweight="bold", color=INK)
        add_full_box(ax)
        bold_axes(ax, tick=9, label=10)
        fig_single, ax_single = plt.subplots(figsize=(4.2, 3.8))
        ax_single.imshow(cm, cmap="Oranges", vmin=0)
        ax_single.set_title(f"UL-94 confusion matrix: {name}\nt = {t:.2f}", fontsize=13.5, fontweight="bold")
        ax_single.set_xticks([0, 1])
        ax_single.set_yticks([0, 1])
        ax_single.set_xticklabels(["Pred 0", "Pred 1"])
        ax_single.set_yticklabels(["True 0", "True 1"])
        for i in range(2):
            for j in range(2):
                ax_single.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=17, fontweight="bold", color=INK)
        add_full_box(ax_single)
        bold_axes(ax_single, tick=9, label=10)
        savefig(fig_single, f"Confusion_UL94_{name}")
    fig.suptitle("UL-94 confusion matrices at selected thresholds", fontsize=16, fontweight="bold", y=1.05)
    savefig(fig, "Confusion_UL94_selected_thresholds")
    pd.DataFrame(rows).to_csv(FIG_DIR / "Confusion_UL94_selected_thresholds_values.csv", index=False, encoding="utf-8-sig")


def plot_waterfall_from_shap(target: str, display: str) -> None:
    base = FINAL_DIR[target]
    required = [base / "X_train.csv", base / "X_test.csv", base / "shap_values.csv", base / "predictions.csv"]
    if not all(p.exists() for p in required):
        return
    X_train = pd.read_csv(base / "X_train.csv")
    X_test = pd.read_csv(base / "X_test.csv")
    shap_values = pd.read_csv(base / "shap_values.csv")
    pred = pd.read_csv(base / "predictions.csv")
    common = [c for c in X_test.columns if c in X_train.columns]
    imputer = SimpleImputer(strategy="median").fit(X_train[common].replace([np.inf, -np.inf], np.nan))
    train = imputer.transform(X_train[common].replace([np.inf, -np.inf], np.nan))
    test = imputer.transform(X_test[common].replace([np.inf, -np.inf], np.nan))
    scaler = StandardScaler().fit(train)
    train_s = scaler.transform(train)
    test_s = scaler.transform(test)
    nn_train = NearestNeighbors(n_neighbors=6).fit(train_s)
    train_dist, _ = nn_train.kneighbors(train_s)
    boundary = np.quantile(train_dist[:, 1:].mean(axis=1), 0.95)
    nn = NearestNeighbors(n_neighbors=min(5, len(train_s))).fit(train_s)
    dist, _ = nn.kneighbors(test_s)
    ad_dist = dist.mean(axis=1)
    outside = np.where(ad_dist > boundary)[0]
    if len(outside) == 0:
        outside = np.argsort(ad_dist)[-3:]
    out_dir = FIG_DIR / "local_SHAP_waterfall_extrapolation"
    out_dir.mkdir(exist_ok=True)
    rows = []
    for idx in outside:
        sv = shap_values.iloc[idx].astype(float)
        top = sv.reindex(sv.abs().sort_values(ascending=False).index).head(14)
        if "Predicted" in pred.columns:
            pred_value = float(pred.iloc[idx]["Predicted"])
        elif "Probability" in pred.columns:
            pred_value = float(pred.iloc[idx]["Probability"])
        else:
            pred_value = float(np.nansum(sv.values))
        base_value = pred_value - float(sv.sum())
        colors = [RED if v > 0 else BLUE for v in top.values]
        fig, ax = plt.subplots(figsize=(7.6, 5.2))
        y = np.arange(len(top))[::-1]
        ax.barh(y, top.values, color=colors, edgecolor=INK, linewidth=0.45)
        ax.axvline(0, color=INK, linewidth=1.0)
        ax.set_yticks(y)
        ax.set_yticklabels(top.index, fontsize=8.2)
        ax.set_xlabel("SHAP contribution")
        ax.set_title(f"{display}: local SHAP waterfall, extrapolation sample {idx}", fontsize=13.5, fontweight="bold", loc="left")
        ax.text(
            0.98,
            0.04,
            f"AD distance = {ad_dist[idx]:.2f}\nAD boundary = {boundary:.2f}\nbase = {base_value:.2f}\npred = {pred_value:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor=INK, boxstyle="round,pad=0.25", linewidth=0.8),
        )
        ax.grid(axis="x", color=GRID, linestyle="--", linewidth=0.8)
        add_full_box(ax)
        bold_axes(ax, tick=8.2, label=11)
        stem = f"Waterfall_extrapolation_{display}_sample_{idx:03d}"
        fig.savefig(out_dir / f"{stem}.png", dpi=600, bbox_inches="tight")
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)
        rows.append({"Target": display, "Sample_Index_in_X_test": idx, "AD_Distance": ad_dist[idx], "AD_Boundary": boundary, "Predicted": pred_value})
    pd.DataFrame(rows).to_csv(out_dir / f"Waterfall_extrapolation_{display}_manifest.csv", index=False, encoding="utf-8-sig")


def copy_existing_figures() -> None:
    for name in ["Figure2_unified_benchmark", "Figure3_reliability_diagnostics"]:
        for ext in [".png", ".pdf"]:
            src = OUT / f"{name}{ext}"
            if src.exists():
                shutil.copy2(src, FIG_DIR / f"{name}{ext}")
    ul94_src = ROOT / "94" / "Result_Final_TabICL_Interpret_Cls" / "png"
    if ul94_src.exists():
        dst = FIG_DIR / "local_SHAP_waterfall_extrapolation" / "UL94_existing_waterfalls"
        dst.mkdir(parents=True, exist_ok=True)
        for src in ul94_src.glob("Fig_Waterfall_*.png"):
            shutil.copy2(src, dst / src.name)


def main() -> None:
    configure_style()
    copy_existing_figures()
    for target, display, _, feat, xlabel, unit in REG_TARGETS:
        plot_full_residual(target, display, feat, xlabel)
        plot_single_error_cdf(target, display, unit)
        plot_waterfall_from_shap(target, display)
    plot_roc()
    plot_confusion_matrices()
    for target, display, task in ALL_TARGETS:
        plot_ad_map(target, display, task)
    print(f"Individual diagnostic figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
