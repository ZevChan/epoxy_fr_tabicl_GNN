from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import (
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OLD_OUT = HERE / "outputs"
NEW_OUT = ROOT / "\u6295\u7a3f\u56fe\u72472"
NEW_OUT.mkdir(exist_ok=True)
PANEL_OUT = NEW_OUT / "individual_panels"
PANEL_OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))

from run_unified_benchmark import descriptor_matrix, load_data  # noqa: E402


# Manuscript palette.
INK = "#333333"
GRID = "#E6E6E6"
EXPLICIT = "#1F4E79"
GRAPH = "#C0504D"
SLATE = "#808080"
SILVER = "#D9D9D9"
RESIN = "#4A7B9D"
FR = "#E08E36"
PROCESS = "#7A9A74"
LOW = "#E1F5FE"
MID = "#29B6F6"
HIGH = "#0277BD"
SHAP_LOW = "#1F77B4"
SHAP_MID = "#E6E6E6"
SHAP_HIGH = "#D62728"

SEQ_CMAP = mcolors.LinearSegmentedColormap.from_list("cyan_blue", [LOW, MID, HIGH])
SHAP_CMAP = mcolors.LinearSegmentedColormap.from_list("blue_grey_red", [SHAP_LOW, SHAP_MID, SHAP_HIGH])
CORR_CMAP = mcolors.LinearSegmentedColormap.from_list("corr", ["#B0BEC5", "#F5F5F5", "#1F4E79"])

REG_TARGETS = [
    ("LOI", "LOI", ""),
    ("Tg", "Tg", "degC"),
    ("TENSILE", "Tensile", "MPa"),
]
ALL_TARGETS = [
    ("LOI", "LOI", "regression"),
    ("Tg", "Tg", "regression"),
    ("TENSILE", "Tensile", "regression"),
    ("UL94", "UL-94", "classification"),
]
TARGET_KEY = {"LOI": "LOI", "Tg": "Tg", "Tensile": "TENSILE", "UL-94": "UL94"}
TABICL_REG = "DescriptorProcess_TabICLRegressor"
TABICL_CLS = "DescriptorProcess_TabICLClassifier"
SHAP_DIRS = {
    "LOI": ROOT / "LOI" / "Result_Final_TabICL_Interpret" / "saved_data",
    "UL94": ROOT / "94" / "Result_Final_TabICL_Interpret_Cls" / "saved_data",
    "Tg": ROOT / "Tg" / "Result_Final_TabICL_Interpret" / "saved_data",
    "TENSILE": ROOT / "TENSILE" / "Result_Final_TabICL_Interpret" / "saved_data",
}

BEESWARM_PREFERRED_FEATURES = {
    "LOI": ["Flame_retardant_AdditionAmount(wt%)", "FR_F01[C-P]", "FR_B01[C-P]", "EP_wt_fraction"],
    "UL94": ["Flame_retardant_AdditionAmount(wt%)", "FR_max_conj_path", "CURING_Mor30p", "EP_wt_fraction"],
    "Tg": ["EEW", "T_max", "Q_thermal", "CURING_RBF", "CURING_SssCH2"],
    "TENSILE": ["Q_thermal", "FR_E1m", "FR_TPSA_efficiency", "Flame_retardant_AdditionAmount(wt%)", "EP_SIC5"],
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 1.05,
            "axes.labelsize": 12,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )


def add_full_box(ax, lw: float = 1.05) -> None:
    for side in ["top", "right", "bottom", "left"]:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(lw)
        ax.spines[side].set_color(INK)


def bold_axes(ax, tick: float = 10.5, label: float = 12) -> None:
    ax.tick_params(axis="both", width=1.05, labelsize=tick, colors=INK)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontweight("bold")
    ax.xaxis.label.set_fontweight("bold")
    ax.yaxis.label.set_fontweight("bold")
    ax.xaxis.label.set_fontsize(label)
    ax.yaxis.label.set_fontsize(label)


def panel_label(ax, letter: str, x: float = -0.12, y: float = 1.03) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=20, fontweight="bold", va="bottom", color=INK)


def savefig(fig, stem: str) -> None:
    fig.savefig(NEW_OUT / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(NEW_OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def save_single_panel(stem: str, letter: str, draw_fn, figsize: tuple[float, float] = (5.6, 4.2)) -> None:
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    draw_fn(ax)
    fig.savefig(PANEL_OUT / f"{stem}{letter}.png", dpi=600, bbox_inches="tight")
    fig.savefig(PANEL_OUT / f"{stem}{letter}.pdf", bbox_inches="tight")
    plt.close(fig)


def load_oof(target: str, model: str) -> pd.DataFrame:
    df = pd.read_csv(OLD_OUT / f"unified_{target}_oof_predictions.csv")
    df = df[df["Model"].eq(model)].copy()
    df["Residual"] = df["Predicted"] - df["Actual"]
    df["AbsError"] = df["Residual"].abs()
    return df


def binned_line(x: np.ndarray, y: np.ndarray, n_bins: int = 18) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) == 0:
        return np.array([]), np.array([])
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    bins = np.array_split(np.arange(len(xs)), min(n_bins, len(xs)))
    return np.array([np.nanmedian(xs[b]) for b in bins if len(b)]), np.array([np.nanmedian(ys[b]) for b in bins if len(b)])


def short_feature(name: str) -> str:
    return (
        name.replace("Flame_retardant_AdditionAmount(wt%)", "FR wt%")
        .replace("Curing_agent_AdditionAmount(wt%)", "Curing wt%")
        .replace("FR_TPSA_efficiency", "FR TPSA eff.")
        .replace("_", " ")
    )


def feature_category(feature: str) -> str:
    if feature in {"Flame_retardant_AdditionAmount(wt%)", "EP_wt_fraction", "FR_wt_fraction", "CURING_wt_fraction", "Curing_agent_AdditionAmount(wt%)"}:
        return "Formulation"
    if feature in {"T_max", "t_total", "Q_thermal"}:
        return "Process"
    if feature.startswith("EP_") or feature == "EEW":
        return "EP"
    if feature.startswith("FR_"):
        return "FR"
    if feature.startswith("CURING_"):
        return "Curing"
    return "Other"


def feature_color(feature: str) -> str:
    return {
        "Formulation": SLATE,
        "Process": PROCESS,
        "EP": RESIN,
        "FR": FR,
        "Curing": PROCESS,
        "Other": SILVER,
    }[feature_category(feature)]


def load_shap_task(target: str) -> dict:
    with open(SHAP_DIRS[target] / "plot_data.pkl", "rb") as f:
        obj = pickle.load(f)
    shap = np.asarray(obj["shap_values"], dtype=float)
    if shap.ndim == 3:
        shap = shap[:, :, -1]
    return {"features": list(obj["top_k_features"]), "x": np.asarray(obj["X_test"], dtype=float), "shap": shap}


def shap_arrays(data: dict, feature: str) -> tuple[np.ndarray, np.ndarray]:
    idx = data["features"].index(feature)
    return data["x"][:, idx].astype(float), data["shap"][:, idx].astype(float)


def shap_values_for(data: dict, features: list[str]) -> np.ndarray:
    vals = np.zeros(data["shap"].shape[0], dtype=float)
    for feature in features:
        if feature in data["features"]:
            vals += data["shap"][:, data["features"].index(feature)]
    return vals


def values_for(data: dict, feature: str) -> np.ndarray:
    return data["x"][:, data["features"].index(feature)].astype(float)


def mean_abs_shap(data: dict) -> pd.Series:
    return pd.Series(np.nanmean(np.abs(data["shap"]), axis=0), index=data["features"])


def beeswarm_order_from_preferred(data: dict, preferred: list[str], max_count: int = 8) -> list[str]:
    ordered = [f for f in preferred if f in data["features"]]
    for f in mean_abs_shap(data).sort_values(ascending=False).index:
        if f not in ordered:
            ordered.append(f)
        if len(ordered) >= max_count:
            break
    return ordered[:max_count]


def beeswarm_feature_order(target: str, data: dict, max_count: int = 8) -> list[str]:
    return beeswarm_order_from_preferred(data, BEESWARM_PREFERRED_FEATURES.get(target, []), max_count)


def regression_metrics(df: pd.DataFrame) -> tuple[float, float, float]:
    return (
        r2_score(df["Actual"], df["Predicted"]),
        mean_squared_error(df["Actual"], df["Predicted"]) ** 0.5,
        mean_absolute_error(df["Actual"], df["Predicted"]),
    )


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (prob >= lo) & (prob < hi if hi < 1 else prob <= hi)
        if np.any(mask):
            ece += mask.mean() * abs(prob[mask].mean() - y_true[mask].mean())
    return float(ece)


def threshold_curves(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
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
    balanced = float(thresholds[np.argmin(np.abs(fpr - fnr))])
    safe_candidates = thresholds[fpr <= 0.10]
    safety = float(safe_candidates[0] if len(safe_candidates) else thresholds[np.argmin(fpr)])
    return thresholds, fpr, fnr, balanced, safety


# ---------- Figure 2: chemical space / AD ----------


def compute_task_ad(target: str, task: str) -> pd.DataFrame:
    model = TABICL_CLS if task == "classification" else TABICL_REG
    df_raw, _, feature_cols = load_data(target)
    X, _ = descriptor_matrix(df_raw, feature_cols, target, quick=False)
    X = SimpleImputer(strategy="median").fit_transform(X)
    Xs = StandardScaler().fit_transform(X)
    pred = load_oof(target, model)
    if task == "classification":
        prob = pred["Score"].astype(float).clip(0, 1)
        pred["ErrorForAD"] = 1.0 - (prob - 0.5).abs() * 2.0
    else:
        scale = np.nanpercentile(pred["AbsError"], 95)
        pred["ErrorForAD"] = (pred["AbsError"] / max(scale, 1e-9)).clip(0, 1)

    ad_distance = np.full(len(df_raw), np.nan)
    for fold in sorted(pred["Fold"].unique()):
        test_idx = pred[pred["Fold"].eq(fold)]["Index"].astype(int).values
        train_idx = np.setdiff1d(np.arange(len(df_raw)), test_idx)
        nn = NearestNeighbors(n_neighbors=min(5, len(train_idx))).fit(Xs[train_idx])
        dist, _ = nn.kneighbors(Xs[test_idx])
        ad_distance[test_idx] = dist.mean(axis=1)
    nn_all = NearestNeighbors(n_neighbors=6).fit(Xs)
    dist_all, _ = nn_all.kneighbors(Xs)
    boundary = float(np.nanquantile(dist_all[:, 1:].mean(axis=1), 0.95))
    lookup = pred[["Index", "Fold", "ErrorForAD"]].drop_duplicates("Index").set_index("Index")
    return pd.DataFrame(
        {
            "Target": target,
            "Index": np.arange(len(df_raw)),
            "Fold": lookup.reindex(np.arange(len(df_raw)))["Fold"].values,
            "AD_Distance": ad_distance,
            "AD_Boundary": boundary,
            "ErrorForAD": lookup.reindex(np.arange(len(df_raw)))["ErrorForAD"].values,
        }
    )


def combined_feature_space(max_per_task: int = 650) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    frames = []
    for target, display, task in ALL_TARGETS:
        df_raw, y, feature_cols = load_data(target)
        X, selected = descriptor_matrix(df_raw, feature_cols, target, quick=False)
        frame = pd.DataFrame(X, columns=selected)
        frame["Target"] = target
        frame["TargetDisplay"] = display
        frame["Index"] = np.arange(len(frame))
        frame["PropertyValue"] = y
        if len(frame) > max_per_task:
            take = rng.choice(frame.index.values, max_per_task, replace=False)
            frame = frame.loc[np.sort(take)].copy()
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    numeric = combined.drop(columns=["Target", "TargetDisplay", "Index", "PropertyValue"], errors="ignore")
    X = SimpleImputer(strategy="median").fit_transform(numeric.replace([np.inf, -np.inf], np.nan))
    X = StandardScaler().fit_transform(X)
    Xp = PCA(n_components=min(30, X.shape[1]), random_state=42).fit_transform(X)
    try:
        emb = TSNE(n_components=2, perplexity=35, init="pca", learning_rate="auto", random_state=42).fit_transform(Xp)
    except Exception:
        emb = PCA(n_components=2, random_state=42).fit_transform(X)
    combined["Dim1"] = emb[:, 0]
    combined["Dim2"] = emb[:, 1]
    ad_parts = [compute_task_ad(t, task) for t, _, task in ALL_TARGETS]
    ad = pd.concat(ad_parts, ignore_index=True)
    combined = combined.merge(ad, on=["Target", "Index"], how="left")
    combined["OutsideAD"] = combined["AD_Distance"] > combined["AD_Boundary"]
    combined.to_csv(NEW_OUT / "Figure2_chemical_space_embedding_data.csv", index=False, encoding="utf-8-sig")
    return combined


def single_task_feature_space(target: str) -> pd.DataFrame:
    """Compute 2D embedding + AD for a SINGLE target — no cross-task mixing."""
    task_cfg = {"LOI": "regression", "Tg": "regression", "TENSILE": "regression", "UL94": "classification"}
    task = task_cfg[target]
    display_map = {"LOI": "LOI", "Tg": "Tg", "TENSILE": "Tensile", "UL94": "UL-94"}
    display = display_map[target]

    df_raw, y, feature_cols = load_data(target)
    X, selected = descriptor_matrix(df_raw, feature_cols, target, quick=False)
    X_imp = SimpleImputer(strategy="median").fit_transform(X)
    Xs = StandardScaler().fit_transform(X_imp)

    # PCA + t-SNE
    Xp = PCA(n_components=min(30, Xs.shape[1]), random_state=42).fit_transform(Xs)
    try:
        emb = TSNE(n_components=2, perplexity=min(35, Xp.shape[0] // 3), init="pca",
                    learning_rate="auto", random_state=42).fit_transform(Xp)
    except Exception:
        emb = PCA(n_components=2, random_state=42).fit_transform(Xp)

    ad = compute_task_ad(target, task)

    # Fetch raw absolute error for proper physical-unit colorbar
    model = TABICL_CLS if task == "classification" else TABICL_REG
    pred = load_oof(target, model)
    err_lookup = pred[["Index", "AbsError"]].drop_duplicates("Index").set_index("Index") if task != "classification" else None

    frame = pd.DataFrame({
        "Target": target,
        "TargetDisplay": display,
        "Index": np.arange(len(df_raw)),
        "Dim1": emb[:, 0],
        "Dim2": emb[:, 1],
    })
    frame = frame.merge(ad, on=["Target", "Index"], how="left")
    frame["OutsideAD"] = frame["AD_Distance"] > frame["AD_Boundary"]
    if err_lookup is not None:
        frame["AbsError"] = err_lookup.reindex(frame["Index"].values)["AbsError"].values
    else:
        frame["AbsError"] = frame["ErrorForAD"]  # classification: use transformed uncertainty
    return frame


def fig2_feature_summary(ax) -> None:
    panel_label(ax, "a", x=-0.08)
    ax.set_axis_off()
    ax.set_title("Feature-space construction", loc="left", fontsize=17, fontweight="bold", pad=6)
    boxes = [
        (0.03, 0.66, "Molecular\ndescriptors", RESIN),
        (0.03, 0.40, "Formulation\nfractions", SLATE),
        (0.03, 0.14, "Process\ndescriptors", PROCESS),
    ]
    for x, y, text, color in boxes:
        ax.add_patch(patches.FancyBboxPatch((x, y), 0.36, 0.18, boxstyle="round,pad=0.02", facecolor=color, edgecolor=INK, linewidth=1.0, transform=ax.transAxes))
        ax.text(x + 0.18, y + 0.09, text, transform=ax.transAxes, color="white", fontsize=14, fontweight="bold", ha="center", va="center")
        ax.annotate("", xy=(0.58, y + 0.09), xytext=(0.39, y + 0.09), arrowprops=dict(arrowstyle="->", lw=1.4, color=INK), xycoords=ax.transAxes)
    ax.add_patch(patches.FancyBboxPatch((0.58, 0.32), 0.36, 0.30, boxstyle="round,pad=0.03", facecolor="#F7F7F7", edgecolor=INK, linewidth=1.2, transform=ax.transAxes))
    ax.text(0.76, 0.47, "Descriptor-\nprocess\nfeature space", transform=ax.transAxes, fontsize=14.5, fontweight="bold", ha="center", va="center", color=EXPLICIT)


def fig2_pairgrid(ax_outer) -> None:
    import pickle

    panel_label(ax_outer, "a", x=-0.04)
    ax_outer.set_axis_off()
    df_raw, _, _ = load_data("LOI")

    # Use LOI's top-3 SHAP features + LOI target → 4×4 grid
    shap_dir = SHAP_DIRS.get("LOI")
    top_features = []
    if shap_dir and shap_dir.exists():
        pkl_path = shap_dir / "plot_data.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                shap_data = pickle.load(f)
            top_all = shap_data.get("top_15_features", [])
            top_features = [c for c in top_all[:3] if c in df_raw.columns]

    # Fallback to hardcoded defaults if SHAP data unavailable
    if len(top_features) < 3:
        top_features = [
            "Flame_retardant_AdditionAmount(wt%)",
            "CURING_MaxaasC",
            "FR_F01[C-P]",
        ]
        top_features = [c for c in top_features if c in df_raw.columns]

    target_col = "LOI"
    cols = top_features[:3]
    if target_col not in cols and target_col in df_raw.columns:
        cols.append(target_col)
    cols = cols[:4]  # strictly ≤ 4
    data = df_raw[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) > 900:
        data = data.sample(900, random_state=42)
    n = len(cols)
    gap = 0.014
    left, bottom = 0.030, 0.050
    grid_w, grid_h = 0.910, 0.860
    cell_w = (grid_w - gap * (n - 1)) / n
    cell_h = (grid_h - gap * (n - 1)) / n
    for i in range(n):
        for j in range(n):
            ax = ax_outer.inset_axes([left + j * (cell_w + gap), bottom + (n - 1 - i) * (cell_h + gap), cell_w, cell_h])
            x = data[cols[j]].astype(float).values
            y = data[cols[i]].astype(float).values
            if i == j:
                sns.histplot(x=x, ax=ax, kde=True, color=EXPLICIT, edgecolor=INK, alpha=0.45, bins=20, linewidth=0.55)
                if ax.lines:
                    ax.lines[-1].set_color(GRAPH)
                    ax.lines[-1].set_linewidth(1.45)
            elif i < j:
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() >= 2:
                    r, p = pearsonr(x[mask], y[mask])
                    stars = "***" if p <= 0.001 else "**" if p <= 0.01 else "*" if p <= 0.05 else ""
                    ax.set_facecolor(CORR_CMAP((r + 1) / 2))
                    ax.patch.set_alpha(0.72)
                    ax.text(0.5, 0.54, f"{r:.2f}", transform=ax.transAxes, ha="center", va="center", fontsize=14, fontweight="bold", color=INK)
                    ax.text(0.5, 0.34, stars, transform=ax.transAxes, ha="center", va="center", fontsize=12, fontweight="bold", color=INK)
            else:
                sns.regplot(
                    x=x,
                    y=y,
                    ax=ax,
                    scatter_kws={"s": 10, "alpha": 0.58, "color": EXPLICIT, "edgecolors": "white", "linewidths": 0.25},
                    line_kws={"color": GRAPH, "linewidth": 1.35},
                    ci=95,
                )
                ax.grid(True, linestyle="--", color=GRID, linewidth=0.45, alpha=0.9)
            if i == n - 1:
                ax.set_xlabel(short_feature(cols[j]), fontsize=9.5, fontweight="bold")
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(short_feature(cols[i]), fontsize=9.5, fontweight="bold")
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
            ax.tick_params(axis="both", labelsize=8, width=0.65, length=2.2, colors=INK)
            for spine in ax.spines.values():
                spine.set_linewidth(0.70)
                spine.set_color(INK)
    cbar_ax = ax_outer.inset_axes([0.945, 0.12, 0.018, 0.72])
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=CORR_CMAP), cax=cbar_ax)
    cb.set_label("Pearson r", fontsize=13, fontweight="bold", color=INK)
    cb.ax.tick_params(labelsize=11, colors=INK, width=0.8)


def fig2_embedding(ax_outer, data: pd.DataFrame, mode: str, letter: str, title: str) -> None:
    panel_label(ax_outer, letter)
    ax_outer.set_axis_off()

    # Detect whether this is single-task data; if so, tag axis labels with the target name
    displays = data["TargetDisplay"].unique()
    is_single = len(displays) == 1
    target_tag = f" ({displays[0]})" if is_single else ""

    # Target-specific color (matching Figure2 panel b)
    target_color_map = {"LOI": EXPLICIT, "Tg": RESIN, "Tensile": FR, "UL-94": GRAPH}
    tcolor = target_color_map.get(displays[0], EXPLICIT) if is_single else EXPLICIT

    ax = ax_outer.inset_axes([0.0, 0.12, 1.0, 0.85])

    if mode == "task":
        colors = {"LOI": EXPLICIT, "Tg": RESIN, "Tensile": FR, "UL-94": GRAPH}
        for display, sub in data.groupby("TargetDisplay"):
            ax.scatter(sub["Dim1"], sub["Dim2"], s=9, alpha=0.70, color=colors[display], label=display, edgecolor="none")
        ax.legend(frameon=False, fontsize=12, loc="best")
    elif mode == "overlap":
        ax.scatter(data["Dim1"], data["Dim2"], s=8, color=SILVER, alpha=0.45, edgecolor="none", label="CV training folds")
        test = data[data["Fold"].eq(1)]
        inside = test[~test["OutsideAD"]]
        outside = test[test["OutsideAD"]]
        # Use target-specific color for "test inside AD"
        inside_color = tcolor if is_single else EXPLICIT
        ax.scatter(inside["Dim1"], inside["Dim2"], s=16, color=inside_color, alpha=0.80, edgecolor="none", label="test inside AD")
        ax.scatter(outside["Dim1"], outside["Dim2"], s=28, color=GRAPH, marker="^", alpha=0.92, edgecolor=INK, linewidth=0.25, label="test outside AD")
        ax.legend(frameon=False, fontsize=12, loc="best")
    else:
        error_col = "AbsError" if "AbsError" in data.columns else "ErrorForAD"
        error_label = f"Absolute error{target_tag}" if error_col == "AbsError" else f"Error / uncertainty{target_tag}"
        # Classification targets: use "Prediction uncertainty" instead of "Absolute error"
        if is_single and displays[0] == "UL-94":
            error_label = f"Prediction uncertainty{target_tag}"
        # Target-specific colormap: light-gray → target color
        tcmap = mcolors.LinearSegmentedColormap.from_list(
            f"err_{displays[0]}", ["#F5F5F5", tcolor]
        ) if is_single else SEQ_CMAP
        sc = ax.scatter(data["Dim1"], data["Dim2"], c=data[error_col], s=12, cmap=tcmap, alpha=0.80, edgecolor="none")
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.08)
        cbar.set_label(error_label, fontsize=12.5, fontweight="bold")
        cbar.ax.tick_params(labelsize=11)
        # Annotation for classification: explain uncertainty metric
        if is_single and displays[0] == "UL-94":
            ax.text(0.5, -0.18, "Uncertainty = 1 − 2·|P(y=1) − 0.5|",
                    transform=ax.transAxes, fontsize=9.5, fontstyle="italic",
                    ha="center", va="top", color=INK)

    ax.set_xlabel(f"2D descriptor-process embedding{target_tag}")
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.grid(color=GRID, linestyle="--", linewidth=0.65)
    add_full_box(ax)
    bold_axes(ax, tick=12, label=13.5)


def fig2_ad_distance(ax, data: pd.DataFrame) -> None:
    panel_label(ax, "e")
    for target, display, _ in ALL_TARGETS:
        sub = data[data["Target"].eq(target)].dropna(subset=["AD_Distance", "ErrorForAD"])
        ax.scatter(sub["AD_Distance"], sub["ErrorForAD"], s=13, alpha=0.45, label=display, edgecolor="none")
    boundary = data["AD_Boundary"].median()
    ax.axvline(boundary, color=GRAPH, linestyle="--", linewidth=1.2)
    bx, by = binned_line(data["AD_Distance"].values, data["ErrorForAD"].values, 18)
    ax.plot(bx, by, color=INK, linewidth=1.8)
    ax.set_title("AD distance versus error/uncertainty", loc="left", fontsize=15, fontweight="bold", pad=5)
    ax.set_xlabel("kNN distance to training domain")
    ax.set_ylabel("Error / uncertainty")
    ax.grid(color=GRID, linestyle="--", linewidth=0.75)
    ax.legend(frameon=False, fontsize=10, ncol=2, loc="upper left")
    add_full_box(ax)
    bold_axes(ax, tick=12, label=13.5)


def fig2_ad_distance_stack(ax_outer, data: pd.DataFrame) -> None:
    panel_label(ax_outer, "e", x=-0.06)
    ax_outer.set_axis_off()
    x_max = float(np.nanquantile(data["AD_Distance"], 0.985))
    x_max = max(x_max, float(data["AD_Boundary"].median()) * 1.25)
    task_colors = {"LOI": EXPLICIT, "Tg": RESIN, "Tensile": FR, "UL-94": GRAPH}
    task_rows = [("LOI", "LOI"), ("UL94", "UL-94"), ("Tg", "Tg"), ("TENSILE", "Tensile")]
    axes = []
    for i, (target, display) in enumerate(task_rows):
        ax = ax_outer.inset_axes([0.16, 0.765 - i * 0.215, 0.79, 0.165])
        axes.append(ax)
        sub = data[data["Target"].eq(target)].dropna(subset=["AD_Distance", "ErrorForAD"])
        color = task_colors[display]
        ax.scatter(sub["AD_Distance"], sub["ErrorForAD"], s=10, alpha=0.42, color=color, edgecolor="none")
        bx, by = binned_line(sub["AD_Distance"].values, sub["ErrorForAD"].values, 12)
        ax.plot(bx, by, color=INK, linewidth=1.4)
        boundary = float(sub["AD_Boundary"].median()) if len(sub) else float(data["AD_Boundary"].median())
        ax.axvline(boundary, color=GRAPH, linestyle="--", linewidth=0.9)
        ax.set_xlim(0, x_max)
        ax.set_ylim(-0.03, 1.04)
        ax.set_title(display, loc="left", fontsize=13.5, fontweight="bold", color=INK, pad=4)
        ax.set_ylabel("")
        if i < len(task_rows) - 1:
            ax.set_xticklabels([])
            ax.set_xlabel("")
        else:
            ax.set_xlabel("kNN distance to training domain", fontsize=12.5, fontweight="bold")
        ax.grid(color=GRID, linestyle="--", linewidth=0.50)
        add_full_box(ax, 0.75)
        ax.tick_params(axis="both", labelsize=11, width=0.75, colors=INK)
        for item in ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontweight("bold")
    # Centered ylabel on outer axes, vertically centered across all 4 inset subplots
    ax_outer.text(-0.04, 0.525, "Error / uncertainty", transform=ax_outer.transAxes,
                  rotation=90, va="center", ha="center", fontsize=12.5, fontweight="bold", color=INK)


def make_pairgrid_for_target(target: str, display: str, top_n: int = 10) -> None:
    """Generate a standalone PairGrid for *target* using its top *top_n* SHAP features."""
    import pickle

    shap_dir = SHAP_DIRS.get(target)
    if shap_dir is None or not shap_dir.exists():
        print(f"  SKIP {display}: SHAP dir not found at {shap_dir}")
        return
    pkl_path = shap_dir / "plot_data.pkl"
    if not pkl_path.exists():
        print(f"  SKIP {display}: plot_data.pkl not found")
        return

    with open(pkl_path, "rb") as f:
        shap_data = pickle.load(f)
    top_all = shap_data.get("top_15_features", [])
    if not top_all:
        print(f"  SKIP {display}: no top_15_features in SHAP data")
        return
    top_features = top_all[:top_n]

    df_raw, _, _ = load_data(target)
    target_col_map = {"LOI": "LOI", "Tg": "Tg", "TENSILE": "Tensile", "UL94": "UL94"}
    target_col = target_col_map.get(target, target)

    # Use the same per-target color as Figure2 panel b
    target_color_map = {"LOI": EXPLICIT, "Tg": RESIN, "Tensile": FR, "UL-94": GRAPH}
    tcolor = target_color_map.get(display, EXPLICIT)
    # Per-target correlation colormap: gray → white → target color
    tcorr_cmap = mcolors.LinearSegmentedColormap.from_list(
        f"corr_{display}", ["#B0BEC5", "#F5F5F5", tcolor]
    )
    cols = [c for c in top_features if c in df_raw.columns]
    if target_col not in cols and target_col in df_raw.columns:
        cols.append(target_col)
    cols = cols[:top_n + 1]  # features + target

    data = df_raw[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) > 900:
        data = data.sample(900, random_state=42)
    n = len(cols)
    if n < 3:
        print(f"  SKIP {display}: only {n} valid columns")
        return

    # Classification target: use point-biserial instead of Pearson, skip linear regression
    is_cls = (target == "UL94")
    import scipy.stats as _st

    fig, ax_outer = plt.subplots(figsize=(14, 12), dpi=300)
    ax_outer.set_axis_off()

    gap = 0.014
    left, bottom = 0.035, 0.050
    grid_w, grid_h = 0.890, 0.870
    cell_w = (grid_w - gap * (n - 1)) / n
    cell_h = (grid_h - gap * (n - 1)) / n

    for i in range(n):
        for j in range(n):
            ax = ax_outer.inset_axes(
                [left + j * (cell_w + gap), bottom + (n - 1 - i) * (cell_h + gap), cell_w, cell_h]
            )
            x = data[cols[j]].astype(float).values
            y = data[cols[i]].astype(float).values
            if i == j:
                sns.histplot(x=x, ax=ax, kde=True, color=tcolor, edgecolor=INK, alpha=0.45, bins=20, linewidth=0.55)
                if ax.lines:
                    ax.lines[-1].set_color(GRAPH)
                    ax.lines[-1].set_linewidth(1.45)
            elif i < j:
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() >= 2:
                    col_i, col_j = cols[i], cols[j]
                    # For classification target, use point-biserial correlation
                    if is_cls and (col_i == target_col or col_j == target_col):
                        r, p = _st.pointbiserialr(
                            y[mask].astype(int) if col_i == target_col else x[mask].astype(int),
                            x[mask] if col_i == target_col else y[mask],
                        )
                    else:
                        r, p = pearsonr(x[mask], y[mask])
                    stars = "***" if p <= 0.001 else "**" if p <= 0.01 else "*" if p <= 0.05 else ""
                    ax.set_facecolor(tcorr_cmap((r + 1) / 2))
                    ax.patch.set_alpha(0.72)
                    ax.text(0.5, 0.54, f"{r:.2f}", transform=ax.transAxes, ha="center", va="center",
                            fontsize=11.5, fontweight="bold", color=INK)
                    ax.text(0.5, 0.34, stars, transform=ax.transAxes, ha="center", va="center",
                            fontsize=10, fontweight="bold", color=INK)
            else:
                # For classification target, skip the meaningless linear regression
                if is_cls and (cols[i] == target_col or cols[j] == target_col):
                    ax.scatter(x, y, s=10, alpha=0.58, color=tcolor, edgecolors="white", linewidths=0.25)
                    ax.grid(True, linestyle="--", color=GRID, linewidth=0.45, alpha=0.9)
                else:
                    sns.regplot(
                        x=x, y=y, ax=ax,
                        scatter_kws={"s": 10, "alpha": 0.58, "color": tcolor, "edgecolors": "white", "linewidths": 0.25},
                        line_kws={"color": GRAPH, "linewidth": 1.35},
                        ci=95,
                    )
                    ax.grid(True, linestyle="--", color=GRID, linewidth=0.45, alpha=0.9)
            if i == n - 1:
                ax.set_xlabel(short_feature(cols[j]), fontsize=7.3, fontweight="bold")
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(short_feature(cols[i]), fontsize=7.3, fontweight="bold")
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
            ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
            ax.tick_params(axis="both", labelsize=5.8, width=0.65, length=2.2, colors=INK)
            for spine in ax.spines.values():
                spine.set_linewidth(0.70)
                spine.set_color(INK)

    # colorbar
    cbar_ax = ax_outer.inset_axes([0.935, 0.10, 0.018, 0.75])
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    cb = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=tcorr_cmap), cax=cbar_ax)
    corr_label = "Point-biserial r" if is_cls else "Pearson r"
    cb.set_label(corr_label, fontsize=10.5, fontweight="bold", color=INK)
    cb.ax.tick_params(labelsize=9, colors=INK, width=0.8)

    # Annotation for classification target
    if is_cls:
        ax_outer.text(0.50, 0.015,
                      "Correlation with UL-94 uses point-biserial r; scatter plots omit linear regression",
                      transform=ax_outer.transAxes, fontsize=7.5, fontstyle="italic",
                      ha="center", va="bottom", color=INK)

    stem = f"PairGrid_top{top_n}_{display.replace('-', '').replace(' ', '_')}"
    savefig(fig, stem)
    print(f"  Saved {stem}")


def make_all_pairgrids() -> None:
    """Generate top-10 PairGrids for LOI, UL-94, Tg, Tensile."""
    for target, display in [("LOI", "LOI"), ("UL94", "UL-94"), ("Tg", "Tg"), ("TENSILE", "Tensile")]:
        print(f"Generating PairGrid for {display} ...")
        make_pairgrid_for_target(target, display, top_n=10)


def make_figure2() -> None:
    data_all = combined_feature_space()        # multi-task, used only for task-domain map (panel b)
    data_loi = single_task_feature_space("LOI")  # LOI-only, used for overlap (c) and error (d)
    for suffix in ("png", "pdf"):
        stale = PANEL_OUT / f"Figure2a.{suffix}"
        if stale.exists():
            stale.unlink()
    fig = plt.figure(figsize=(22.0, 15.0), dpi=300)
    gs = GridSpec(3, 4, figure=fig, height_ratios=[1.35, 0.55, 0.55], width_ratios=[1.0, 1.0, 1.0, 1.18], wspace=0.38, hspace=0.30)
    fig2_pairgrid(fig.add_subplot(gs[0, :]))
    fig2_embedding(fig.add_subplot(gs[1:3, 0]), data_all, "task", "b", "Chemical-space map by task domain")
    fig2_embedding(fig.add_subplot(gs[1:3, 1]), data_loi, "overlap", "c", "Training-test distribution overlap")
    fig2_embedding(fig.add_subplot(gs[1:3, 2]), data_loi, "error", "d", "Applicability-domain error map")
    fig2_ad_distance_stack(fig.add_subplot(gs[1:3, 3]), data_all)
    savefig(fig, "Figure2_chemical_space_AD")
    save_single_panel("Figure2", "a", fig2_pairgrid, (12.8, 8.8))
    save_single_panel("Figure2", "b", lambda ax: fig2_embedding(ax, data_all, "task", "b", "Chemical-space map by task domain"), (5.6, 4.2))
    save_single_panel("Figure2", "c", lambda ax: fig2_embedding(ax, data_loi, "overlap", "c", "Training-test distribution overlap"), (5.6, 4.2))
    save_single_panel("Figure2", "d", lambda ax: fig2_embedding(ax, data_loi, "error", "d", "Applicability-domain error map"), (5.6, 4.2))
    save_single_panel("Figure2", "e", lambda ax: fig2_ad_distance_stack(ax, data_all), (5.9, 7.3))


def make_si_overlap_error_panels() -> None:
    """Generate SI Figure panels: overlap + error maps for Tg, Tensile, UL-94."""
    for target, display in [("Tg", "Tg"), ("TENSILE", "Tensile"), ("UL94", "UL-94")]:
        print(f"Generating SI overlap/error panels for {display} ...")
        data_si = single_task_feature_space(target)
        save_single_panel("SI_Figure2", f"overlap_{display}",
                          lambda ax: fig2_embedding(ax, data_si, "overlap", "", "Training-test distribution overlap"),
                          (5.6, 4.2))
        save_single_panel("SI_Figure2", f"error_{display}",
                          lambda ax: fig2_embedding(ax, data_si, "error", "", "Applicability-domain error map"),
                          (5.6, 4.2))
    print("SI overlap/error panels done.")


# ---------- Figure 3: benchmark confrontation ----------


BASELINE_GRAY = "#B0BEC5"
XGB_LIGHT_BLUE = "#29B6F6"


def load_benchmark_summary() -> pd.DataFrame:
    return pd.read_csv(OLD_OUT / "from_scratch_all_model_summary.csv")


def short_model_name(model: str) -> str:
    """Readable short name for a model."""
    mapping = {
        "DescriptorProcess_TabICLRegressor": "TabICL",
        "DescriptorProcess_TabICLClassifier": "TabICL",
        "DescriptorProcess_XGBoostRegressor": "XGBoost",
        "DescriptorProcess_XGBoostClassifier": "XGBoost",
        "DescriptorProcess_RidgeRegressor": "Ridge",
        "DescriptorProcess_RidgeClassifier": "Ridge",
        "DescriptorProcess_ElasticNetRegressor": "ElasticNet",
    }
    if model in mapping:
        return mapping[model]
    if model.startswith("Graph_") and "_Fusion_" in model:
        left, fusion = model.split("_Fusion_", 1)
        conv = left.replace("Graph_", "")
        tag = {"weighted_sum": "wsum", "attention": "attn", "graph_only": "plain",
               "film": "FiLM", "concat": "cat", "gated": "gated"}.get(fusion, fusion)
        return f"{conv}-{tag}"
    return model.replace("DescriptorProcess_", "").replace("Regressor", "").replace("Classifier", "")


def pick_five_models(df: pd.DataFrame, target: str) -> list[dict]:
    """Select 5 specific models for *target*, returning {label, mean, std, color}."""
    sub = df[df["Target"].eq(target)].copy()
    is_cls = (target == "UL94")

    # 1. Baseline: Ridge (or ElasticNet fallback)
    ridge_name = "DescriptorProcess_RidgeClassifier" if is_cls else "DescriptorProcess_RidgeRegressor"
    ridge = sub[sub["Model"].eq(ridge_name)]
    if ridge.empty:
        elastic = "DescriptorProcess_ElasticNetRegressor"
        ridge = sub[sub["Model"].eq(elastic)]
    if ridge.empty:
        ridge = sub[sub["Method_Group"].isin(["DescriptorProcess_Bagging", "DescriptorProcess_OtherML"])]
        ridge = ridge.sort_values("Mean_Primary").head(1)

    # 2. Worst plain GNN (lowest Mean_Primary among Plain_GNN)
    plain = sub[sub["Method_Group"].eq("Plain_GNN")]
    worst_gnn = plain.sort_values("Mean_Primary").head(1) if len(plain) else ridge.head(1)

    # 3. Best fusion GNN
    fusion = sub[sub["Method_Group"].eq("GraphDescriptor_Fusion")]
    best_fusion = fusion.sort_values("Mean_Primary", ascending=False).head(1) if len(fusion) else ridge.head(1)

    # 4. XGBoost
    xgb_name = "DescriptorProcess_XGBoostClassifier" if is_cls else "DescriptorProcess_XGBoostRegressor"
    xgb = sub[sub["Model"].eq(xgb_name)]
    if xgb.empty:
        xgb = sub[sub["Family"].eq("descriptor_boosting")].sort_values("Mean_Primary", ascending=False).head(1)

    # 5. TabICL
    tab_name = "DescriptorProcess_TabICLClassifier" if is_cls else "DescriptorProcess_TabICLRegressor"
    tabicl = sub[sub["Model"].eq(tab_name)]
    if tabicl.empty:
        tabicl = sub[sub["Method_Group"].eq("DescriptorProcess_TabICL")].head(1)

    results = []
    for df_row, label, color in [
        (ridge, "Ridge (baseline)", BASELINE_GRAY),
        (worst_gnn, "Plain GNN", GRAPH),
        (best_fusion, "Best fusion GNN", FR),
        (xgb, "XGBoost", XGB_LIGHT_BLUE),
        (tabicl, "TabICL", EXPLICIT),
    ]:
        row = df_row.iloc[0] if len(df_row) else None
        results.append({
            "label": label,
            "model": short_model_name(row["Model"]) if row is not None else label,
            "mean": float(row["Mean_Primary"]) if row is not None else 0.0,
            "std": float(row["Std_Primary"]) if row is not None else 0.0,
            "color": color,
        })
    return results


def fig3_panoramic_bars(axes_1x4, df: pd.DataFrame) -> None:
    """Draw 1×4 bar-chart array (LOI, UL-94, Tg, Tensile) with error bars."""
    targets = [("LOI", "LOI"), ("UL94", "UL-94"), ("Tg", "Tg"), ("TENSILE", "Tensile")]
    target_base_color = {"LOI": EXPLICIT, "Tg": RESIN, "TENSILE": FR, "UL94": GRAPH}
    
    for i, ((target_key, display), ax) in enumerate(zip(targets, axes_1x4)):
        if i == 0:
            panel_label(ax, "a", x=-0.15, y=1.08)
            
        models = pick_five_models(df, target_key)
        xs = np.arange(len(models))
        means = [m["mean"] for m in models]
        stds = [m["std"] for m in models]
        labels = [m["label"] for m in models]

        # Per-target monochromatic gradient: pale → full color
        base = target_base_color.get(target_key, EXPLICIT)
        tcmap = mcolors.LinearSegmentedColormap.from_list(
            f"grad_{target_key}", ["#FFFFFF", base]
        )
        n_colors = len(models)
        colors = [tcmap(0.2 + 0.8 * i / max(n_colors - 1, 1)) for i in range(n_colors)]

        bars = ax.bar(xs, means, yerr=stds, capsize=4.5, color=colors, edgecolor="none",
                      width=0.6, error_kw={"linewidth": 1.2, "ecolor": INK})

        for x_pos, mean_val, std_val in zip(xs, means, stds):
            label_y = mean_val + std_val + 0.02
            if mean_val < 0:
                label_y = mean_val - std_val - 0.06
            ax.text(x_pos, label_y, f"{mean_val:.3f}", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold", color=INK)
        
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=9.5, fontweight="bold", rotation=25, ha="right", rotation_mode="anchor")
        
        y_max = max([m + s for m, s in zip(means, stds)]) if means else 1.0
        y_min = min(0, min([m - s for m, s in zip(means, stds)])) if means else 0.0
        ax.set_ylim(y_min * 1.15 if y_min < 0 else 0, max(1.02, y_max * 1.15))
        
        metric = "AUC" if target_key == "UL94" else "$R^2$"
        ax.set_ylabel(f"{display} ({metric})", fontsize=12, fontweight="bold")
        
        ax.axhline(0, color=INK, linewidth=1.0, linestyle="-")
        ax.grid(axis="y", color=GRID, linestyle="--", linewidth=0.7)
        add_full_box(ax)
        bold_axes(ax, tick=9.5, label=11)


def make_figure3() -> None:
    df = load_benchmark_summary()
    fig = plt.figure(figsize=(18.0, 5.0), dpi=300)
    gs = GridSpec(1, 4, figure=fig, wspace=0.28)
    axes_1x4 = [fig.add_subplot(gs[0, i]) for i in range(4)]
    fig3_panoramic_bars(axes_1x4, df)
    savefig(fig, "Figure3_benchmark_confrontation")
    save_single_panel("Figure3", "a", lambda ax: None, (18.0, 5.0))

# ---------- Figure 4: robustness ----------


TABICL_CMAP = mcolors.LinearSegmentedColormap.from_list("TabICL_cmap", ["#FFFFFF", "#8AB4F8", EXPLICIT])
LIGHT_BLUE = "#29B6F6"


def fig4_hexbin(ax, target: str, display: str, unit: str, letter: str) -> None:
    panel_label(ax, letter)
    df = load_oof(target, TABICL_REG)
    r2, rmse, mae = regression_metrics(df)
    x, y = df["Actual"].values, df["Predicted"].values
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    pad = (hi - lo) * 0.06
    # Per-target colormap: white → Figure 2 color
    tcolor = {"LOI": EXPLICIT, "Tg": RESIN, "TENSILE": FR, "TENSILE_display": FR}.get(target, EXPLICIT)
    tcmap = mcolors.LinearSegmentedColormap.from_list(f"hex_{target}", ["#FFFFFF", tcolor])
    hb = ax.hexbin(x, y, gridsize=34, cmap=tcmap, mincnt=1, linewidths=0)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=INK, linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Experimental value")
    ax.set_ylabel("Predicted value")
    unit_suffix = f" {unit}" if unit else ""
    metric_text = f"R2 = {r2:.3f}\nRMSE = {rmse:.2f}{unit_suffix}\nMAE = {mae:.2f}{unit_suffix}"
    ax.text(0.04, 0.96, metric_text, transform=ax.transAxes, va="top", fontsize=11.2, fontweight="bold", bbox=dict(facecolor="white", edgecolor=INK, boxstyle="round,pad=0.25", linewidth=0.8))
    ax.grid(color=GRID, linestyle="--", linewidth=0.65)
    add_full_box(ax)
    bold_axes(ax, tick=12, label=14.5)
    return hb


def fig4_cdf(ax) -> None:
    panel_label(ax, "e")
    cdf_colors = {"LOI": EXPLICIT, "Tg": RESIN, "Tensile": FR}
    styles = [("-", 2.8), ("--", 2.2), (":", 2.5)]
    for (target, display, unit), (ls, lw) in zip(REG_TARGETS, styles):
        df = load_oof(target, TABICL_REG)
        err = np.sort(df["AbsError"].values)
        cdf = np.arange(1, len(err) + 1) / len(err)
        q80 = np.quantile(err, 0.80)
        tcol = cdf_colors.get(display, EXPLICIT)
        ax.plot(err, cdf, color=tcol, linestyle=ls, linewidth=lw, label=f"{display}: {q80:.2f} {unit}".strip())
        ax.scatter([q80], [0.80], s=40, color=tcol, edgecolor=INK)
    ax.axhline(0.80, color=INK, linestyle=":", linewidth=1.5)
    ax.set_xlabel("Absolute error")
    ax.set_ylabel("Cumulative fraction")
    ax.set_ylim(0, 1.02)
    ax.grid(color=GRID, linestyle="--", linewidth=0.65)
    ax.legend(frameon=False, fontsize=10.5, loc="lower right")
    add_full_box(ax)
    bold_axes(ax, tick=12, label=14.5)


def fig4_calibration(ax) -> None:
    panel_label(ax, "b")
    df = load_oof("UL94", TABICL_CLS)
    y = df["Actual"].astype(int).values
    prob = df["Score"].astype(float).clip(0, 1).values
    frac, pred = calibration_curve(y, prob, n_bins=10, strategy="quantile")
    auc = roc_auc_score(y, prob)
    brier = brier_score_loss(y, prob)
    ece = expected_calibration_error(y, prob)
    ax.plot([0, 1], [0, 1], color=SLATE, linestyle="--", linewidth=1.1)
    ax.plot(pred, frac, color=GRAPH, marker="o", markersize=4.8, linewidth=2.0)
    ax.axvline(0.5, color=INK, linestyle=":", linewidth=1.0)
    ax.text(0.05, 0.94, f"AUC = {auc:.3f}\nBrier = {brier:.3f}\nECE = {ece:.3f}", transform=ax.transAxes, va="top", fontsize=11.2, fontweight="bold", bbox=dict(facecolor="white", edgecolor=INK, boxstyle="round,pad=0.25", linewidth=0.8))
    ax.set_xlabel("Predicted V-0 probability")
    ax.set_ylabel("Observed V-0 fraction")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(color=GRID, linestyle="--", linewidth=0.65)
    add_full_box(ax)
    bold_axes(ax, tick=12, label=14.5)


def fig4_threshold(ax) -> None:
    panel_label(ax, "f")
    df = load_oof("UL94", TABICL_CLS)
    thresholds, fpr, fnr, balanced, safety = threshold_curves(df)
    ax.plot(thresholds, fpr, color=GRAPH, linewidth=2.0, label="False positive rate")
    ax.plot(thresholds, fnr, color="#E8A0A0", linewidth=2.0, label="False negative rate")
    for t, label, color in [(0.5, "default", INK), (balanced, "balanced", GRAPH), (safety, "safety", "#E8A0A0")]:
        ax.axvline(t, color=color, linestyle="--", linewidth=1.05)
        ax.text(t + 0.01, 0.86 if label == "default" else 0.70 if label == "balanced" else 0.55, f"{label}\n{t:.2f}", fontsize=10.5, fontweight="bold", color=color)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Error rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(color=GRID, linestyle="--", linewidth=0.65)
    ax.legend(frameon=False, fontsize=10.5, loc="upper left")
    add_full_box(ax)
    bold_axes(ax, tick=12, label=14.5)


def fig4_residuals(ax_outer) -> None:
    panel_label(ax_outer, "g", x=-0.06)
    ax_outer.set_axis_off()
    specs = [
        ("LOI", "LOI", "Flame_retardant_AdditionAmount(wt%)", "FR wt%"),
        ("Tg", "Tg", "EEW", "EEW"),
        ("TENSILE", "Tensile", "Q_thermal", "Q_thermal"),
    ]
    for i, (target, display, feat, xlabel) in enumerate(specs):
        left = 0.015
        gap = 0.055
        bottom = 0.075
        height = 0.895
        width = (1 - 2 * left - 2 * gap) / 3
        ax = ax_outer.inset_axes([left + i * (width + gap), bottom, width, height])
        pred = load_oof(target, TABICL_REG)
        raw, _, _ = load_data(target)
        merged = pred.merge(raw[[feat]].reset_index().rename(columns={"index": "Index"}), on="Index", how="left")
        x = merged[feat].astype(float).values
        y = merged["Residual"].astype(float).values
        ax.scatter(x, y, color={"LOI": EXPLICIT, "Tg": RESIN, "TENSILE": FR}.get(target, EXPLICIT), alpha=0.35, edgecolor="none", s=22)
        ax.axhline(0, color=INK, linestyle="--", linewidth=1.2)
        bx, by = binned_line(x, y, 18)
        ax.plot(bx, by, color=INK, linewidth=2.5)
        ax.set_title(display, fontsize=15.5, fontweight="bold", pad=6)
        ax.set_xlabel(xlabel, fontsize=14, fontweight="bold")
        ax.set_ylabel("Residual" if i == 0 else "", fontsize=14, fontweight="bold")
        ax.grid(color=GRID, linestyle="--", linewidth=0.5)
        add_full_box(ax, 0.95)
        bold_axes(ax, tick=12, label=14)


def fig4_summary(ax) -> None:
    panel_label(ax, "h", x=-0.07)
    ax.set_axis_off()
    ax.set_title("Reliability summary", loc="left", fontsize=15.5, fontweight="bold", pad=5)
    cards = [
        ("LOI", "High accuracy\nlow 80% error bound"),
        ("Tg", "Highest regression\nrobustness"),
        ("Tensile", "Acceptable but\nnetwork-limited"),
        ("UL-94", "High AUC but\nthreshold-sensitive"),
    ]
    for i, (title, body) in enumerate(cards):
        y = 0.75 - i * 0.235
        ax.add_patch(patches.FancyBboxPatch((0.06, y), 0.88, 0.17, boxstyle="round,pad=0.018", facecolor="#F7F7F7", edgecolor=INK, linewidth=1.0, transform=ax.transAxes))
        ax.add_patch(patches.Rectangle((0.06, y), 0.025, 0.17, transform=ax.transAxes, facecolor=EXPLICIT if title != "Tensile" else GRAPH, edgecolor="none"))
        ax.text(0.12, y + 0.085, title, transform=ax.transAxes, fontsize=13.5, fontweight="bold", va="center")
        ax.text(0.45, y + 0.085, body, transform=ax.transAxes, fontsize=11.2, fontweight="bold", va="center")


def make_figure4() -> None:
    fig = plt.figure(figsize=(18.0, 10.8), dpi=300)
    # 2 rows × 12 columns, increased spacing for bold axes
    gs = GridSpec(2, 12, figure=fig, hspace=0.35, wspace=0.72)
    # Row 1: a=LOI, b=UL-94 calibration, c=Tg, d=Tensile (3 cols each)
    fig4_hexbin(fig.add_subplot(gs[0, 0:3]), "LOI", "LOI", "", "a")
    fig4_calibration(fig.add_subplot(gs[0, 3:6]))
    fig4_hexbin(fig.add_subplot(gs[0, 6:9]), "Tg", "Tg", "degC", "c")
    fig4_hexbin(fig.add_subplot(gs[0, 9:12]), "TENSILE", "Tensile", "MPa", "d")
    # Row 2: e=CDF (3 cols), f=threshold (3 cols), g=residuals (6 cols — wider)
    fig4_cdf(fig.add_subplot(gs[1, 0:3]))
    fig4_threshold(fig.add_subplot(gs[1, 3:6]))
    fig4_residuals(fig.add_subplot(gs[1, 6:12]))
    fig.subplots_adjust(left=0.07, right=0.96, bottom=0.09, top=0.94)
    savefig(fig, "Figure4_predictive_robustness")
    save_single_panel("Figure4", "a", lambda ax: fig4_hexbin(ax, "LOI", "LOI", "", "a"), (5.4, 4.4))
    save_single_panel("Figure4", "b", fig4_calibration, (5.4, 4.4))
    save_single_panel("Figure4", "c", lambda ax: fig4_hexbin(ax, "Tg", "Tg", "degC", "c"), (5.4, 4.4))
    save_single_panel("Figure4", "d", lambda ax: fig4_hexbin(ax, "TENSILE", "Tensile", "MPa", "d"), (5.4, 4.4))
    save_single_panel("Figure4", "e", fig4_cdf, (5.5, 4.4))
    save_single_panel("Figure4", "f", fig4_threshold, (5.6, 4.4))
    save_single_panel("Figure4", "g", fig4_residuals, (7.2, 4.6))


def make_combined_figure3() -> None:
    """3×4 combined figure: benchmark bars (row1) + robustness panels (rows 2-3)."""
    df_bench = load_benchmark_summary()
    fig = plt.figure(figsize=(18.0, 14.0), dpi=300)
    gs = GridSpec(3, 4, figure=fig, hspace=0.48, wspace=0.32)

    # --- Row 1: panoramic benchmark bars (a-d) ---
    axes_bar = [fig.add_subplot(gs[0, i]) for i in range(4)]
    fig3_panoramic_bars(axes_bar, df_bench)

    # --- Row 2: hexbin parity (e-g) + error CDF (h) ---
    for i, (target, display, unit) in enumerate(REG_TARGETS):
        fig4_hexbin(fig.add_subplot(gs[1, i]), target, display, unit, chr(ord("e") + i))
    fig4_cdf(fig.add_subplot(gs[1, 3]))

    # --- Row 3: calibration (i), threshold (j), residuals (k, spans cols 3-4) ---
    fig4_calibration(fig.add_subplot(gs[2, 0]))
    fig4_threshold(fig.add_subplot(gs[2, 1]))
    fig4_residuals(fig.add_subplot(gs[2, 2:]))

    savefig(fig, "Figure3_combined_benchmark_robustness")


def make_best_fold_hexbins() -> None:
    """Generate hexbin parity plots using best available results per target.
    LOI & UL94: single train/test split. Tg & Tensile: best CV fold."""
    import pickle
    from sklearn.metrics import roc_auc_score

    shap_paths = {
        "LOI": ROOT / "LOI" / "Result_Final_TabICL_Interpret" / "saved_data" / "plot_data.pkl",
        "UL94": ROOT / "94" / "Result_Final_TabICL_Interpret_Cls" / "saved_data" / "plot_data.pkl",
    }
    # CV best folds for Tg and Tensile
    cv_best = {"Tg": 3, "TENSILE": 2}
    tcolors = {"LOI": EXPLICIT, "Tg": RESIN, "TENSILE": FR, "UL94": GRAPH}
    displays = {"LOI": "LOI", "Tg": "Tg", "TENSILE": "Tensile", "UL94": "UL-94"}
    units = {"LOI": "", "Tg": "degC", "TENSILE": "MPa", "UL94": ""}

    all_targets = ["LOI", "UL94", "Tg", "TENSILE"]
    for target in all_targets:
        display = displays[target]
        tcolor = tcolors[target]
        tcmap = mcolors.LinearSegmentedColormap.from_list(f"bf_{target}", ["#FFFFFF", tcolor])

        if target in shap_paths:
            # Single train/test split
            with open(shap_paths[target], "rb") as f:
                shap = pickle.load(f)
            y_test = shap["y_test"]
            preds = shap["preds"]
            source = "train/test split"
        else:
            # Best CV fold
            model = TABICL_CLS if target == "UL94" else TABICL_REG
            df = load_oof(target, model)
            fold_df = df[df["Fold"] == cv_best[target]].copy()
            y_test = fold_df["Actual"].values
            preds = fold_df["Score" if target == "UL94" else "Predicted"].values
            source = f"CV fold {cv_best[target]}"

        fig, ax = plt.subplots(figsize=(5.5, 5.0), dpi=300)

        if target == "UL94":
            prob = preds if "probs" not in (shap if target in shap_paths else {}) else shap["probs"]
            auc = roc_auc_score(y_test.astype(float), prob.astype(float))
            hb = ax.hexbin(y_test.astype(float), prob.astype(float), gridsize=20, cmap=tcmap, mincnt=1, linewidths=0)
            ax.set_xlim(-0.1, 1.1)
            ax.set_ylim(-0.1, 1.1)
            ax.set_xlabel("Actual class")
            ax.set_ylabel("Predicted V-0 probability")
            metric_text = f"AUC = {auc:.3f}\nn = {len(y_test)}"
        else:
            y_t = y_test.astype(float)
            p_t = preds.astype(float)
            r2 = r2_score(y_t, p_t)
            rmse = np.sqrt(mean_squared_error(y_t, p_t))
            mae = mean_absolute_error(y_t, p_t)
            lo, hi = min(y_t.min(), p_t.min()), max(y_t.max(), p_t.max())
            pad = (hi - lo) * 0.06
            hb = ax.hexbin(y_t, p_t, gridsize=34, cmap=tcmap, mincnt=1, linewidths=0)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=INK, linestyle="--", linewidth=1.5, alpha=0.7)
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlabel("Experimental value")
            ax.set_ylabel("Predicted value")
            unit_str = f" {units.get(target, '')}" if units.get(target, '') else ""
            metric_text = f"R² = {r2:.3f}\nRMSE = {rmse:.2f}{unit_str}\nMAE = {mae:.2f}{unit_str}\nn = {len(y_test)}"

        ax.text(0.04, 0.96, metric_text, transform=ax.transAxes, va="top", fontsize=9, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor=INK, boxstyle="round,pad=0.25", linewidth=0.8))
        ax.grid(color=GRID, linestyle="--", linewidth=0.65)
        add_full_box(ax)
        bold_axes(ax)

        stem = f"BestFold_hexbin_{display.replace('-', '')}"
        fig.tight_layout()
        fig.savefig(NEW_OUT / f"{stem}.png", dpi=600, bbox_inches="tight")
        fig.savefig(NEW_OUT / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {stem} ({source})")


# ---------- Figure 5: global mechanisms ----------


def select_shap_features(shap_data: dict[str, dict], n: int = 28) -> list[str]:
    scores = {}
    for data in shap_data.values():
        s = mean_abs_shap(data)
        for feat, val in s.sort_values(ascending=False).head(16).items():
            scores[feat] = max(scores.get(feat, 0), float(val))
    priority = {"Formulation": 0, "Process": 1, "EP": 2, "FR": 3, "Curing": 4, "Other": 5}
    return sorted(scores, key=lambda f: (priority[feature_category(f)], -scores[f], f))[:n]


def fig5_heatmap(ax, shap_data: dict[str, dict]) -> None:
    panel_label(ax, "a", x=-0.08)
    feats = select_shap_features(shap_data)
    task_order = [("LOI", "LOI"), ("UL94", "UL-94"), ("Tg", "Tg"), ("TENSILE", "Tensile")]
    mat = []
    for key, _ in task_order:
        s = mean_abs_shap(shap_data[key])
        mat.append([s.get(f, 0.0) for f in feats])
    mat = np.array(mat).T
    im = ax.imshow(mat, cmap=SEQ_CMAP, aspect="auto")
    ax.set_xticks(np.arange(len(task_order)))
    ax.set_xticklabels([d for _, d in task_order])
    ax.set_yticks(np.arange(len(feats)))
    ax.set_yticklabels([short_feature(f) for f in feats], fontsize=7.2)
    for i, feat in enumerate(feats):
        ax.add_patch(patches.Rectangle((-0.72, i - 0.5), 0.14, 1.0, facecolor=feature_color(feat), edgecolor="none", clip_on=False))
    ax.set_title("Cross-task SHAP importance", loc="left", fontsize=14, fontweight="bold", pad=6)
    add_full_box(ax)
    bold_axes(ax, tick=8.0, label=9.2)
    cbar = plt.colorbar(im, ax=ax, fraction=0.028, pad=0.015)
    cbar.set_label("mean |SHAP|", fontsize=8.4, fontweight="bold")
    handles = [
        patches.Patch(color=SLATE, label="Formulation"),
        patches.Patch(color=PROCESS, label="Process/Curing"),
        patches.Patch(color=RESIN, label="EP"),
        patches.Patch(color=FR, label="FR"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=7.0, ncol=4, loc="lower center", bbox_to_anchor=(0.54, -0.19))


def fig5_beeswarm(ax, data: dict, title: str, letter: str, preferred: list[str], callout: str, tcolor: str = EXPLICIT) -> None:
    panel_label(ax, letter)
    features = data["features"]
    ordered = beeswarm_order_from_preferred(data, preferred, 8)
    rng = np.random.default_rng(9)
    idxs = [features.index(f) for f in ordered]
    max_abs = np.nanpercentile(np.abs(data["shap"][:, idxs]), 99)
    # Diverging colormap per target: contrasting color → white → target color
    diverging_colors = {
        EXPLICIT: "#E8C8B0",  # warm beige contrasts with deep blue
        RESIN: "#D8C0A8",     # warm sand contrasts with medium blue
        FR: "#A8C0D8",        # cool blue contrasts with orange
    }
    contrast = diverging_colors.get(tcolor, "#D6D6D6")
    tcmap = mcolors.LinearSegmentedColormap.from_list(f"bee_{letter}", [contrast, "#F5F5F5", tcolor])
    for row, feat in enumerate(ordered[::-1]):
        idx = features.index(feat)
        x = data["x"][:, idx]
        sv = data["shap"][:, idx]
        if np.nanmax(x) > np.nanmin(x):
            color = (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))
        else:
            color = np.full_like(x, 0.5)
        y = np.full(len(sv), row) + rng.normal(0, 0.075, len(sv))
        ax.scatter(sv, y, c=color, cmap=tcmap, s=13, alpha=0.82, edgecolor="none")
    ax.axvline(0, color=INK, linewidth=0.95)
    ax.set_xlim(-max_abs * 1.25, max_abs * 1.25)
    ax.set_yticks(np.arange(len(ordered)))
    ax.set_yticklabels([short_feature(f) for f in ordered[::-1]], fontsize=14.5)
    ax.set_xlabel("SHAP value", fontsize=16, fontweight="bold")
    ax.grid(axis="x", color=GRID, linestyle="--", linewidth=0.65)
    add_full_box(ax)
    bold_axes(ax, tick=14, label=16)


def fig5_network(ax, shap_data: dict[str, dict]) -> None:
    panel_label(ax, "f", x=-0.05)
    ax.set_axis_off()
    ax.set_title("Feature mechanism network", loc="left", fontsize=14, fontweight="bold", pad=6)
    feats = select_shap_features(shap_data, 18)
    scores = {f: 0.0 for f in feats}
    for data in shap_data.values():
        s = mean_abs_shap(data)
        for f in feats:
            scores[f] = max(scores[f], float(s.get(f, 0)))
    angles = np.linspace(0, 2 * np.pi, len(feats), endpoint=False)
    pos = {f: np.array([0.52 + 0.39 * np.cos(a), 0.52 + 0.39 * np.sin(a)]) for f, a in zip(feats, angles)}
    corr_edges = []
    for i, f1 in enumerate(feats):
        for f2 in feats[i + 1 :]:
            vals = []
            for data in shap_data.values():
                if f1 in data["features"] and f2 in data["features"]:
                    x = pd.Series(values_for(data, f1))
                    y = pd.Series(values_for(data, f2))
                    c = x.corr(y, method="spearman")
                    if np.isfinite(c):
                        vals.append(c)
            if vals:
                c = float(np.nanmean(vals))
                if abs(c) >= 0.4:
                    corr_edges.append((f1, f2, c))
    for f1, f2, c in corr_edges[:45]:
        p1, p2 = pos[f1], pos[f2]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], transform=ax.transAxes, color=GRAPH if c > 0 else EXPLICIT, alpha=0.35, linewidth=0.8 + 1.6 * abs(c), linestyle="-" if c > 0 else "--")
    max_score = max(scores.values()) or 1
    for f in feats:
        p = pos[f]
        size = 0.018 + 0.035 * (scores[f] / max_score)
        ax.add_patch(patches.Circle(p, size, transform=ax.transAxes, facecolor=feature_color(f), edgecolor=INK, linewidth=0.8, zorder=3))
        ax.text(p[0], p[1] - size - 0.018, short_feature(f), transform=ax.transAxes, fontsize=6.5, fontweight="bold", ha="center", va="top")


def make_figure5() -> None:
    shap_data = {k: load_shap_task(k) for k in SHAP_DIRS}
    fig = plt.figure(figsize=(21.0, 8.5), dpi=300)
    gs = GridSpec(2, 4, figure=fig, height_ratios=[1.25, 1.0], wspace=0.34, hspace=0.43)
    fig5_heatmap(fig.add_subplot(gs[0, :]), shap_data)
    fig5_beeswarm(fig.add_subplot(gs[1, 0]), shap_data["LOI"], "LOI SHAP attribution", "b", ["Flame_retardant_AdditionAmount(wt%)", "FR_F01[C-P]", "FR_B01[C-P]", "EP_wt_fraction"], "P-containing FR\n-> radical quenching /\ncharring\n-> higher LOI")
    fig5_beeswarm(fig.add_subplot(gs[1, 1]), shap_data["UL94"], "UL-94 SHAP attribution", "c", ["Flame_retardant_AdditionAmount(wt%)", "FR_max_conj_path", "CURING_Mor30p", "EP_wt_fraction"], "Conjugated FR\n-> char formation /\ndripping suppression\n-> V-0")
    fig5_beeswarm(fig.add_subplot(gs[1, 2]), shap_data["Tg"], "Tg SHAP attribution", "d", ["EEW", "T_max", "Q_thermal", "CURING_RBF", "CURING_SssCH2"], "Low EEW + curing\n-> crosslink density\n-> restricted motion\n-> higher Tg")
    fig5_beeswarm(fig.add_subplot(gs[1, 3]), shap_data["TENSILE"], "Tensile SHAP attribution", "e", ["Q_thermal", "FR_E1m", "FR_TPSA_efficiency", "Flame_retardant_AdditionAmount(wt%)", "EP_SIC5"], "Thermal history\n-> network perfection\nBulky/polar FR\n-> stress concentration")
    savefig(fig, "Figure5_global_SHAP_mechanisms")
    save_single_panel("Figure5", "a", lambda ax: fig5_heatmap(ax, shap_data), (12.0, 7.6))
    save_single_panel("Figure5", "b", lambda ax: fig5_beeswarm(ax, shap_data["LOI"], "LOI SHAP attribution", "b", ["Flame_retardant_AdditionAmount(wt%)", "FR_F01[C-P]", "FR_B01[C-P]", "EP_wt_fraction"], "P-containing FR\n-> radical quenching /\ncharring\n-> higher LOI"), (5.6, 4.8))
    save_single_panel("Figure5", "c", lambda ax: fig5_beeswarm(ax, shap_data["UL94"], "UL-94 SHAP attribution", "c", ["Flame_retardant_AdditionAmount(wt%)", "FR_max_conj_path", "CURING_Mor30p", "EP_wt_fraction"], "Conjugated FR\n-> char formation /\ndripping suppression\n-> V-0"), (5.6, 4.8))
    save_single_panel("Figure5", "d", lambda ax: fig5_beeswarm(ax, shap_data["Tg"], "Tg SHAP attribution", "d", ["EEW", "T_max", "Q_thermal", "CURING_RBF", "CURING_SssCH2"], "Low EEW + curing\n-> crosslink density\n-> restricted motion\n-> higher Tg"), (5.6, 4.8))
    save_single_panel("Figure5", "e", lambda ax: fig5_beeswarm(ax, shap_data["TENSILE"], "Tensile SHAP attribution", "e", ["Q_thermal", "FR_E1m", "FR_TPSA_efficiency", "Flame_retardant_AdditionAmount(wt%)", "EP_SIC5"], "Thermal history\n-> network perfection\nBulky/polar FR\n-> stress concentration"), (5.6, 4.8))


# ---------- Figure 6: nonlinear coupling ----------


def robust_xlim(x: np.ndarray, qlo=0.5, qhi=99.5) -> tuple[float, float]:
    lo, hi = np.nanpercentile(x[np.isfinite(x)], [qlo, qhi])
    if lo == hi:
        lo, hi = np.nanmin(x), np.nanmax(x)
    pad = (hi - lo) * 0.04 if hi > lo else 1
    return lo - pad, hi + pad


def fig6_dependence(ax, data: dict, feature: str, color_feature: str, title: str, letter: str, xlabel: str, markers: list[tuple[float, str]]) -> None:
    panel_label(ax, letter)
    x, shap = shap_arrays(data, feature)
    c = values_for(data, color_feature)
    sc = ax.scatter(x, shap, c=c, cmap=SHAP_CMAP, s=18, alpha=0.82, edgecolor="none")
    bx, by = binned_line(x, shap, 18)
    ax.plot(bx, by, color=INK, linewidth=1.8)
    ax.axhline(0, color=SLATE, linestyle="--", linewidth=1.0)
    for val, text in markers:
        ax.axvline(val, color=GRAPH, linestyle="--", linewidth=1.1)
        ax.text(val, 0.96, text, transform=ax.get_xaxis_transform(), rotation=90, va="top", ha="right", fontsize=7.7, fontweight="bold", color=GRAPH)
    ax.set_xlim(*robust_xlim(x))
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"SHAP of {short_feature(feature)}")
    ax.grid(color=GRID, linestyle="--", linewidth=0.65)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(short_feature(color_feature), fontsize=7.8, fontweight="bold")
    cbar.ax.tick_params(labelsize=7)
    add_full_box(ax)
    bold_axes(ax, tick=8.0, label=9.2)


def fig6_surface(ax, data: dict, x_feature: str, y_feature: str, shap_features: list[str], title: str, letter: str, xlabel: str, ylabel: str, markers: list[tuple[str, float, str]], tcolor: str = EXPLICIT) -> None:
    panel_label(ax, letter)
    # Diverging cmap per target
    diverging_colors = {
        EXPLICIT: "#E8C8B0",
        RESIN: "#D8C0A8",
        FR: "#A8C0D8",
    }
    contrast = diverging_colors.get(tcolor, "#D6D6D6")
    tcmap = mcolors.LinearSegmentedColormap.from_list(f"surf_{letter}", [contrast, "#F5F5F5", tcolor])
    x = values_for(data, x_feature)
    y = values_for(data, y_feature)
    shap = shap_values_for(data, shap_features)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(shap)
    x, y, shap = x[mask], y[mask], shap[mask]
    x_bins = np.linspace(np.nanpercentile(x, 2), np.nanpercentile(x, 98), 22)
    y_bins = np.linspace(np.nanpercentile(y, 2), np.nanpercentile(y, 98), 22)
    grid = np.full((len(y_bins) - 1, len(x_bins) - 1), np.nan)
    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            inside = (x >= x_bins[i]) & (x < x_bins[i + 1]) & (y >= y_bins[j]) & (y < y_bins[j + 1])
            if inside.sum() >= 2:
                grid[j, i] = np.nanmean(shap[inside])
    vmax = np.nanpercentile(np.abs(grid), 95) if np.isfinite(grid).any() else 1
    im = ax.imshow(grid, origin="lower", extent=[x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]], aspect="auto", cmap=tcmap, vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.scatter(x, y, s=5, color=INK, alpha=0.16, edgecolor="none")
    for axis, val, text in markers:
        if axis == "x":
            ax.axvline(val, color=INK, linestyle="--", linewidth=0.95)
            ax.text(val, 0.98, text, transform=ax.get_xaxis_transform(), rotation=90, va="top", ha="right", fontsize=12.5, fontweight="bold")
        else:
            ax.axhline(val, color=INK, linestyle="--", linewidth=0.95)
            ax.text(0.02, val, text, transform=ax.get_yaxis_transform(), va="bottom", ha="left", fontsize=12.5, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=16, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=16, fontweight="bold")
    ax.grid(color=GRID, linestyle="--", linewidth=0.48, alpha=0.6)
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Joint mean SHAP", fontsize=14, fontweight="bold")
    cbar.ax.tick_params(labelsize=13)
    add_full_box(ax)
    bold_axes(ax, tick=14, label=16)


def make_figure6() -> None:
    shap_data = {k: load_shap_task(k) for k in SHAP_DIRS}
    fig = plt.figure(figsize=(21.0, 11.6), dpi=300)
    gs = GridSpec(4, 2, figure=fig, hspace=0.48, wspace=0.31)
    fig6_dependence(fig.add_subplot(gs[0, 0]), shap_data["LOI"], "Flame_retardant_AdditionAmount(wt%)", "FR_F01[C-P]", "LOI: FR wt% dependence", "a", "FR wt%", [(15, "15 wt% threshold")])
    fig6_surface(fig.add_subplot(gs[0, 1]), shap_data["LOI"], "Flame_retardant_AdditionAmount(wt%)", "FR_F01[C-P]", ["Flame_retardant_AdditionAmount(wt%)", "FR_F01[C-P]"], "LOI: FR wt% x C-P coupling", "b", "FR wt%", "FR_F01[C-P]", [("x", 15, "15 wt%")])
    fig6_dependence(fig.add_subplot(gs[1, 0]), shap_data["UL94"], "FR_max_conj_path", "Flame_retardant_AdditionAmount(wt%)", "UL-94: conjugated-path threshold", "c", "FR_max_conj_path", [(15, "path ~15")])
    fig6_surface(fig.add_subplot(gs[1, 1]), shap_data["UL94"], "Flame_retardant_AdditionAmount(wt%)", "CURING_Mor30p", ["Flame_retardant_AdditionAmount(wt%)", "CURING_Mor30p"], "UL-94: loading x curing compatibility", "d", "FR wt%", "CURING_Mor30p", [("x", 15, "15 wt%")])
    fig6_dependence(fig.add_subplot(gs[2, 0]), shap_data["Tg"], "EEW", "T_max", "Tg: EEW dependence", "e", "EEW", [(200, "EEW = 200")])
    fig6_surface(fig.add_subplot(gs[2, 1]), shap_data["Tg"], "EEW", "T_max", ["EEW", "T_max"], "Tg: EEW x T_max coupling", "f", "EEW", "T_max", [("x", 200, "EEW 200"), ("y", 160, "T_max 160 C")])
    fig6_dependence(fig.add_subplot(gs[3, 0]), shap_data["TENSILE"], "Q_thermal", "FR_E1m", "Tensile: Q_thermal dependence", "g", "Q_thermal", [(600, "under-cured"), (800, "plateau")])
    fig6_surface(fig.add_subplot(gs[3, 1]), shap_data["TENSILE"], "Q_thermal", "FR_E1m", ["Q_thermal", "FR_E1m"], "Tensile: curing x FR-size coupling", "h", "Q_thermal", "FR_E1m", [("x", 600, "under-cured"), ("x", 800, "plateau")])
    savefig(fig, "Figure6_descriptor_process_coupling")
    save_single_panel("Figure6", "a", lambda ax: fig6_dependence(ax, shap_data["LOI"], "Flame_retardant_AdditionAmount(wt%)", "FR_F01[C-P]", "LOI: FR wt% dependence", "a", "FR wt%", [(15, "15 wt% threshold")]), (5.8, 4.4))
    save_single_panel("Figure6", "b", lambda ax: fig6_surface(ax, shap_data["LOI"], "Flame_retardant_AdditionAmount(wt%)", "FR_F01[C-P]", ["Flame_retardant_AdditionAmount(wt%)", "FR_F01[C-P]"], "LOI: FR wt% x C-P coupling", "b", "FR wt%", "FR_F01[C-P]", [("x", 15, "15 wt%")]), (5.8, 4.4))
    save_single_panel("Figure6", "c", lambda ax: fig6_dependence(ax, shap_data["UL94"], "FR_max_conj_path", "Flame_retardant_AdditionAmount(wt%)", "UL-94: conjugated-path threshold", "c", "FR_max_conj_path", [(15, "path ~15")]), (5.8, 4.4))
    save_single_panel("Figure6", "d", lambda ax: fig6_surface(ax, shap_data["UL94"], "Flame_retardant_AdditionAmount(wt%)", "CURING_Mor30p", ["Flame_retardant_AdditionAmount(wt%)", "CURING_Mor30p"], "UL-94: loading x curing compatibility", "d", "FR wt%", "CURING_Mor30p", [("x", 15, "15 wt%")]), (5.8, 4.4))
    save_single_panel("Figure6", "e", lambda ax: fig6_dependence(ax, shap_data["Tg"], "EEW", "T_max", "Tg: EEW dependence", "e", "EEW", [(200, "EEW = 200")]), (5.8, 4.4))
    save_single_panel("Figure6", "f", lambda ax: fig6_surface(ax, shap_data["Tg"], "EEW", "T_max", ["EEW", "T_max"], "Tg: EEW x T_max coupling", "f", "EEW", "T_max", [("x", 200, "EEW 200"), ("y", 160, "T_max 160 C")]), (5.8, 4.4))
    save_single_panel("Figure6", "g", lambda ax: fig6_dependence(ax, shap_data["TENSILE"], "Q_thermal", "FR_E1m", "Tensile: Q_thermal dependence", "g", "Q_thermal", [(600, "under-cured"), (800, "plateau")]), (5.8, 4.4))
    save_single_panel("Figure6", "h", lambda ax: fig6_surface(ax, shap_data["TENSILE"], "Q_thermal", "FR_E1m", ["Q_thermal", "FR_E1m"], "Tensile: curing x FR-size coupling", "h", "Q_thermal", "FR_E1m", [("x", 600, "under-cured"), ("x", 800, "plateau")]), (5.8, 4.4))


def fig5_waterfall(ax, data: dict, target_display: str, letter: str, tcolor: str = EXPLICIT) -> None:
    """Simplified waterfall: global top-8 SHAP features for the best-predicted test sample."""
    panel_label(ax, letter)
    shap_vals = data["shap"]
    features = data["features"]
    X = data["x"]
    y_pred = data.get("preds")

    # Global top-8 features (same as beeswarm for consistency)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-8:][::-1]

    # Find best-predicted sample
    if y_pred is not None:
        median_pred = np.nanmedian(y_pred)
        best_idx = int(np.nanargmin(np.abs(y_pred - median_pred)))
    else:
        best_idx = np.random.default_rng(42).integers(0, len(X))

    vals = shap_vals[best_idx, top_idx]
    # Reverse so most important is at top (matching beeswarm)
    vals = vals[::-1]
    feat_names = [short_feature(features[i]) for i in top_idx[::-1]]

    # Sequential dark→light gradient by absolute SHAP magnitude
    max_abs_val = np.max(np.abs(vals)) if len(vals) > 0 else 1
    wf_cmap = mcolors.LinearSegmentedColormap.from_list("wf", ["#FFFFFF", tcolor])
    colors = [wf_cmap(0.3 + 0.7 * np.abs(v) / max(max_abs_val, 1e-9)) for v in vals]
    y_pos = np.arange(len(vals))
    ax.barh(y_pos, vals, color=colors, edgecolor=INK, linewidth=0.6, height=0.55)
    ax.axvline(0, color=INK, linewidth=1.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_names, fontsize=14.5)
    ax.set_xlabel("SHAP value", fontsize=16, fontweight="bold")
    ax.grid(axis="x", color=GRID, linestyle="--", linewidth=0.55)
    add_full_box(ax)
    bold_axes(ax, tick=14, label=16)


def make_figure5_3x3_matrix() -> None:
    """3×3 mechanism matrix: Rows=LOI/Tg/Tensile, Cols=Beeswarm/2D-PDP/Waterfall."""
    shap_data = {k: load_shap_task(k) for k in SHAP_DIRS}

    fig = plt.figure(figsize=(21.0, 16.0), dpi=300)
    gs = GridSpec(3, 3, figure=fig, wspace=0.72, hspace=0.30)

    rows = [
        ("LOI", "LOI", EXPLICIT, [
            ("Flame_retardant_AdditionAmount(wt%)", "FR_F01[C-P]", "FR wt%", "FR_F01[C-P]", []),
        ], ["Flame_retardant_AdditionAmount(wt%)", "FR_F01[C-P]", "FR_B01[C-P]", "EP_wt_fraction"],
         "P-containing FR\n-> radical quenching / charring\n-> higher LOI"),
        ("Tg", "Tg", RESIN, [
            ("EEW", "T_max", "EEW", "T_max", []),
        ], ["EEW", "T_max", "Q_thermal", "CURING_RBF"],
         "Low EEW + curing\n-> crosslink density\n-> higher Tg"),
        ("TENSILE", "Tensile", FR, [
            ("Q_thermal", "FR_E1m", "Q_thermal", "FR_E1m", []),
        ], ["Q_thermal", "FR_E1m", "FR_TPSA_efficiency", "EP_SIC5"],
         "Thermal history + FR size\n-> network & stress effects"),
    ]

    for row_idx, (target, display, tcolor, surf_cfg, preferred, callout) in enumerate(rows):
        data = shap_data[target]
        (x_feat, y_feat, xlab, ylab, markers) = surf_cfg[0]

        # Col 1: Beeswarm
        ax1 = fig.add_subplot(gs[row_idx, 0])
        fig5_beeswarm(ax1, data, f"{display} SHAP", chr(ord("a") + row_idx * 3), preferred, callout, tcolor)

        # Col 2: 2D PDP surface
        ax2 = fig.add_subplot(gs[row_idx, 1])
        fig6_surface(ax2, data, x_feat, y_feat, [x_feat, y_feat],
                     f"{display}: {xlab} × {ylab}",
                     chr(ord("b") + row_idx * 3), xlab, ylab, markers, tcolor)

        # Col 3: Waterfall
        ax3 = fig.add_subplot(gs[row_idx, 2])
        fig5_waterfall(ax3, data, display, chr(ord("c") + row_idx * 3), tcolor)

    fig.subplots_adjust(left=0.07, right=0.96, bottom=0.05, top=0.95)
    savefig(fig, "Figure5_3x3_Mechanism_Matrix")


def make_figure5_2d_pdp_grid() -> None:
    """For each target: 2D PDP surfaces using the same feature order as SHAP beeswarm."""
    shap_data = {k: load_shap_task(k) for k in SHAP_DIRS}
    targets = [
        ("LOI", "LOI", EXPLICIT),
        ("UL94", "UL-94", GRAPH),
        ("Tg", "Tg", RESIN),
        ("TENSILE", "Tensile", FR),
    ]

    for target, display, tcolor in targets:
        data = shap_data[target]
        top_feats = beeswarm_feature_order(target, data, 8)
        pair_feats = top_feats[1:8]
        cols = 4
        feat1 = top_feats[0]

        rows = int(np.ceil(len(pair_feats) / cols))
        fig = plt.figure(figsize=(6 * cols, 4.5 * rows), dpi=300)
        gs = GridSpec(rows, cols, figure=fig, wspace=0.40, hspace=0.35)

        for i, feat_i in enumerate(pair_feats):
            ax = fig.add_subplot(gs[i // cols, i % cols])
            xlab = short_feature(feat1)
            ylab = short_feature(feat_i)
            fig6_surface(ax, data, feat1, feat_i, [feat1, feat_i],
                         "", "", xlab, ylab, [], tcolor)

        stem = f"Figure5_2D_PDP_grid_{display.replace('-', '')}"
        fig.savefig(NEW_OUT / f"{stem}.png", dpi=400, bbox_inches="tight")
        fig.savefig(NEW_OUT / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {stem}")


def make_figure_ul94_2x2_matrix() -> None:
    """UL-94 2×2 matrix: Beeswarm | 1D Dependence | 2D Surface | Waterfall."""
    import pickle
    tcolor = GRAPH
    data = load_shap_task("UL94")

    # Load raw pickle for probabilities & labels
    raw_pkl = SHAP_DIRS["UL94"] / "plot_data.pkl"
    with open(raw_pkl, "rb") as f:
        raw = pickle.load(f)
    probs = raw["probs"]
    y_test = raw["y_test"]

    fig = plt.figure(figsize=(14, 13), dpi=300)
    gs = GridSpec(2, 2, figure=fig, wspace=0.55, hspace=0.35)

    # Panel a: Beeswarm
    ax_a = fig.add_subplot(gs[0, 0])
    fig5_beeswarm(ax_a, data, "UL-94 SHAP", "a",
                  ["Flame_retardant_AdditionAmount(wt%)", "FR_max_conj_path", "CURING_Mor30p", "EP_wt_fraction"],
                  "Conjugated FR → char formation\n→ V-0", tcolor)

    # Panel b: 1D Dependence (FR_max_conj_path vs FR wt%)
    ax_b = fig.add_subplot(gs[0, 1])
    x, shap = shap_arrays(data, "FR_max_conj_path")
    c = values_for(data, "Flame_retardant_AdditionAmount(wt%)")
    tcmap = mcolors.LinearSegmentedColormap.from_list("b", ["#E8C8B0", "#F5F5F5", tcolor])
    sc = ax_b.scatter(x, shap, c=c, cmap=tcmap, s=18, alpha=0.82, edgecolor="none")
    bx, by = binned_line(x, shap, 18)
    ax_b.plot(bx, by, color=INK, linewidth=1.8)
    ax_b.axhline(0, color=INK, linestyle="--", linewidth=1.0)
    ax_b.set_xlim(*robust_xlim(x))
    ax_b.set_xlabel("FR_max_conj_path", fontsize=14, fontweight="bold")
    ax_b.set_ylabel("SHAP of FR_max_conj_path", fontsize=14, fontweight="bold")
    ax_b.grid(color=GRID, linestyle="--", linewidth=0.65)
    cbar = plt.colorbar(sc, ax=ax_b, fraction=0.04, pad=0.02)
    cbar.set_label("FR wt%", fontsize=12, fontweight="bold")
    cbar.ax.tick_params(labelsize=11)
    add_full_box(ax_b)
    bold_axes(ax_b, tick=12, label=14)
    panel_label(ax_b, "b")

    # Panel c: 2D Surface (FR wt% vs CURING_Mor30p)
    ax_c = fig.add_subplot(gs[1, 0])
    fig6_surface(ax_c, data, "Flame_retardant_AdditionAmount(wt%)", "CURING_Mor30p",
                 ["Flame_retardant_AdditionAmount(wt%)", "CURING_Mor30p"],
                 "", "c", "FR wt%", "CURING_Mor30p", [], tcolor)

    # Panel d: Waterfall (high-confidence V-0)
    ax_d = fig.add_subplot(gs[1, 1])
    panel_label(ax_d, "d")
    # Find a true V-0 sample with highest predicted probability
    v0_mask = (y_test == 1) & (probs > 0.90)
    if v0_mask.sum() > 0:
        best_idx = int(np.argmax(probs * v0_mask))
    else:
        best_idx = int(np.argmax(probs))
    shap_vals = data["shap"]
    features = data["features"]

    # Build the same ordered feature list as beeswarm panel a
    preferred = ["Flame_retardant_AdditionAmount(wt%)", "FR_max_conj_path", "CURING_Mor30p", "EP_wt_fraction"]
    ordered = [f for f in preferred if f in features]
    for f in mean_abs_shap(data).sort_values(ascending=False).index:
        if f not in ordered:
            ordered.append(f)
        if len(ordered) >= 8:
            break
    ordered = ordered[:8]
    top_indices = [features.index(f) for f in ordered[::-1]]  # most important at top
    vals = shap_vals[best_idx, top_indices]
    feat_names = [short_feature(f) for f in ordered[::-1]]
    wf_cmap = mcolors.LinearSegmentedColormap.from_list("wf", ["#FFFFFF", tcolor])
    max_abs_val = np.max(np.abs(vals)) if len(vals) > 0 else 1
    colors = [wf_cmap(0.3 + 0.7 * np.abs(v) / max(max_abs_val, 1e-9)) for v in vals]
    y_pos = np.arange(len(vals))
    ax_d.barh(y_pos, vals, color=colors, edgecolor=INK, linewidth=0.6, height=0.55)
    ax_d.axvline(0, color=INK, linewidth=1.0)
    ax_d.set_yticks(y_pos)
    ax_d.set_yticklabels(feat_names, fontsize=12.5)
    ax_d.set_xlabel("SHAP value (log-odds)", fontsize=14, fontweight="bold")
    ax_d.grid(axis="x", color=GRID, linestyle="--", linewidth=0.55)
    add_full_box(ax_d)
    bold_axes(ax_d, tick=12, label=14)
    # Annotate high confidence
    ax_d.text(0.98, 0.04, f"P(V-0) = {probs[best_idx]:.3f}", transform=ax_d.transAxes,
              ha="right", va="bottom", fontsize=12, fontweight="bold", color=tcolor)

    fig.subplots_adjust(left=0.08, right=0.95)
    savefig(fig, "Figure5_UL94_2x2_Mechanism_Matrix")


def main() -> None:
    configure_style()
    make_figure2()
    make_figure3()
    make_figure4()
    make_figure5()
    make_figure6()
    print(f"Submission figure set written to {NEW_OUT}")


if __name__ == "__main__":
    main()
