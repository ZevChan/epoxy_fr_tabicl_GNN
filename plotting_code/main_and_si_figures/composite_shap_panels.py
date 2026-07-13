"""Composite figures from existing SHAP images."""
import os
from pathlib import Path
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.stats import gaussian_kde

SRC = Path(r"C:\Users\WINDOWS\Desktop\GNN\translated_text2")
DST = SRC  # save to same folder

INK = "#333333"
GRID = "#E6E6E6"

TARGETS = [
    ("LOI", "LOI"),
    ("UL94", "UL-94"),
    ("Tg", "Tg"),
    ("TENSILE", "Tensile"),
]
TARGET_COLORS = {
    "LOI": "#1F4E79",
    "UL94": "#C0504D",
    "Tg": "#4A7B9D",
    "TENSILE": "#E08E36",
}

def load_img(target, stem):
    """Load PNG for target+stem."""
    p = SRC / f"{target}_{stem}.png"
    if p.exists():
        return mpimg.imread(p)
    return None


def quantile_groups(x_vals, shap_vals):
    """Split SHAP values by feature quantiles, falling back for low-cardinality features."""
    if len(x_vals) == 0:
        return ["Low"], [np.array([0.0])]

    unique = np.unique(x_vals[np.isfinite(x_vals)])
    if len(unique) >= 3:
        q1, q2 = np.nanquantile(x_vals, [1 / 3, 2 / 3])
        if q1 < q2:
            bins = np.digitize(x_vals, [q1, q2], right=True)
            labels = ["Low", "Medium", "High"]
            groups = [shap_vals[bins == i] for i in range(3)]
            keep = [i for i, g in enumerate(groups) if len(g) > 0]
            return [labels[i] for i in keep], [groups[i] for i in keep]

    if len(unique) >= 2:
        med = np.nanmedian(x_vals)
        bins = (x_vals > med).astype(int)
        labels = ["Low", "High"]
        groups = [shap_vals[bins == i] for i in range(2)]
        keep = [i for i, g in enumerate(groups) if len(g) > 0]
        return [labels[i] for i in keep], [groups[i] for i in keep]

    return ["Constant"], [shap_vals]


def composite_waterfalls():
    """Combine all Waterfall images per target (3×3 grid: Best/Worst/Highest × Top1/Top2/Top3)."""
    for target, disp in TARGETS:
        if target == "UL94":
            wf_types = ["True_Positive", "False_Positive"]
        else:
            wf_types = ["Best_Prediction", "Worst_Prediction", f"Highest_{target}"]

        imgs = []
        for wf_type in wf_types:
            for top in ["Top1", "Top2", "Top3"]:
                stem = f"4_Waterfall_{wf_type}_{top}"
                img = load_img(target, stem)
                if img is not None:
                    imgs.append((img, f"{wf_type}_{top}"))

        if not imgs:
            print(f"  SKIP Waterfall {disp}: no images")
            continue

        n = len(imgs)
        cols = min(3, n)
        rows = (n + cols - 1) // cols

        fig = plt.figure(figsize=(6 * cols, 5 * rows), dpi=300)
        gs = GridSpec(rows, cols, figure=fig, hspace=0.06, wspace=0.15)

        for i, (img, label) in enumerate(imgs):
            ax = fig.add_subplot(gs[i // cols, i % cols])
            ax.imshow(img)
            ax.axis("off")
            # Add subplot letter
            letter = chr(ord("a") + i)
            ax.text(0.02, 0.98, letter, transform=ax.transAxes, fontsize=18, fontweight="bold",
                    va="top", ha="left", color=INK,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85))

        stem = f"Composite_Waterfall_{disp.replace('-', '')}"
        fig.savefig(DST / f"{stem}.png", dpi=400, bbox_inches="tight")
        fig.savefig(DST / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {stem} ({n} panels)")


def composite_quantile_impact():
    """Draw QuantileImpact panels per target in the same feature order as the Figure 5 beeswarm."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from make_submission_figures_v2 import beeswarm_feature_order, load_shap_task, short_feature

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    for target, disp in TARGETS:
        try:
            data = load_shap_task(target)
            ordered = [f for f in beeswarm_feature_order(target, data, 8) if f in data["features"]]
        except Exception as exc:
            print(f"  SKIP QuantileImpact {disp}: {exc}")
            continue

        n = len(ordered)
        cols = 4
        rows = (n + cols - 1) // cols
        fig = plt.figure(figsize=(6.3 * cols, 5.2 * rows), dpi=300)
        gs = GridSpec(rows, cols, figure=fig, hspace=0.34, wspace=0.30)
        color = TARGET_COLORS[target]

        for i, feat in enumerate(ordered):
            ax = fig.add_subplot(gs[i // cols, i % cols])
            f_idx = data["features"].index(feat)
            x_vals = np.asarray(data["x"][:, f_idx], dtype=float)
            shap_vals = np.asarray(data["shap"][:, f_idx], dtype=float)
            mask = np.isfinite(x_vals) & np.isfinite(shap_vals)
            x_vals = x_vals[mask]
            shap_vals = shap_vals[mask]

            labels, groups = quantile_groups(x_vals, shap_vals)
            positions = np.arange(1, len(groups) + 1)
            bp = ax.boxplot(
                groups,
                positions=positions,
                widths=0.52,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "#777777", "linewidth": 1.4},
                boxprops={"facecolor": "white", "edgecolor": color, "linewidth": 2.1},
                whiskerprops={"color": "#AAAAAA", "linewidth": 1.2},
                capprops={"color": "#AAAAAA", "linewidth": 1.2},
            )
            for box in bp["boxes"]:
                box.set_alpha(0.98)

            rng = np.random.default_rng(2024 + i)
            for pos, group in zip(positions, groups):
                jitter = rng.normal(0, 0.055, size=len(group))
                ax.scatter(
                    np.full(len(group), pos) + jitter,
                    group,
                    s=26,
                    color=color,
                    alpha=0.62,
                    edgecolors="#666666",
                    linewidths=0.35,
                )

            ax.axhline(0, color="#666666", linestyle="--", linewidth=1.35)
            ax.set_title(short_feature(feat), fontsize=20, fontweight="bold", pad=10)
            ax.set_ylabel(f"SHAP on {disp}", fontsize=18, fontweight="bold")
            ax.set_xticks(positions)
            ax.set_xticklabels(labels, fontsize=16, fontweight="bold")
            ax.tick_params(axis="y", labelsize=16, width=1.2)
            ax.tick_params(axis="x", width=1.2)
            for tick in ax.get_yticklabels():
                tick.set_fontweight("bold")
            for tick in ax.get_xticklabels():
                tick.set_fontweight("bold")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.25)
                spine.set_color("#333333")

        stem = f"Composite_QuantileImpact_{disp.replace('-', '')}"
        fig.savefig(DST / f"{stem}.png", dpi=400, bbox_inches="tight")
        fig.savefig(DST / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {stem} ({n} panels)")


def composite_ridgeline_bar():
    """Draw SHAP_Ridgeline + SHAP_Bar per target with shared font settings."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from make_submission_figures_v2 import beeswarm_feature_order, load_shap_task, mean_abs_shap, short_feature

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    for target, disp in TARGETS:
        try:
            data = load_shap_task(target)
        except Exception as exc:
            print(f"  SKIP Ridgeline+Bar {disp}: {exc}")
            continue
        features = beeswarm_feature_order(target, data, 8)
        if not features:
            print(f"  SKIP Ridgeline+Bar {disp}: no features")
            continue

        color = TARGET_COLORS[target]
        shap_values = data["shap"]
        feature_names = data["features"]
        idxs = [feature_names.index(f) for f in features]
        y_offsets = np.arange(len(features))[::-1]

        fig = plt.figure(figsize=(14, 6.2), dpi=300)
        gs = GridSpec(1, 2, figure=fig, wspace=0.38)
        ax_ridge = fig.add_subplot(gs[0, 0])
        ax_bar = fig.add_subplot(gs[0, 1])

        # Panel a: ridgeline
        ridge_vals = shap_values[:, idxs]
        gmin = float(np.nanmin(ridge_vals))
        gmax = float(np.nanmax(ridge_vals))
        margin = max((gmax - gmin) * 0.12, 1e-6)
        x_grid = np.linspace(gmin - margin, gmax + margin, 500)
        ax_ridge.axvline(0, color="#777777", linestyle="--", linewidth=1.2, zorder=0)
        for row, (feat, f_idx) in enumerate(zip(features, idxs)):
            f_shap = shap_values[:, f_idx]
            f_shap = f_shap[np.isfinite(f_shap)]
            if len(np.unique(f_shap)) > 1:
                try:
                    dens = gaussian_kde(f_shap)(x_grid)
                    dens = dens / max(dens.max(), 1e-12) * 0.82
                except Exception:
                    dens = np.zeros_like(x_grid)
            else:
                dens = np.zeros_like(x_grid)
            base = y_offsets[row]
            ax_ridge.plot(x_grid, dens + base, color=color, linewidth=1.5)
            ax_ridge.fill_between(x_grid, base, dens + base, color=color, alpha=0.55)
        ax_ridge.set_yticks(y_offsets + 0.08)
        ax_ridge.set_yticklabels([short_feature(f) for f in features], fontsize=11.5, fontweight="bold")
        ax_ridge.set_xlabel(f"SHAP value (impact on {disp})", fontsize=12.5, fontweight="bold")
        ax_ridge.tick_params(axis="x", labelsize=11)
        ax_ridge.tick_params(axis="y", length=0)
        ax_ridge.spines["top"].set_visible(False)
        ax_ridge.spines["right"].set_visible(False)
        ax_ridge.spines["left"].set_visible(False)
        ax_ridge.spines["bottom"].set_linewidth(1.1)
        ax_ridge.text(-0.12, 1.03, "a", transform=ax_ridge.transAxes, fontsize=22, fontweight="bold", color=INK)

        # Panel b: SHAP bar
        shap_abs = mean_abs_shap(data)
        values = np.array([float(shap_abs.get(f, 0.0)) for f in features])
        vmax = max(values.max(), 1e-12)
        colors = []
        for v in values:
            alpha = 0.18 + 0.82 * (v / vmax)
            colors.append((*mcolors.to_rgb(color), alpha))
        ax_bar.barh(y_offsets, values, color=colors, edgecolor=INK, linewidth=0.9, height=0.72)
        ax_bar.set_yticks(y_offsets)
        ax_bar.set_yticklabels([short_feature(f) for f in features], fontsize=11.5, fontweight="bold")
        ax_bar.set_xlabel(f"Mean |SHAP value| (impact on {disp})", fontsize=12.5, fontweight="bold")
        ax_bar.tick_params(axis="x", labelsize=11)
        ax_bar.grid(axis="x", color=GRID, linestyle="--", linewidth=0.7)
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)
        ax_bar.spines["left"].set_linewidth(1.1)
        ax_bar.spines["bottom"].set_linewidth(1.1)
        ax_bar.text(-0.12, 1.03, "b", transform=ax_bar.transAxes, fontsize=22, fontweight="bold", color=INK)

        stem = f"Composite_Ridgeline_Bar_{disp.replace('-', '')}"
        fig.savefig(DST / f"{stem}.png", dpi=400, bbox_inches="tight")
        fig.savefig(DST / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {stem}")


if __name__ == "__main__":
    print("=== Waterfall composites ===")
    composite_waterfalls()
    print("=== QuantileImpact composites ===")
    composite_quantile_impact()
    print("=== Ridgeline+Bar composites ===")
    composite_ridgeline_bar()
    print("All done.")
