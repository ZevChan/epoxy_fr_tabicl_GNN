"""Composite Figures 5 & 6 from existing SHAP images with Figure 2b colors."""
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec

SRC = Path(r"C:\Users\WINDOWS\Desktop\GNN\translated_text-TABICL-SHAPtranslated_text_translated_text")
DST = Path(r"C:\Users\WINDOWS\Desktop\GNN\translated_text2")

TARGETS = [
    ("LOI", "LOI"),
    ("UL94", "UL-94"),
    ("Tg", "Tg"),
    ("TENSILE", "Tensile"),
]

INK = "#333333"

# Per-target top-2 1D PDP features and top-2 2D PDP pairs
PDP_FEATURES = {
    "LOI": [
        "Flame_retardant_AdditionAmount(wt%)",
        "FR_F01[C-P]",
    ],
    "UL94": [
        "Flame_retardant_AdditionAmount(wt%)",
        "Curing_agent_AdditionAmount(wt%)",
    ],
    "Tg": [
        "EEW",
        "T_max",
    ],
    "TENSILE": [
        "Q_thermal",
        "FR_E1m",
    ],
}

PDP_2D_PAIRS = {
    "LOI": [
        ("Flame_retardant_AdditionAmount(wt%)", "CURING_MaxaasC"),
        ("Flame_retardant_AdditionAmount(wt%)", "FR_Eta_betaP_A"),
    ],
    "UL94": [
        ("Flame_retardant_AdditionAmount(wt%)", "CURING_Mor30p"),
        ("Flame_retardant_AdditionAmount(wt%)", "EP_wt_fraction"),
    ],
    "Tg": [
        ("EEW", "T_max"),
        ("EEW", "EP_wt_fraction"),
    ],
    "TENSILE": [
        ("Q_thermal", "FR_E1m"),
        ("Q_thermal", "FR_TPSA_efficiency"),
    ],
}


def load_img(target, stem, fallback_stem=None):
    """Load PNG image for target+stem, with optional fallback."""
    for s in [stem] + ([fallback_stem] if fallback_stem else []):
        p = SRC / f"{target}_{s}.png"
        if p.exists():
            return mpimg.imread(p)
    return None


def make_figure5_per_target():
    """Figure 5: Beeswarm + 2×1D PDP + 2×2D PDP per target."""
    for target, display in TARGETS:
        print(f"  Figure5 for {display} ...")
        img_beeswarm = load_img(target, "6_SHAP_Beeswarm")
        if img_beeswarm is None:
            print(f"    SKIP: no beeswarm for {target}")
            continue

        pdp_imgs = []
        for feat in PDP_FEATURES.get(target, []):
            safe = feat.replace('/', '_').replace('\\', '_')
            img = load_img(target, f"2_PDP_{feat}", f"2_PDP_{safe}")
            if img is not None:
                pdp_imgs.append((img, feat))
        for f in os.listdir(SRC):
            if f.startswith(f"{target}_2_PDP_") and f.endswith(".png"):
                # fill remaining slots
                pass

        dp2_imgs = []
        for (f1, f2) in PDP_2D_PAIRS.get(target, []):
            safe_pair = f"{f1}_vs_{f2}".replace('/', '_').replace('\\', '_')
            img = load_img(target, f"8_2D_PDP_{f1}_vs_{f2}", f"8_2D_PDP_{safe_pair}")
            if img is not None:
                dp2_imgs.append((img, f"{f1} vs {f2}"))

        # Layout: 3 rows
        # Row 1: beeswarm (full width)
        # Row 2: PDP1, PDP2, 2D_PDP1
        # Row 3: 2D_PDP2 (full width or with another)
        fig = plt.figure(figsize=(18, 14), dpi=300)
        gs = GridSpec(3, 3, figure=fig, height_ratios=[1.2, 1.0, 0.9], hspace=0.35, wspace=0.30)

        # Row 1: beeswarm spans all 3 cols
        ax_a = fig.add_subplot(gs[0, :])
        ax_a.imshow(img_beeswarm)
        ax_a.axis("off")
        ax_a.text(-0.02, 1.02, "a", transform=ax_a.transAxes, fontsize=20, fontweight="bold", color=INK)

        # Row 2: PDP1, PDP2, 2D_PDP1
        for j, (img, label) in enumerate((pdp_imgs[:2] + dp2_imgs[:1])[:3]):
            ax = fig.add_subplot(gs[1, j])
            if img is not None:
                ax.imshow(img)
            ax.axis("off")
            letter = chr(ord("b") + j)
            ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontsize=20, fontweight="bold", color=INK)

        # Row 3: remaining 2D PDP
        if len(dp2_imgs) >= 2:
            ax = fig.add_subplot(gs[2, :])
            ax.imshow(dp2_imgs[1][0])
            ax.axis("off")
            ax.text(-0.02, 1.02, "e", transform=ax.transAxes, fontsize=20, fontweight="bold", color=INK)

        stem = f"Composite_Figure5_{display.replace('-', '')}"
        fig.savefig(DST / f"{stem}.png", dpi=400, bbox_inches="tight")
        fig.savefig(DST / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved {stem}")


def make_figure6_per_target():
    """Figure 6: Waterfall ×3 + DecisionPath per target."""
    for target, display in TARGETS:
        print(f"  Figure6 for {display} ...")
        # Waterfall images
        wf_types = [
            ("Highest", display.replace("-", "").upper() if target != "UL94" else "True_Positive"),
            ("Best_Prediction", "Best_Prediction"),
            ("Worst_Prediction", "Worst_Prediction"),
        ]
        if target == "UL94":
            wf_types = [
                ("True_Positive", "True_Positive"),
                ("False_Positive", "False_Positive"),
                ("Best_Prediction", "True_Positive"),
            ]

        wf_imgs = []
        for wf_key, fallback in wf_types:
            # e.g. LOI_4_Waterfall_Highest_LOI_Top1
            stem_candidates = [
                f"4_Waterfall_{wf_key}_{target}_Top1",
                f"4_Waterfall_{wf_key}_{display}_Top1",
            ]
            if target == "UL94":
                stem_candidates = [
                    f"4_Waterfall_{wf_key}_Top1",
                ]
            img = None
            for s in stem_candidates:
                p = SRC / f"{target}_{s}.png"
                if p.exists():
                    img = mpimg.imread(p)
                    break
            if img is None:
                # try generic search
                for f in sorted(os.listdir(SRC)):
                    if f.startswith(f"{target}_4_Waterfall_") and wf_key.lower() in f.lower() and "_Top1" in f and f.endswith(".png"):
                        img = mpimg.imread(SRC / f)
                        break
            if img is not None:
                wf_imgs.append((img, wf_key))

        # DecisionPath
        dp_img = load_img(target, "DecisionPath")

        if not wf_imgs and dp_img is None:
            print(f"    SKIP: no images for {target}")
            continue

        fig = plt.figure(figsize=(14, 12), dpi=300)
        gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.28)

        letters = ["a", "b", "c", "d"]
        for i, (img, label) in enumerate(wf_imgs[:3]):
            ax = fig.add_subplot(gs[i // 2, i % 2])
            ax.imshow(img)
            ax.axis("off")
            ax.text(-0.02, 1.02, letters[i], transform=ax.transAxes, fontsize=20, fontweight="bold", color=INK)

        if dp_img is not None:
            # Place decision path in the last available slot
            used = len(wf_imgs)
            ax = None
            if used <= 3:
                row, col = used // 2, used % 2
                ax = fig.add_subplot(gs[row, col])
            if ax is not None:
                ax.imshow(dp_img)
                ax.axis("off")
                ax.text(-0.02, 1.02, letters[min(3, used)], transform=ax.transAxes, fontsize=20, fontweight="bold", color=INK)

        stem = f"Composite_Figure6_{display.replace('-', '')}"
        fig.savefig(DST / f"{stem}.png", dpi=400, bbox_inches="tight")
        fig.savefig(DST / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved {stem}")


if __name__ == "__main__":
    print("Generating composite Figure 5 ...")
    make_figure5_per_target()
    print("Generating composite Figure 6 ...")
    make_figure6_per_target()
    print("All done.")
