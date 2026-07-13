from __future__ import annotations

from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "\u6295\u7a3f\u56fe\u7247"
OUT = FIG_DIR / "\u5355\u72ec\u5b50\u56fe"
OUT.mkdir(parents=True, exist_ok=True)


def crop_fraction(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
    w, h = image.size
    left, top, right, bottom = box
    return image.crop((round(left * w), round(top * h), round(right * w), round(bottom * h)))


def save_panel(image: Image.Image, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    image.save(png)
    image.convert("RGB").save(pdf, "PDF", resolution=600.0)


def export_panels(figure_stem: str, panels: dict[str, tuple[float, float, float, float]]) -> None:
    image = Image.open(FIG_DIR / f"{figure_stem}.png")
    out_dir = OUT / figure_stem
    for label, box in panels.items():
        save_panel(crop_fraction(image, box), out_dir, f"{figure_stem}_{label}")


def main() -> None:
    export_panels(
        "Figure2_unified_benchmark",
        {
            "a_Best_performance_by_method_group": (0.00, 0.00, 0.47, 0.50),
            "b_TabICL_vs_strongest_graph_fusion": (0.47, 0.00, 1.00, 0.50),
            "c_Individual_model_ranking_heatmap": (0.00, 0.50, 0.32, 1.00),
            "d_Graph_fusion_benchmark_landscape": (0.32, 0.50, 0.61, 1.00),
            "e_Task_predictability_cards": (0.61, 0.50, 1.00, 1.00),
        },
    )

    export_panels(
        "Figure3_reliability_diagnostics",
        {
            "a_LOI_parity": (0.00, 0.00, 0.25, 0.50),
            "b_Tg_parity": (0.25, 0.00, 0.50, 0.50),
            "c_Tensile_parity": (0.50, 0.00, 0.75, 0.50),
            "d_Error_CDF": (0.75, 0.00, 1.00, 0.50),
            "e_UL94_calibration": (0.00, 0.50, 0.25, 1.00),
            "f_UL94_threshold_errors": (0.25, 0.50, 0.50, 1.00),
            "g_Residual_diagnostics": (0.50, 0.50, 0.75, 1.00),
            "h_Applicability_domain": (0.75, 0.50, 1.00, 1.00),
        },
    )

    export_panels(
        "Figure4_SHAP_mechanisms",
        {
            "a_Cross_task_SHAP_heatmap": (0.00, 0.00, 0.41, 0.51),
            "b_LOI_mechanism_attribution": (0.41, 0.00, 0.705, 0.51),
            "c_UL94_mechanism_attribution": (0.705, 0.00, 1.00, 0.51),
            "d_Tg_mechanism_attribution": (0.00, 0.51, 0.41, 1.00),
            "e_Tensile_mechanism_attribution": (0.41, 0.51, 0.705, 1.00),
            "f_Mechanism_summary_map": (0.705, 0.51, 1.00, 1.00),
        },
    )

    export_panels(
        "Figure5_SHAP_dependencies_boundaries",
        {
            "a_LOI_FR_loading_dependence": (0.00, 0.00, 0.50, 0.25),
            "b_LOI_interaction_landscape": (0.50, 0.00, 1.00, 0.25),
            "c_UL94_conjugation_dependence": (0.00, 0.25, 0.50, 0.50),
            "d_UL94_interaction_landscape": (0.50, 0.25, 1.00, 0.50),
            "e_Tg_EEW_dependence": (0.00, 0.50, 0.50, 0.75),
            "f_Tg_interaction_landscape": (0.50, 0.50, 1.00, 0.75),
            "g_Tensile_Qthermal_dependence": (0.00, 0.75, 0.50, 1.00),
            "h_Tensile_interaction_landscape": (0.50, 0.75, 1.00, 1.00),
        },
    )

    print(f"Exported individual panels to {OUT}")


if __name__ == "__main__":
    main()
