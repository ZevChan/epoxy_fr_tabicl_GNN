# Figure Code Map

This map provides a high-level guide to the plotting-code groups.

## Manuscript Figures

- Figure 2: `plotting_code/main_and_si_figures/make_figure2_unified_benchmark.py`, `make_figure3_reliability_diagnostics.py`, and related diagnostic scripts
- Figure 3: benchmark-comparison scripts in `plotting_code/main_and_si_figures/`
- Figure 4: robustness, calibration, and residual scripts in `plotting_code/main_and_si_figures/`
- Figure 5: SHAP mechanism and dependency scripts in `plotting_code/main_and_si_figures/`
- Figure 6: local SHAP and composite scripts in `plotting_code/main_and_si_figures/`
- Figure 7: external-validation plotting scripts in `plotting_code/main_and_si_figures/`

Figure 1 was prepared as a conceptual schematic and is not represented as a Python plotting script in this code-only package.

## Supporting-Information Figures

- Correlation, overlap, representative-fold, and task-level diagnostic plots: `plotting_code/task_level_plots/`
- SHAP response, interaction, quantile, waterfall, and supplementary diagnostic plots: `plotting_code/supplementary_figures/`
- Benchmark and baseline comparison panels: `plotting_code/supplementary_figures/`

Some SI chemical-structure panels were prepared outside the Python plotting workflow. They are not included here because this package contains code only and no rendered image assets.
