# Results Directory

This directory contains benchmark outputs, model performance summaries, and SHAP interpretation data from the study.

## Structure

```
results/
├── benchmark_performance/       # Aggregated 583-model performance across all tasks
│   ├── all_model_performance_merged.csv  # Complete performance table
│   ├── best_by_family.csv                # Best model per algorithm family
│   ├── model_family_manifest.csv         # Model-to-family mapping
│   └── tabicl_superiority_table.csv      # TabICL vs. competitors summary
├── unified_benchmark/           # Descriptor-based model benchmark (14+ models per task)
│   ├── *_summary.csv            # Mean ± std metrics across 5 folds
│   ├── *_all_folds.csv          # Per-fold breakdown
│   └── *_skipped_models.csv     # Models excluded (convergence issues)
├── gnn_fusion_benchmark/        # GNN × fusion-strategy benchmark summaries
│   └── *_summary.csv            # Per-task GNN-fusion results
├── from_scratch/                # From-scratch model comparison (no feature pre-selection)
│   └── *_summary.csv            # Comparative results
├── experiment_predictions/      # External validation on 34 new formulations
│   ├── experiment_predictions.csv        # Forward predictions
│   ├── experiment_validation_full.csv    # Full validation with AD metrics
│   ├── finetune_results.csv              # After incorporating external labels
│   └── ad_expansion_analysis.csv         # Applicability domain expansion
└── shap_interpretation/         # SHAP feature importance per task
    ├── loi/                     # LOI SHAP data
    ├── tg/                      # Tg SHAP data
    ├── tensile/                 # Tensile SHAP data
    └── ul94/                    # UL-94 SHAP data
```

Each SHAP subfolder contains:
- `X_train.csv`, `X_test.csv` — Feature matrices
- `y_train.csv`, `y_test.csv` — Target values
- `predictions.csv` — Model predictions
- `shap_values.csv` — SHAP values
- `feature_importance.csv` — Ranked importance
- `metrics.json` — Performance metrics

## Usage

Plotting scripts read these files to reproduce the manuscript figures. See `plotting_code/` and `FIGURE_CODE_MAP.md` in the repository root for the script-to-figure mapping.
