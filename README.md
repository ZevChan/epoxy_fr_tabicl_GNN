# Explicit Descriptor–Process Learning for Flame-Retardant Epoxy Resin Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository contains the complete code, datasets, and benchmark results for the paper:

> **Explicit Descriptor–Process Learning for Flame Retardancy Epoxy Resin Prediction under Data-Limited Conditions**

*Chen Z., Yang X., Tang C., Zeng X., Zang H., Ying Z., Xiang L., Guo Y., Liu R., Jiang J., Yu Y.*

## Overview

A descriptor–process–tabular in-context learning framework that integrates molecular descriptors, formulation variables, and curing thermal histories into an auditable tabular representation. Across **583 heterogeneous models**, explicit tabular learners (represented by **TabICL**) define a higher and more stable performance envelope than pure graph neural networks and graph–descriptor fusion models for predicting:

- **LOI** (Limiting Oxygen Index) — regression
- **UL-94** (Vertical Burning Rating) — classification  
- **Tg** (Glass Transition Temperature) — regression
- **Tensile Strength** — regression

## Repository Structure

```
├── data/                           # All datasets (included)
│   ├── task_datasets/              # LOI, Tg, Tensile, UL-94 task-specific CSVs
│   ├── experiment/                 # 34 external experimental validation formulations
│   ├── smiles/                     # EP, FR, and Curing agent SMILES files
├── results/                        # Benchmark outputs and SHAP interpretation (included)
│   ├── benchmark_performance/      # Aggregated 583-model performance summary
│   ├── unified_benchmark/          # Per-task unified benchmark (14 descriptor models)
│   ├── gnn_fusion_benchmark/       # GNN × fusion-strategy benchmark summaries
│   ├── from_scratch/               # From-scratch model comparison results
│   ├── experiment_predictions/     # External validation predictions and AD analysis
│   └── shap_interpretation/        # SHAP feature importance per task (LOI/Tg/Tensile/UL94)
├── training_code/
│   ├── task_models/                # Task-specific training for LOI, Tg, Tensile, UL-94
│   │   ├── loi/                    # Descriptor-only, GNN, System-FiLM, TabICL, Optuna HPO
│   │   ├── tg/
│   │   ├── tensile/
│   │   └── ul94/
│   ├── benchmark_workflows/        # Unified benchmark, GNN-fusion, fine-tuning, validation
│   ├── screening_workflows/        # PubChem screening and candidate prediction
│   └── external_validation/        # External experiment prediction entry point
├── plotting_code/
│   ├── main_and_si_figures/        # Manuscript Figures 2–7 and SI composite panels
│   ├── task_level_plots/           # Per-task diagnostic and representative-fold plots
│   └── supplementary_figures/      # SHAP, benchmark, baseline, and diagnostic plots
├── requirements.txt                # Python dependencies
├── CODE_STRUCTURE.md               # Detailed code layout
├── DATA_AVAILABILITY.md            # Dataset descriptions
├── FIGURE_CODE_MAP.md              # Figure-to-script mapping
└── .gitignore
```

## Quick Start

### Environment

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Reproduce Training

```powershell
# Train TabICL on LOI task
python training_code\task_models\loi\tabicl.py

# Run full unified benchmark (14 models × 4 tasks)
python training_code\benchmark_workflows\run_unified_benchmark.py

# Run GNN-fusion benchmark (5 GNNs × 5 fusion strategies)
python training_code\benchmark_workflows\run_gnn_fusion_benchmark.py
```

### Reproduce Figures

```powershell
# Figure 2: Benchmark comparison
python plotting_code\main_and_si_figures\make_figure2_unified_benchmark.py

# Figure 3: Reliability diagnostics
python plotting_code\main_and_si_figures\make_figure3_reliability_diagnostics.py

# SHAP mechanism figures
python plotting_code\main_and_si_figures\make_figure4_shap_mechanisms.py
```

Training and plotting scripts read data from `data/` and `results/` folders. Adjust path constants at the top of each script if needed.

## Model Families Tested

| Family | Models |
|--------|--------|
| **Linear / Ridge** | Ridge Regression, Elastic Net, Bayesian Ridge |
| **Support Vector** | SVR (RBF kernel) |
| **Tree Ensemble** | Random Forest, Extra Trees, Gradient Boosting, XGBoost, LightGBM, CatBoost |
| **Neural Network** | Multi-Layer Perceptron |
| **In-Context Learning** | TabPFN, **TabICL** |
| **Graph Neural Network** | GCN, GraphSAGE, GAT, GIN, GINE |
| **Graph–Descriptor Fusion** | System-FiLM-Embedding, Concat, Weighted-Sum, Gated, Attention |

## Data Description

All datasets are included in the `data/` directory. See `DATA_AVAILABILITY.md` for detailed descriptions.

## Citation

If you use this code or data, please cite the corresponding paper.

## License

MIT License — see [LICENSE](LICENSE) file for details.
