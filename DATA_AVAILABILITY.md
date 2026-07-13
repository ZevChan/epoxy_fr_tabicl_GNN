# Data Availability

The datasets, descriptor tables, and intermediate result tables required to rerun training and reproduce the figures are **included** in this repository under the `data/` and `results/` directories.

## Included Datasets (`data/`)

### Task Datasets (`data/task_datasets/`)

Four task-specific CSV files, each containing:
- SMILES strings for EP monomer, FR, and Curing Agent
- AlvaDesc molecular descriptors (~5,000 columns per component) prefixed with `EP_`, `FR_`, `CURING_`
- Formulation features (mass fractions, EEW)
- Process features (multi-stage curing temperatures and times)
- Target property values

| File | Task | Samples | Size |
|------|------|---------|------|
| `loi_dataset.csv` | LOI (regression) | ~500 | 36 MB |
| `tg_dataset.csv` | Tg (regression) | ~400 | 29 MB |
| `tensile_dataset.csv` | Tensile Strength (regression) | ~250 | 14 MB |
| `ul94_dataset.csv` | UL-94 (classification) | ~580 | 34 MB |

The sample counts differ because each literature entry reports a different subset of properties.

### Experiment Data (`data/experiment/`)

External experimental validation data for 34 newly formulated flame-retardant epoxy systems:

| File | Description |
|------|-------------|
| `experiment_formulations.csv` | Formulation table with SMILES, EEW, wt%, curing schedule |
| `experiment_actual_data.xlsx` | Measured LOI, Tg, Tensile, UL-94 values |
| `experiment_ep_descriptors.csv` | AlvaDesc descriptors for EP monomers |
| `experiment_fr_descriptors.csv` | AlvaDesc descriptors for flame retardants |
| `experiment_curing_descriptors.csv` | AlvaDesc descriptors for curing agents |
| `experiment_smiles.txt` | SMILES strings list |
| `experiment_smiles_d.txt` | SMILES with computed descriptors |

### SMILES Files (`data/smiles/`)

| File | Description |
|------|-------------|
| `ep_smiles.txt` | Epoxy resin SMILES (~400 unique) |
| `fr_smiles.txt` | Flame retardant SMILES (~900 unique) |
| `curing_smiles.txt` | Curing agent SMILES (~280 unique) |

## Included Results (`results/`)

### Benchmark Performance (`results/benchmark_performance/`)

| File | Description |
|------|-------------|
| `all_model_performance_merged.csv` | Complete 583-model performance across all 4 tasks |
| `best_by_family.csv` | Best model per algorithm family per task |
| `model_family_manifest.csv` | Model-to-family mapping |
| `tabicl_superiority_table.csv` | TabICL vs. best competitors summary |

### Unified Benchmark (`results/unified_benchmark/`)

Per-task results for 14 descriptor-based models (Ridge, ElasticNet, SVR, RF, XGBoost, LightGBM, CatBoost, MLP, TabPFN, TabICL, etc.):
- `*_summary.csv` — mean ± std metrics across 5 folds
- `*_all_folds.csv` — per-fold breakdown
- `*_skipped_models.csv` — models excluded due to convergence issues

### GNN-Fusion Benchmark (`results/gnn_fusion_benchmark/`)

Per-task results for 5 GNN architectures × fusion strategies:
- `*_summary.csv` — full run summaries
- `*_quick_summary.csv` — quick-run summaries

### From-Scratch Benchmark (`results/from_scratch/`)

Comparison of models trained from scratch (no feature pre-selection):
- `from_scratch_all_model_summary.csv`
- `from_scratch_best_by_method_group.csv`

### Experiment Predictions (`results/experiment_predictions/`)

| File | Description |
|------|-------------|
| `experiment_predictions.csv` | Forward predictions on 34 external samples |
| `experiment_validation_full.csv` | Full validation including AD metrics |
| `finetune_results.csv` | Fine-tuning results after incorporating external labels |
| `ad_expansion_analysis.csv` | Applicability domain expansion analysis |

### SHAP Interpretation (`results/shap_interpretation/`)

Per-task folders (`loi/`, `tg/`, `tensile/`, `ul94/`) containing:
- `X_train.csv`, `X_test.csv` — train/test feature matrices
- `y_train.csv`, `y_test.csv` — train/test target values
- `predictions.csv` — model predictions
- `shap_values.csv` — SHAP values for all features
- `feature_importance.csv` — ranked feature importance
- `metrics.json` — performance metrics

## Using the Scripts

Training and plotting scripts read data from the `data/` and `results/` folders. If you place the repository elsewhere, update the path constants at the top of each script.

```python
# Example path configuration in a training script:
DATA_DIR = 'data/task_datasets'
RESULTS_DIR = 'results'
```

## Data Sources

- Literature data curated from publicly available flame-retardant epoxy resin studies
- Molecular descriptors computed with AlvaDesc (https://www.alvascience.com/alvadesc/)
- Experimental validation data acquired by the authors
