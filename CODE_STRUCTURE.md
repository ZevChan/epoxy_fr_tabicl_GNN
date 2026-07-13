# Code Structure

## Training Code

- `training_code/task_models/loi/`: LOI model training and interpretation workflows
- `training_code/task_models/tg/`: Tg model training and interpretation workflows
- `training_code/task_models/tensile/`: tensile-strength model training and interpretation workflows
- `training_code/task_models/ul94/`: UL-94 classification model training and interpretation workflows
- `training_code/benchmark_workflows/`: unified benchmark, graph-fusion benchmark, fine-tuning, and validation workflows
- `training_code/screening_workflows/`: candidate screening and PubChem prediction workflows
- `training_code/external_validation/`: external experimental validation prediction workflow

## Plotting Code

- `plotting_code/main_and_si_figures/`: composite manuscript and SI figure scripts
- `plotting_code/task_level_plots/`: task-level diagnostic and representative-fold plots
- `plotting_code/supplementary_figures/`: supplementary SHAP, benchmark, baseline, and diagnostic figure scripts

## Data (`data/`)

- `data/task_datasets/`: LOI, Tg, tensile-strength, and UL-94 task-specific datasets (CSV)
- `data/experiment/`: 34 external experimental validation formulations, descriptors, and actual measurements
- `data/smiles/`: EP, FR, and curing agent SMILES files

## Results (`results/`)

- `results/benchmark_performance/`: aggregated 583-model performance across all tasks
- `results/unified_benchmark/`: per-task unified benchmark (14 descriptor models × 5 folds)
- `results/gnn_fusion_benchmark/`: GNN × fusion-strategy benchmark summaries
- `results/from_scratch/`: from-scratch model comparison results
- `results/experiment_predictions/`: external validation predictions, AD analysis, and fine-tuning results
- `results/shap_interpretation/`: SHAP feature importance and prediction data per task (loi/tg/tensile/ul94)

## Running the Workflows

All scripts expect the `data/` and `results/` folders at the repository root. Path constants are defined at the top of each script — adjust them if you reorganize the directory structure.
