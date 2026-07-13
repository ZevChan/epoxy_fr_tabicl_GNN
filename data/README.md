# Data Directory

This directory contains all datasets used in the study. The datasets are organized by type.

## Structure

```
data/
├── task_datasets/        # Per-task datasets with SMILES, descriptors, process params, and targets
│   ├── loi_dataset.csv       # LOI (Limiting Oxygen Index) regression dataset
│   ├── tg_dataset.csv        # Tg (Glass Transition Temperature) regression dataset
│   ├── tensile_dataset.csv   # Tensile Strength regression dataset
│   └── ul94_dataset.csv      # UL-94 Vertical Burning classification dataset
├── experiment/           # External experimental validation data (34 new formulations)
│   ├── experiment_formulations.csv     # Formulation table (SMILES, EEW, wt%, curing schedule)
│   ├── experiment_actual_data.xlsx     # Measured experimental values
│   ├── experiment_ep_descriptors.csv   # AlvaDesc descriptors for EP monomers
│   ├── experiment_fr_descriptors.csv   # AlvaDesc descriptors for flame retardants
│   ├── experiment_curing_descriptors.csv # AlvaDesc descriptors for curing agents
│   ├── experiment_smiles.txt           # SMILES list
│   └── experiment_smiles_d.txt         # SMILES with computed descriptors
├── smiles/               # SMILES files for each component type
│   ├── ep_smiles.txt         # Epoxy resin SMILES
│   ├── fr_smiles.txt         # Flame retardant SMILES
│   └── curing_smiles.txt     # Curing agent SMILES
```

## Dataset Format

Each task dataset CSV contains:
- **SMILES columns**: `EP_SMILES`, `FR_SMILES`, `CURING_SMILES`
- **Descriptor columns**: `EP_*`, `FR_*`, `CURING_*` (AlvaDesc molecular descriptors with component prefix)
- **Formulation columns**: `EP_wt_fraction`, `FR_wt_fraction`, `CURING_wt_fraction`, `EEW`, `Flame_retardant_AdditionAmount(wt%)`, `Curing_agent_AdditionAmount(wt%)`
- **Process columns**: `T_max`, `t_total`, `Q_thermal` (derived from multi-stage curing parameters)
- **Target columns**: `LOI`, `Tg`, `Tensile`, `UL94` (task-dependent)

## Data Sources

- Literature data curated from publicly available flame-retardant epoxy resin studies
- Molecular descriptors computed with AlvaDesc (https://www.alvascience.com/alvadesc/)
- External experimental validation data acquired by the authors
- SMILES strings from literature and PubChem database

## Usage

Training scripts expect these files in the `data/` directory. See `training_code/` for model training workflows and `DATA_AVAILABILITY.md` in the repository root for detailed file descriptions.
