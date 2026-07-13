$ErrorActionPreference = "Stop"

$Python = "E:\ProgramData\miniconda3\envs\DFT_FR_GNN_transformer\python.exe"
$Targets = @("LOI", "Tg", "TENSILE", "UL94")

Write-Host "Using Python: $Python"
Write-Host "Running all benchmarks from scratch. Existing result summaries are ignored."

foreach ($target in $Targets) {
    Write-Host "`n=== From scratch descriptor/process benchmark: $target ==="
    & $Python "$PSScriptRoot\run_unified_benchmark.py" --target $target
}

foreach ($target in $Targets) {
    Write-Host "`n=== From scratch GNN/fusion benchmark: $target ==="
    & $Python "$PSScriptRoot\run_gnn_fusion_benchmark.py" --target $target
}

Write-Host "`n=== Redrawing figures from from-scratch outputs ==="
& $Python "$PSScriptRoot\plot_detailed_method_figures.py"

Write-Host "`nAll from-scratch benchmarks finished. Outputs are in $PSScriptRoot\outputs"
