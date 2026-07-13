$ErrorActionPreference = "Stop"

$Python = "E:\ProgramData\miniconda3\envs\DFT_FR_GNN_transformer\python.exe"
$Root = Split-Path -Parent $PSScriptRoot

Write-Host "Using Python: $Python"
& $Python "$PSScriptRoot\collect_existing_performance.py"

foreach ($target in @("LOI", "Tg", "TENSILE", "UL94")) {
    Write-Host "`n=== Descriptor benchmark: $target ==="
    & $Python "$PSScriptRoot\run_unified_benchmark.py" --target $target --quick
}

foreach ($target in @("LOI", "Tg", "TENSILE", "UL94")) {
    Write-Host "`n=== GNN/fusion benchmark: $target ==="
    & $Python "$PSScriptRoot\run_gnn_fusion_benchmark.py" --target $target --quick
}

Write-Host "`nAll quick benchmarks finished. Outputs are in $PSScriptRoot\outputs"

