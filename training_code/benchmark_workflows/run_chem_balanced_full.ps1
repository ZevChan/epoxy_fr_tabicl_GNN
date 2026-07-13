$ErrorActionPreference = "Stop"

$PythonExe = "E:\ProgramData\miniconda3\envs\DFT_FR_GNN_transformer\python.exe"
$Script = Join-Path $PSScriptRoot "run_gnn_fusion_benchmark.py"
$Log = Join-Path $PSScriptRoot "outputs\chem_balanced_full_transcript.log"
Start-Transcript -Path $Log -Append

$targets = @("LOI", "Tg", "TENSILE", "UL94")

try {
    foreach ($target in $targets) {
        Write-Host "=== Running chem-balanced full GNN/fusion grid for $target ==="
        & $PythonExe -W ignore::UserWarning $Script `
            --target $target `
            --convs gcn sage gat gin gine `
            --fusions concat weighted_sum gated attention film `
            --chem-presets balanced wide deep
    }
}
finally {
    Stop-Transcript
}
