@echo off
cd /d C:\Users\WINDOWS\Desktop\GNN\TabICL_superiority_benchmark
set PYTHON=E:\ProgramData\miniconda3\envs\DFT_FR_GNN_transformer\python.exe
set OUTDIR=outputs

echo [%date% %time%] Starting Tg... >> %OUTDIR%\chem_balanced_Tg_runlog.txt
%PYTHON% -W ignore::UserWarning -u run_gnn_fusion_benchmark.py --target Tg --convs gcn sage gat gin gine --fusions concat weighted_sum gated attention film --chem-presets balanced wide deep >> %OUTDIR%\chem_balanced_Tg_runlog.txt 2>&1
echo [%date% %time%] Tg DONE >> %OUTDIR%\chem_balanced_Tg_runlog.txt

echo [%date% %time%] Starting TENSILE... >> %OUTDIR%\chem_balanced_Tg_runlog.txt
%PYTHON% -W ignore::UserWarning -u run_gnn_fusion_benchmark.py --target TENSILE --convs gcn sage gat gin gine --fusions concat weighted_sum gated attention film --chem-presets balanced wide deep >> %OUTDIR%\chem_balanced_Tg_runlog.txt 2>&1
echo [%date% %time%] TENSILE DONE >> %OUTDIR%\chem_balanced_Tg_runlog.txt

echo [%date% %time%] Starting UL94... >> %OUTDIR%\chem_balanced_Tg_runlog.txt
%PYTHON% -W ignore::UserWarning -u run_gnn_fusion_benchmark.py --target UL94 --convs gcn sage gat gin gine --fusions concat weighted_sum gated attention film --chem-presets balanced wide deep >> %OUTDIR%\chem_balanced_Tg_runlog.txt 2>&1
echo [%date% %time%] UL94 DONE >> %OUTDIR%\chem_balanced_Tg_runlog.txt

echo [%date% %time%] ALL DONE >> %OUTDIR%\chem_balanced_Tg_runlog.txt
