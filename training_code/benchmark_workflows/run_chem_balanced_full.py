from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

from run_gnn_fusion_benchmark import run


HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

TARGETS = ["LOI", "Tg", "TENSILE", "UL94"]
CONVS = ["gcn", "sage", "gat", "gin", "gine"]
FUSIONS = ["concat", "weighted_sum", "gated", "attention", "film"]
PRESETS = ["balanced", "wide", "deep"]


def write_status(status: dict) -> None:
    status_path = OUT / "chem_balanced_full_status.json"
    status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    status = {
        "pid": os.getpid(),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "targets": TARGETS,
        "convs": CONVS,
        "fusions": FUSIONS,
        "presets": PRESETS,
        "per_target_model_count": len(CONVS) * len(FUSIONS) * len(PRESETS),
        "total_model_count": len(CONVS) * len(FUSIONS) * len(PRESETS) * len(TARGETS),
        "state": "running",
        "current_target": None,
    }
    write_status(status)
    try:
        for target in TARGETS:
            status["current_target"] = target
            status["target_started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            write_status(status)
            run(
                target=target,
                quick=False,
                convs=CONVS,
                fusions=FUSIONS,
                chem_presets=PRESETS,
            )
        status["state"] = "complete"
        status["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        status["current_target"] = None
        write_status(status)
    except Exception as exc:
        status["state"] = "failed"
        status["failed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        status["error"] = repr(exc)
        status["traceback"] = traceback.format_exc()
        write_status(status)
        raise


if __name__ == "__main__":
    main()
