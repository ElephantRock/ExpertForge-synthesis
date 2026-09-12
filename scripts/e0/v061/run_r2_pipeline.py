"""Driver for the r2 preflight pipeline (correctness -> depletion probe).

Runs detached; writes a machine-readable completion record next to the
evidence files. Correctness uses n=500 families/depth/surface with ALL-family
renaming + order-permutation invariance and the independent GI check on
every repeated group (4,000 families fully tested — 167x the rejected r1
:families[:3] coverage). Depletion measures the ID 32k/depth discovery curve
and fresh-namespace disjoint acceptance at 2,000 families/depth/surface.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent.parent  # v061 -> e0 -> scripts -> repo root
PY = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
EVIDENCE = REPO_ROOT / "docs" / "experiments" / "e0" / "v061"


def run(mode: str, extra: list[str]) -> dict:
    cmd = [str(PY), "-u", str(HERE / "run_structural_preflight_r2.py"), "--mode", mode] + extra
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "mode": mode,
        "extra": extra,
        "exit": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-2000:],
        "minutes": round((time.time() - t0) / 60, 1),
    }


def main() -> None:
    started = time.time()
    log = {"schema_id": "E0-V061-R2-PIPELINE-LOG-v0", "stages": []}

    correctness = run("correctness", ["--families-per-depth", "500"])
    log["stages"].append(correctness)
    (EVIDENCE / "R2_PIPELINE_PARTIAL.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    if correctness["exit"] != 0:
        log["pipeline_status"] = "CORRECTNESS_FAILED_STOPPED"
        (EVIDENCE / "R2_PIPELINE_LOG.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
        raise SystemExit(1)

    depletion = run("depletion", ["--discovery-per-depth", "32000", "--fresh-request", "2000"])
    log["stages"].append(depletion)
    log["pipeline_status"] = "COMPLETED" if depletion["exit"] == 0 else "DEPLETION_FAILED"
    log["wall_minutes"] = round((time.time() - started) / 60, 1)
    (EVIDENCE / "R2_PIPELINE_LOG.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    raise SystemExit(0 if depletion["exit"] == 0 else 1)


if __name__ == "__main__":
    main()
