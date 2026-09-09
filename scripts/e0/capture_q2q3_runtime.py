"""Capture the pinned Q2/Q3 qualification runtime environment.

Records the exact fields listed in Q2_Q3_EXECUTION_RELEASE.md and fails
(exit 1) if CUDA is unavailable or BF16 is unsupported — per the release,
no silent FP16 substitution is permitted.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = (Path(__file__).resolve().parent.parent.parent).resolve()
OUT_PATH = REPO_ROOT / "docs" / "experiments" / "e0" / "q2_q3_runtime.snapshot.json"

PROBE = r"""
import json
import torch
import transformers
import accelerate
import safetensors

info = {
    "torch_version": torch.__version__,
    "torch_version_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    "cuda_device_name_0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "cuda_total_memory_0": (
        torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None
    ),
    "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
    "transformers_version": transformers.__version__,
    "accelerate_version": accelerate.__version__,
    "safetensors_version": safetensors.__version__,
}
print(json.dumps(info))
"""


def main() -> None:
    import torch

    result = subprocess.run(
        [sys.executable, "-c", PROBE],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    runtime = json.loads(result.stdout.strip().splitlines()[-1])

    pins = {}
    for package in ["torch", "transformers", "accelerate", "safetensors", "numpy", "scikit-learn", "pyyaml"]:
        shown = subprocess.run(
            [sys.executable, "-m", "pip", "show", package],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        version = next(
            line.split(":", 1)[1].strip() for line in shown.splitlines() if line.startswith("Version:")
        )
        pins[package] = version

    snapshot = {
        "schema_id": "E0-Q2Q3-RUNTIME-SNAPSHOT-v0",
        "captured_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "release_reference": "docs/experiments/e0/Q2_Q3_EXECUTION_RELEASE.md @ branch e0/qualification-bootstrap",
        "runtime": runtime,
        "pinned_packages": pins,
        "gates": {
            "cuda_available": runtime["cuda_available"],
            "bf16_supported": runtime["bf16_supported"],
            "cuda_runtime": runtime["torch_version_cuda"],
        },
    }

    failures = []
    if not runtime["cuda_available"]:
        failures.append("CUDA is unavailable — frozen contract requires CUDA.")
    if not runtime["bf16_supported"]:
        failures.append("BF16 unsupported on device 0 — frozen S0 BF16 contract cannot run.")
    if runtime["torch_version_cuda"] is None:
        failures.append("torch.version.cuda is None (CPU-only build installed).")
    snapshot["gates"]["status"] = "FAIL" if failures else "PASS"
    snapshot["gates"]["failures"] = failures

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(snapshot, indent=2))

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
