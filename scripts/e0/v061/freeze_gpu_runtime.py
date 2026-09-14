"""V06-GPU-BOOTSTRAP-REMEDIATION-1 (2/5): freeze the GPU execution runtime.

Binds the ACTUAL execution environment for all remediation GPU work and
verifies it IS the frozen pre-execution runtime (repo .venv: Python 3.12.10 /
transformers 5.16.1 / tokenizers 0.23.2 / torch 2.14.0+cu126), superseding the
r0 drift documented in incidents/EXECUTION_RUNTIME_SNAPSHOT_DRIFT.md.

Fail-closed gates:
  G1  interpreter is the repo .venv (r0 drift root cause eliminated)
  G2  P0 snapshot file digests == P0_SNAPSHOT_VERIFICATION.json
  G3  P0 classification-form interface exact (44,670,976 + 1,539, FP32,
      head removed, finite readout) under THIS runtime, both P-FT/P-RANDOM
  G4  complete 402k P0 tokenizer audit within §6.2 bounds (max ≤ 384,
      truncation 0, invalid 0) AND exactly reproducing the prior frozen
      audit values (same environment must reproduce them)
  G5  runtime identity fields exactly match the frozen pre-execution
      snapshot (versions, CUDA/driver/cuDNN, pip-freeze SHA, igraph binary
      SHA); deterministic state QUERIED (not asserted) == preamble contract

Audit and snapshot computations mirror run_preexec_remediation.py bases
exactly (encode add_special_tokens=False; percentiles via sorted[int(n*q)];
igraph hash over _igraph.pyd; pip freeze via sys.executable -m pip).

MUST be executed as:
  .venv/Scripts/python.exe scripts/e0/v061/freeze_gpu_runtime.py
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

REPO_ROOT = HERE.parent.parent.parent
VENV_ROOT = (REPO_ROOT / ".venv").resolve()
CORPUS = REPO_ROOT / "local_data" / "e0_v061_msel"
DOCS = REPO_ROOT / "docs" / "experiments" / "e0" / "v061"
OUTPUT = DOCS / "GPU_RUNTIME_FREEZE.json"
PRIOR_SNAPSHOT_SOURCE = DOCS / "PREEXEC_REMEDIATION_EVIDENCE.json"
P0_VERIFICATION = DOCS / "P0_SNAPSHOT_VERIFICATION.json"
P0_SNAP = Path(
    r"C:\huggingface_cache\hub\models--EleutherAI--pythia-70m\snapshots"
    r"\a39f36b100fe8a5377810d56c3f4789b9c53ac42"
)

SPLITS = ("train_pool", "dev_ID", "eval_ID", "eval_STRUCT")
DEPTHS = (1, 2, 3, 4)
EXPECTED_BACKBONE = 44_670_976
EXPECTED_CLASSIFIER = 1_539
NATIVE_VOCAB = 50304
NATIVE_MAX_LEN = 384
FIRST_PRIMARY_SEED = 806915476


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def record_sha(distribution: str) -> str | None:
    """SHA-256 of the installed distribution's RECORD file (binds every file
    the distribution ships, transitively)."""
    try:
        dist = importlib.metadata.distribution(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None
    # For wheel installs, dist._path is the .dist-info directory and RECORD
    # sits directly inside it.
    cand = Path(str(dist._path)) / "RECORD"
    if cand.is_file():
        return sha256_file(cand)
    return None


def main() -> None:
    started = time.time()
    gates: dict[str, bool] = {}

    # === G1: interpreter identity (r0 drift root cause) ===
    sys_exe = Path(sys.executable).resolve()
    interpreter_in_venv = VENV_ROOT in sys_exe.parents
    gates["G1_interpreter_is_repo_venv"] = interpreter_in_venv
    print(f"G1 interpreter: {sys_exe} | in .venv: {interpreter_in_venv}", flush=True)
    if not interpreter_in_venv:
        raise SystemExit(
            f"FAIL-CLOSED: execute under the frozen runtime: {VENV_ROOT / 'Scripts' / 'python.exe'}"
        )

    from deterministic_preamble import apply as apply_det, query as query_det, verify_applied
    det_state = apply_det()
    det_query = query_det()
    det_verified = verify_applied(det_state)
    gates["G5a_deterministic_preamble_verified"] = bool(det_verified)

    import torch
    import transformers
    import tokenizers
    import numpy
    import scipy
    import sklearn
    import igraph

    # === Section A: runtime identity binding ===
    print("=== A. Runtime identity binding ===", flush=True)
    runtime = {
        "sys_executable": str(sys_exe),
        "os": platform.platform(),
        "python_version": sys.version.split()[0],
        "python_implementation": sys.implementation.name,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "transformers_version": transformers.__version__,
        "tokenizers_version": tokenizers.__version__,
        "numpy_version": numpy.__version__,
        "scipy_version": scipy.__version__,
        "sklearn_version": sklearn.__version__,
        "igraph_version": igraph.__version__,
        "igraph_py_sha256": sha256_file(VENV_ROOT / "Lib" / "site-packages" / "igraph" / "_igraph.pyd"),
        "deterministic_state_queried": det_query,
        "deterministic_state_verified": bool(det_verified),
    }
    if torch.cuda.is_available():
        runtime["cuda_version"] = torch.version.cuda
        runtime["gpu_name"] = torch.cuda.get_device_name(0)
        runtime["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
        runtime["bf16_supported"] = torch.cuda.is_bf16_supported()
        runtime["cudnn_version"] = torch.backends.cudnn.version()
        runtime["cudnn_enabled"] = torch.backends.cudnn.enabled
    nvidia = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True, text=True)
    if nvidia.returncode == 0:
        runtime["nvidia_driver"] = nvidia.stdout.strip()
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    runtime["pip_freeze_text"] = freeze.stdout
    runtime["pip_freeze_sha256"] = sha256_bytes(freeze.stdout.encode())
    runtime["package_record_sha256"] = {
        dist: record_sha(dist) for dist in
        ("torch", "transformers", "tokenizers", "numpy", "scipy",
         "scikit-learn", "igraph", "psutil")
    }
    for k, v in runtime.items():
        if k in ("pip_freeze_text",):
            continue
        print(f"  {k}: {v}", flush=True)

    # === Section B: P0 snapshot file digests (G2) ===
    print("=== B. P0 snapshot digests ===", flush=True)
    prior_p0 = json.loads(P0_VERIFICATION.read_text(encoding="utf-8"))
    frozen_digests = prior_p0["file_digests"]
    digest_check = {}
    for name, expected in frozen_digests.items():
        actual = sha256_file(P0_SNAP / name)
        digest_check[name] = {"expected": expected, "actual": actual, "match": actual == expected}
        print(f"  {name}: {'OK' if actual == expected else 'MISMATCH'}", flush=True)
    gates["G2_p0_snapshot_digests"] = all(v["match"] for v in digest_check.values())

    # === Section C: P0 classification-form interface check (G3) ===
    print("=== C. P0 classification-form interface ===", flush=True)
    from gpu_smoke_harness import build_p0_classification, p0_readout_logits
    interface = {}
    for arm in ("P-FT", "P-RANDOM"):
        bb, cls = build_p0_classification(arm, FIRST_PRIMARY_SEED, torch.device("cpu"))
        bp = sum(p.numel() for p in bb.parameters())
        cp = sum(p.numel() for p in cls.parameters())
        ids = torch.randint(0, NATIVE_VOCAB, (2, 20))
        mask = torch.ones(2, 20, dtype=torch.long)
        mask[1, 15:] = 0
        out = p0_readout_logits(bb, cls, ids, mask)
        dtypes = {p.dtype for p in bb.parameters()} | {p.dtype for p in cls.parameters()}
        interface[arm] = {
            "backbone_params": bp,
            "classifier_params": cp,
            "total": bp + cp,
            "dtypes": str(sorted(str(d) for d in dtypes)),
            "readout_shape": list(out.shape),
            "readout_finite": bool(torch.isfinite(out).all()),
        }
        del bb, cls
    interface_ok = all(
        v["backbone_params"] == EXPECTED_BACKBONE
        and v["classifier_params"] == EXPECTED_CLASSIFIER
        and v["dtypes"] == "['torch.float32']"
        and v["readout_shape"] == [2, 3]
        and v["readout_finite"]
        for v in interface.values()
    )
    gates["G3_p0_classification_interface"] = interface_ok
    print(f"  interface_ok: {interface_ok} ({interface['P-FT']['backbone_params']}/{interface['P-FT']['classifier_params']})", flush=True)

    # === Section D: complete 402k P0 tokenizer audit (G4) ===
    print("=== D. Complete 402k P0 tokenizer audit ===", flush=True)
    from transformers import AutoTokenizer
    p0_tok = AutoTokenizer.from_pretrained(str(P0_SNAP))
    lengths = []
    max_id = 0
    truncations = 0
    invalid = 0
    n_examples = 0
    for split in SPLITS:
        for dep in DEPTHS:
            path = CORPUS / f"{split}_d{dep}.jsonl"
            with open(path, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    ids = p0_tok.encode(row["rendered"], add_special_tokens=False)
                    lengths.append(len(ids))
                    max_id = max(max_id, max(ids))
                    if len(ids) > NATIVE_MAX_LEN:
                        truncations += 1
                    if any(i >= NATIVE_VOCAB for i in ids):
                        invalid += 1
                    n_examples += 1
    s = sorted(lengths)
    n = len(s)
    audit = {
        "tokenizer": "P0_native (pinned snapshot)",
        "tokenizer_len": len(p0_tok),
        "examples": n_examples,
        "token_length_min": s[0],
        "token_length_median": s[n // 2],
        "token_length_p95": s[int(n * 0.95)],
        "token_length_p99": s[int(n * 0.99)],
        "token_length_max": s[-1],
        "max_token_id": max_id,
        "truncation_count": truncations,
        "invalid_or_unknown_count": invalid,
        "special_tokens_added": 0,
        "add_special_tokens": False,
    }
    print(f"  n={n} min={audit['token_length_min']} med={audit['token_length_median']} "
          f"p95={audit['token_length_p95']} p99={audit['token_length_p99']} "
          f"max={audit['token_length_max']} trunc={truncations} invalid={invalid}", flush=True)

    prior_ev = json.loads(PRIOR_SNAPSHOT_SOURCE.read_text(encoding="utf-8"))
    prior_audit = prior_ev["tokenizer_audit"]["P0_native"]
    compare_fields = ("tokenizer_len", "examples", "token_length_min", "token_length_median",
                      "token_length_p95", "token_length_p99", "token_length_max",
                      "max_token_id", "truncation_count", "invalid_or_unknown_count")
    audit_match = {f: {"prior": prior_audit.get(f), "now": audit.get(f),
                       "match": prior_audit.get(f) == audit.get(f)} for f in compare_fields}
    audit_all_match = all(v["match"] for v in audit_match.values())
    gates["G4a_tokenizer_audit_bounds"] = (
        audit["token_length_max"] <= NATIVE_MAX_LEN and truncations == 0 and invalid == 0)
    gates["G4b_tokenizer_audit_reproduces_prior"] = audit_all_match
    if not audit_all_match:
        for f, v in audit_match.items():
            if not v["match"]:
                print(f"  AUDIT MISMATCH {f}: prior={v['prior']} now={v['now']}", flush=True)

    # === Section E: frozen-snapshot field comparison (G5) ===
    print("=== E. Frozen-snapshot field comparison ===", flush=True)
    prior_runtime = prior_ev["runtime_snapshot"]
    runtime_compare_fields = (
        "python_version", "python_implementation", "igraph_version", "igraph_py_sha256",
        "torch_version", "cuda_version", "gpu_name", "gpu_memory_bytes", "bf16_supported",
        "cudnn_version", "cudnn_enabled", "transformers_version", "numpy_version",
        "scipy_version", "sklearn_version", "tokenizers_version", "nvidia_driver",
        "pip_freeze_sha256", "cuda_available",
    )
    runtime_match = {}
    for f in runtime_compare_fields:
        if f not in prior_runtime:
            runtime_match[f] = {"prior": None, "now": runtime.get(f), "match": False,
                                "note": "absent from prior snapshot"}
            continue
        runtime_match[f] = {"prior": prior_runtime[f], "now": runtime.get(f),
                            "match": prior_runtime[f] == runtime.get(f)}
        if not runtime_match[f]["match"]:
            print(f"  RUNTIME MISMATCH {f}: prior={prior_runtime[f]} now={runtime.get(f)}", flush=True)
    gates["G5b_runtime_matches_frozen_snapshot"] = all(v["match"] for v in runtime_match.values())

    all_pass = all(gates.values())
    evidence = {
        "schema_id": "E0-V061-GPU-RUNTIME-FREEZE-v0",
        "authority": "V06-GPU-BOOTSTRAP-REMEDIATION-1",
        "supersedes": "r0 GPU execution under system Python (transformers 4.50.0) — see incidents/EXECUTION_RUNTIME_SNAPSHOT_DRIFT.md",
        "prior_snapshot_source": str(PRIOR_SNAPSHOT_SOURCE.name),
        "code_git_commit": git_head(),
        "runtime": runtime,
        "p0_digest_check": digest_check,
        "p0_classification_interface": interface,
        "tokenizer_audit": audit,
        "tokenizer_audit_prior_comparison": audit_match,
        "runtime_prior_comparison": runtime_match,
        "gates": gates,
        "status": "PASS" if all_pass else "FAIL",
        "wall_seconds": round(time.time() - started, 1),
    }
    OUTPUT.write_text(json.dumps(evidence, indent=2, default=str) + "\n",
                      encoding="utf-8", newline="\n")
    print(f"\nGATES: {json.dumps(gates)}", flush=True)
    print(f"STATUS: {evidence['status']}  evidence: {OUTPUT}", flush=True)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
