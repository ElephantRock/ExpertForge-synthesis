"""V06-GPU-BOOTSTRAP deterministic model replay — C0/R and P0 trainable paths.

Each family is trained twice in SEPARATE processes (fresh CUDA context) on
the burned replay fixture corpus (local_data/e0_v061_replay, namespace
ExpertForge-E0-v061-replay-burn). At each checkpoint update (40, 80, 120)
the run records:
  - an exact SHA-256 digest of the full trainable state,
  - the full dev argmax prediction vector,
  - a digest of the raw float32 dev logits,
  - the dev metric (macro-cell SMA for C0, accuracy for P0).

The --compare mode asserts EXACT equality (bitwise state digests, identical
prediction vectors, identical metric values) between run 1 and run 2.

Usage (each invocation is its own process):
  python scripts/e0/v061/gpu_replay_harness.py --family c0  --run-index 1
  python scripts/e0/v061/gpu_replay_harness.py --family c0  --run-index 2
  python scripts/e0/v061/gpu_replay_harness.py --family p0  --run-index 1
  python scripts/e0/v061/gpu_replay_harness.py --family p0  --run-index 2
  python scripts/e0/v061/gpu_replay_harness.py --compare c0
  python scripts/e0/v061/gpu_replay_harness.py --compare p0
  python scripts/e0/v061/gpu_replay_harness.py --compare all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from deterministic_preamble import apply as apply_det, query as query_det, verify_applied

DET_STATE = apply_det()  # BEFORE torch import / CUDA probe

import torch
import torch.nn.functional as F
import numpy as np

from q2_m0_qualify import (
    BETAS,
    ADAM_EPS,
    GRAD_CLIP,
    MICROBATCH,
    GRAD_ACCUM,
    EFFECTIVE_BATCH,
    TrainStream,
    collate,
    learning_rate,
    parameter_groups,
)
from gpu_smoke_harness import (
    FIRST_PRIMARY_SEED,
    P0_SNAP,
    REPO_ROOT,
    build_p0_classification,
    collate_p0,
    load_smoke_raw,
    load_smoke_rows,
    p0_lr_at,
    p0_parameter_groups,
    p0_readout_logits,
    rng_substream,
    sha256_file,
)

REPLAY_DIR = REPO_ROOT / "local_data" / "e0_v061_replay_r2"
RUNS_DIR = REPLAY_DIR / "runs"
DOCS = REPO_ROOT / "docs" / "experiments" / "e0" / "v061"

REPLAY_UPDATES = 120
CHECKPOINT_UPDATES = (40, 80, 120)
EVAL_BATCH = 64


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def state_digest(named_tensors) -> str:
    """Exact digest of an ordered (name, tensor) sequence: dtype, shape, bytes."""
    h = hashlib.sha256()
    for name, tensor in named_tensors:
        t = tensor.detach().to("cpu", torch.float32).contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def logits_digest(logits: torch.Tensor) -> str:
    t = logits.detach().to("cpu", torch.float32).contiguous()
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()


def replay_c0(seed: int) -> dict:
    """C0/R path on the replay fixture: identical construction to the smoke."""
    from m0_model import build_model
    from q2_m0_qualify import macro_cell_accuracy

    device = torch.device("cuda")
    train_rows, dev_rows = load_smoke_rows(REPLAY_DIR)
    print(f"  replay train={len(train_rows)} dev={len(dev_rows)}", flush=True)

    model = build_model("C0", rng_substream(seed, "backbone_init"), device)
    optimizer = torch.optim.AdamW(
        parameter_groups(model, "exclude_norm_bias"),
        lr=learning_rate(1), betas=BETAS, eps=ADAM_EPS,
    )
    stream = TrainStream(len(train_rows), rng_substream(seed, "data_order"))

    y_true = [r["label_id"] for r in dev_rows]
    depths = [r["depth"] for r in dev_rows]
    checkpoints = []
    started = time.time()
    for update in range(1, REPLAY_UPDATES + 1):
        lr = learning_rate(update)
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch_indices = stream.take(EFFECTIVE_BATCH)
        optimizer.zero_grad(set_to_none=True)
        for micro in range(GRAD_ACCUM):
            rows = [train_rows[i] for i in batch_indices[micro * MICROBATCH:(micro + 1) * MICROBATCH]]
            ids, decide, labels = collate(rows, device)
            logits = model(ids, decide)
            loss = F.cross_entropy(logits, labels, label_smoothing=0.0)
            (loss / GRAD_ACCUM).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        if update in CHECKPOINT_UPDATES:
            model.eval()
            preds = []
            all_logits = []
            with torch.no_grad():
                for start in range(0, len(dev_rows), EVAL_BATCH):
                    b = dev_rows[start:start + EVAL_BATCH]
                    ids, decide, _ = collate(b, device)
                    logits = model(ids, decide)
                    all_logits.append(logits)
                    preds.extend(int(x) for x in logits.argmax(-1).tolist())
            model.train()
            sma = macro_cell_accuracy(
                np.asarray(y_true, dtype=np.int64),
                np.asarray(preds, dtype=np.int64),
                np.asarray(depths, dtype=np.int64),
            )
            checkpoints.append({
                "update": update,
                "state_digest": state_digest(model.state_dict().items()),
                "dev_argmax": preds,
                "dev_logits_sha256": logits_digest(torch.cat(all_logits, dim=0)),
                "dev_macro_cell_sma": float(sma),
            })
            print(f"    update {update}: sma={sma:.6f} state={checkpoints[-1]['state_digest'][:16]}", flush=True)
    return {
        "family": "c0",
        "seed": seed,
        "updates": REPLAY_UPDATES,
        "checkpoint_updates": list(CHECKPOINT_UPDATES),
        "train_examples": len(train_rows),
        "dev_examples": len(dev_rows),
        "checkpoints": checkpoints,
        "wall_seconds": round(time.time() - started, 1),
        "deterministic_state": query_det(),
        "code_git_commit": git_head(),
    }


def replay_p0(seed: int) -> dict:
    """P-FT construction on the replay fixture: identical construction to the smoke."""
    from transformers import AutoTokenizer

    device = torch.device("cuda")
    train_raw, dev_raw = load_smoke_raw(REPLAY_DIR)
    tok = AutoTokenizer.from_pretrained(str(P0_SNAP))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    backbone, classifier = build_p0_classification("P-FT", seed, device)
    optimizer = torch.optim.AdamW(
        p0_parameter_groups(backbone, classifier),
        lr=p0_lr_at(1), betas=BETAS, eps=ADAM_EPS,
    )
    stream = TrainStream(len(train_raw), rng_substream(seed, "data_order"))

    y_true = [r["label_id"] for r in dev_raw]
    checkpoints = []
    started = time.time()
    for update in range(1, REPLAY_UPDATES + 1):
        lr = p0_lr_at(update)
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch_indices = stream.take(EFFECTIVE_BATCH)
        optimizer.zero_grad(set_to_none=True)
        for micro in range(GRAD_ACCUM):
            rows = [train_raw[i] for i in batch_indices[micro * MICROBATCH:(micro + 1) * MICROBATCH]]
            input_ids, attention_mask, labels = collate_p0(rows, tok, device)
            logits = p0_readout_logits(backbone, classifier, input_ids, attention_mask)
            loss = F.cross_entropy(logits, labels, label_smoothing=0.0)
            (loss / GRAD_ACCUM).backward()
        params = list(backbone.parameters()) + list(classifier.parameters())
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        optimizer.step()
        if update in CHECKPOINT_UPDATES:
            backbone.eval(); classifier.eval()
            preds = []
            all_logits = []
            with torch.no_grad():
                for start in range(0, len(dev_raw), EVAL_BATCH):
                    b = dev_raw[start:start + EVAL_BATCH]
                    input_ids, attention_mask, _ = collate_p0(b, tok, device)
                    logits = p0_readout_logits(backbone, classifier, input_ids, attention_mask)
                    all_logits.append(logits)
                    preds.extend(int(x) for x in logits.argmax(-1).tolist())
            backbone.train(); classifier.train()
            acc = sum(int(p == t) for p, t in zip(preds, y_true)) / len(y_true)
            named = ([(f"backbone.{n}", p) for n, p in backbone.state_dict().items()]
                     + [(f"classifier.{n}", p) for n, p in classifier.state_dict().items()])
            checkpoints.append({
                "update": update,
                "state_digest": state_digest(named),
                "dev_argmax": preds,
                "dev_logits_sha256": logits_digest(torch.cat(all_logits, dim=0)),
                "dev_accuracy": acc,
            })
            print(f"    update {update}: acc={acc:.6f} state={checkpoints[-1]['state_digest'][:16]}", flush=True)
    return {
        "family": "p0",
        "seed": seed,
        "updates": REPLAY_UPDATES,
        "checkpoint_updates": list(CHECKPOINT_UPDATES),
        "train_examples": len(train_raw),
        "dev_examples": len(dev_raw),
        "checkpoints": checkpoints,
        "wall_seconds": round(time.time() - started, 1),
        "deterministic_state": query_det(),
        "code_git_commit": git_head(),
    }


def compare_family(family: str) -> dict:
    run1 = json.loads((RUNS_DIR / f"replay_{family}_run1.json").read_text(encoding="utf-8"))
    run2 = json.loads((RUNS_DIR / f"replay_{family}_run2.json").read_text(encoding="utf-8"))
    diffs = []
    if run1["checkpoint_updates"] != run2["checkpoint_updates"]:
        diffs.append("checkpoint update lists differ")
    metric_key = "dev_macro_cell_sma" if family == "c0" else "dev_accuracy"
    for cp1, cp2 in zip(run1["checkpoints"], run2["checkpoints"]):
        u = cp1["update"]
        if cp1["state_digest"] != cp2["state_digest"]:
            diffs.append(f"update {u}: state digest {cp1['state_digest'][:16]} != {cp2['state_digest'][:16]}")
        if cp1["dev_argmax"] != cp2["dev_argmax"]:
            n = sum(1 for a, b in zip(cp1["dev_argmax"], cp2["dev_argmax"]) if a != b)
            diffs.append(f"update {u}: dev argmax differs at {n}/{len(cp1['dev_argmax'])} positions")
        if cp1["dev_logits_sha256"] != cp2["dev_logits_sha256"]:
            diffs.append(f"update {u}: dev logits digest differs")
        if repr(cp1[metric_key]) != repr(cp2[metric_key]):
            diffs.append(f"update {u}: metric {cp1[metric_key]!r} != {cp2[metric_key]!r}")
    if len(run1["checkpoints"]) != len(run2["checkpoints"]):
        diffs.append("checkpoint counts differ")
    checkpoint_bindings = []
    for cp1, cp2 in zip(run1["checkpoints"], run2["checkpoints"]):
        checkpoint_bindings.append({
            "update": cp1["update"],
            "state_digest_run1": cp1["state_digest"],
            "state_digest_run2": cp2["state_digest"],
            "dev_argmax_sha256_run1": hashlib.sha256(
                json.dumps(cp1["dev_argmax"]).encode()).hexdigest(),
            "dev_argmax_sha256_run2": hashlib.sha256(
                json.dumps(cp2["dev_argmax"]).encode()).hexdigest(),
            "dev_logits_sha256_run1": cp1["dev_logits_sha256"],
            "dev_logits_sha256_run2": cp2["dev_logits_sha256"],
            "metric_run1": cp1[metric_key],
            "metric_run2": cp2[metric_key],
        })
    return {
        "family": family,
        "run1_commit": run1["code_git_commit"],
        "run2_commit": run2["code_git_commit"],
        "metric_key": metric_key,
        "checkpoint_updates": run1["checkpoint_updates"],
        "checkpoint_bindings": checkpoint_bindings,
        "exact_match": not diffs,
        "differences": diffs,
        "run1_wall_seconds": run1["wall_seconds"],
        "run2_wall_seconds": run2["wall_seconds"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["c0", "p0"], default=None)
    parser.add_argument("--run-index", type=int, choices=[1, 2], default=None)
    parser.add_argument("--compare", choices=["c0", "p0", "all"], default=None)
    parser.add_argument("--seed", type=int, default=FIRST_PRIMARY_SEED)
    args = parser.parse_args()

    if args.compare:
        families = ["c0", "p0"] if args.compare == "all" else [args.compare]
        comparisons = [compare_family(f) for f in families]
        evidence = {
            "schema_id": "E0-V061-GPU-REPLAY-EVIDENCE-v0",
            "authority": "V06-GPU-BOOTSTRAP-EXECUTION-AUTHORIZED",
            "code_git_commit": git_head(),
            "corpus_digests": {
                "train.jsonl": sha256_file(REPLAY_DIR / "train.jsonl"),
                "dev.jsonl": sha256_file(REPLAY_DIR / "dev.jsonl"),
            },
            "replay_updates": REPLAY_UPDATES,
            "checkpoints_per_run": len(CHECKPOINT_UPDATES),
            "comparisons": comparisons,
            "deterministic_state": query_det(),
            "status": "PASS" if all(c["exact_match"] for c in comparisons) else "FAIL",
        }
        out = DOCS / "GPU_REPLAY_EVIDENCE.json"
        out.write_text(json.dumps(evidence, indent=2, default=str) + "\n",
                       encoding="utf-8", newline="\n")
        for c in comparisons:
            print(f"  {c['family']}: exact_match={c['exact_match']} ({len(c['differences'])} diffs)", flush=True)
        print(f"\nSTATUS: {evidence['status']}  evidence: {out}", flush=True)
        sys.exit(0 if evidence["status"] == "PASS" else 1)

    if not args.family or not args.run_index:
        raise SystemExit("either --compare or both --family and --run-index are required")

    if args.family == "c0":
        record = replay_c0(args.seed)
    else:
        record = replay_p0(args.seed)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / f"replay_{args.family}_run{args.run_index}.json"
    out.write_text(json.dumps(record, indent=2, default=str) + "\n",
                   encoding="utf-8", newline="\n")
    print(f"\nrun record: {out}", flush=True)


if __name__ == "__main__":
    main()
