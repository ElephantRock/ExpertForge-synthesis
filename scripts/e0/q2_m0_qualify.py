"""E0 Q2 — M0 qualification runner (frozen contract implementation).

Usage:
  python scripts/e0/q2_m0_qualify.py --smoke
  python scripts/e0/q2_m0_qualify.py --candidate C0        # full 3-seed run

Frozen contract: docs/experiments/e0/Q2_Q3_EXECUTION_RELEASE.md.
Data: hash-verified Q1 corpus under local_data/e0_qualification_bootstrap/Q1.

Interpretation notes — RESOLVED by project authority (Q2_FULL_RUN_RELEASE.md):
  - R1 (weight decay): exclude_norm_bias is authoritative. Weight decay 0.05
    applies to token embeddings, attention matrices, MLP matrices, and the
    classifier matrix; 0 applies to every RMSNorm scale and the classifier
    bias (frozen Decision-22 parameter-group semantics). Scientific Q2
    invocations fail closed if `all` is requested; `all` remains available
    only for explicitly non-scientific diagnostics.
  - R2 (permutation derivation): approved as implemented —
    sha256("ExpertForge-E0-Q2|M0|perm|<seed>|<epoch>")[:8] big-endian,
    zero-based epochs.
  - Gradient accumulation divides each microbatch mean loss by 8 so the
    accumulated gradient equals the exact 128-example batch mean.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F

from m0_model import CANDIDATES, MAX_LEN, PAD_TOKEN_ID, build_model

LABELS = ["ENTAILED", "CONTRADICTED", "UNKNOWN"]
LABEL_TO_ID = {name: i for i, name in enumerate(LABELS)}
FROZEN_SEEDS = [1647674144, 1110194409, 335767543]
EFFECTIVE_BATCH = 128
MICROBATCH = 16
GRAD_ACCUM = EFFECTIVE_BATCH // MICROBATCH
UPDATES = 8000
WARMUP = 400
PEAK_LR = 5e-4
FINAL_LR_FRACTION = 0.1
WEIGHT_DECAY = 0.05
BETAS = (0.9, 0.95)
ADAM_EPS = 1e-8
GRAD_CLIP = 1.0
EVAL_EVERY = 400
EVAL_BATCH = 256
EPOCH_PERM_NAMESPACE = "ExpertForge-E0-Q2|M0|perm"

# Q2 qualification gate (frozen)
GATE = {
    "median_Q_min": 0.55,
    "median_Q_max": 0.70,
    "median_Q24_min": 0.50,
    "median_Q_STRUCT_min": 0.50,
    "median_min_label_recall_min": 0.45,
    "seed_Q_min": 0.50,
    "seed_Q_max": 0.75,
}

# P1 (Q2_PRELAUNCH_INTEGRITY_AUDIT.md): frozen Q1 input digests, identical to
# the exact local reproduction record in q1_local_reproduction.json. Scientific
# Q2 must re-verify these at startup and fail closed before model construction.
FROZEN_Q1_INPUT_DIGESTS = {
    "train_ID.jsonl": "79daa2c007bc914e228a0312d8278b8f5ca7cb2c28f1450d508f9924362aacf1",
    "eval_ID.jsonl": "b66b6629173449f71022fba0e7907826733aa7ae28e0b7502716ba4fdd45fce8",
    "eval_STRUCT.jsonl": "688865d27b8e1fd208b2b453670306d1b4ecc9ff9a21130a297e67cc4b07c7d8",
    "CMDR-Lex-v1.json": "c3d44e7a99163456ef149fdd1b1caf2fd694e8412e6f082480cb05c2ef237471",
}


class Q2DivergenceError(RuntimeError):
    """P2: scientific divergence (non-finite state) — fail closed, no rescue."""

    def __init__(self, code: str, update: int, detail: str | None = None) -> None:
        super().__init__(f"{code} at update {update}" + (f" ({detail})" if detail else ""))
        self.code = code
        self.update = update
        self.detail = detail


def cjson(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_q1_inputs(data_root: Path) -> dict[str, str]:
    """P1: re-hash every frozen Q1 input; fail closed on missing/mismatch."""
    verified: dict[str, str] = {}
    for name, expected in FROZEN_Q1_INPUT_DIGESTS.items():
        path = data_root / name
        if not path.is_file():
            raise SystemExit(f"P1 fail-closed: frozen Q1 input missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise SystemExit(
                f"P1 fail-closed: {name} sha256 {actual} != frozen {expected}"
            )
        verified[name] = actual
    return verified


def _require_finite(tensor: torch.Tensor, code: str, update: int) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise Q2DivergenceError(code, update)


def perm_seed(seed: int, epoch: int) -> int:
    digest = hashlib.sha256(f"{EPOCH_PERM_NAMESPACE}|{seed}|{epoch}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


class TrainStream:
    """Repeated deterministic full-corpus permutations; batches cross
    permutation boundaries; no example is dropped at epoch boundaries."""

    def __init__(self, n: int, seed: int) -> None:
        self.n = n
        self.seed = seed
        self._epoch = 0
        self._buffer: list[int] = []

    def _epoch_indices(self, epoch: int) -> list[int]:
        rng = random.Random(perm_seed(self.seed, epoch))
        indices = list(range(self.n))
        rng.shuffle(indices)
        return indices

    def take(self, count: int) -> list[int]:
        out: list[int] = []
        while len(out) < count:
            if not self._buffer:
                self._buffer = self._epoch_indices(self._epoch)
                self._epoch += 1
            need = count - len(out)
            out.extend(self._buffer[:need])
            self._buffer = self._buffer[need:]
        return out


def load_split(root: Path, name: str) -> list[dict]:
    from q1_cmdr_bootstrap import LEX

    rows = []
    rejected_overlength = 0
    with (root / f"{name}.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            ids = LEX.encode(row["rendered"], append_decide=True)
            if len(ids) != row["student_token_count_with_DECIDE"]:
                raise AssertionError(
                    f"{name}:{row['sample_id']} token count mismatch "
                    f"{len(ids)} != {row['student_token_count_with_DECIDE']}"
                )
            if len(ids) > MAX_LEN:
                rejected_overlength += 1
                continue
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "token_ids": ids,
                    "label_id": LABEL_TO_ID[row["gold_label"]],
                    "depth": row["reasoning_depth_stratum"],
                    "surface": row["surface"],
                }
            )
    return rows, rejected_overlength


def collate(rows: list[dict], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = [len(r["token_ids"]) for r in rows]
    max_len = max(lengths)
    token_ids = torch.full((len(rows), max_len), PAD_TOKEN_ID, dtype=torch.long)
    for i, r in enumerate(rows):
        token_ids[i, : len(r["token_ids"])] = torch.tensor(r["token_ids"], dtype=torch.long)
    decide = torch.tensor([l - 1 for l in lengths], dtype=torch.long)
    labels = torch.tensor([r["label_id"] for r in rows], dtype=torch.long)
    return token_ids.to(device), decide.to(device), labels.to(device)


def learning_rate(update: int) -> float:
    if update <= WARMUP:
        return PEAK_LR * update / WARMUP
    final = PEAK_LR * FINAL_LR_FRACTION
    progress = (update - WARMUP) / (UPDATES - WARMUP)
    return final + (PEAK_LR - final) * 0.5 * (1.0 + float(np.cos(np.pi * progress)))


def macro_cell_accuracy(y_true: np.ndarray, y_pred: np.ndarray, depths: np.ndarray) -> float:
    values = []
    for depth in range(1, 5):
        for label_id in range(len(LABELS)):
            idx = (depths == depth) & (y_true == label_id)
            values.append(float(np.mean(y_pred[idx] == y_true[idx])))
    return float(np.mean(values))


def macro_cell_accuracy_depths(
    y_true: np.ndarray, y_pred: np.ndarray, depths: np.ndarray, wanted: tuple[int, ...]
) -> float:
    values = []
    for depth in wanted:
        for label_id in range(len(LABELS)):
            idx = (depths == depth) & (y_true == label_id)
            values.append(float(np.mean(y_pred[idx] == y_true[idx])))
    return float(np.mean(values))


@torch.no_grad()
def predict(model: torch.nn.Module, rows: list[dict], device: torch.device, update: int | None = None) -> np.ndarray:
    model.eval()
    preds = []
    for start in range(0, len(rows), EVAL_BATCH):
        batch = rows[start : start + EVAL_BATCH]
        ids, decide, _ = collate(batch, device)
        logits = model(ids, decide)
        _require_finite(logits, "nonfinite_eval_logits", update or -1)
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
    model.train()
    return np.asarray(preds, dtype=np.int64)


def evaluate(model: torch.nn.Module, rows: list[dict], device: torch.device, update: int | None = None) -> dict:
    y_true = np.asarray([r["label_id"] for r in rows], dtype=np.int64)
    depths = np.asarray([r["depth"] for r in rows], dtype=np.int64)
    y_pred = predict(model, rows, device, update)
    return {
        "cmdr_sma": macro_cell_accuracy(y_true, y_pred, depths),
        "sma_depth_2_4": macro_cell_accuracy_depths(y_true, y_pred, depths, (2, 3, 4)),
    }


def host_peak_memory_bytes() -> int | None:
    try:
        import psutil

        return psutil.Process().memory_info().peak_wset
    except Exception:
        return None


def parameter_groups(model: torch.nn.Module, wd_scope: str) -> list[dict]:
    if wd_scope == "all":
        return [{"params": list(model.parameters()), "weight_decay": WEIGHT_DECAY}]
    if wd_scope == "exclude_norm_bias":
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            target = no_decay if name.endswith("norm.weight") or name.endswith("classifier.bias") else decay
            target.append(param)
        return [
            {"params": decay, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ]
    raise ValueError(f"unknown wd scope {wd_scope}")


def scientific_setup(
    candidate: str,
    wd_scope: str,
    seed: int,
    data_root: Path,
    repo_root: Path,
) -> dict:
    """Shared scientific pre-training setup (Q2_C0_INFRASTRUCTURE_RETRY_RELEASE.md).

    Both the full --candidate qualification path and --preflight traverse this
    exact function, in this order:
      1. reject non-authoritative --wd-scope all;
      2. enable deterministic algorithms, disable TF32;
      3. require CUDA;
      4. re-hash all four frozen Q1 inputs (P1);
      5. assemble lineage: Git HEAD, clean-tree-at-start, frozen input digests,
         runtime snapshot digest (P3);
      6. load train_ID / eval_ID / eval_STRUCT with no truncation;
      7. construct the requested M0 candidate at the requested frozen seed;
      8. construct AdamW with the frozen exclude_norm_bias parameter groups.
    """
    if wd_scope == "all":
        raise SystemExit(
            "Scientific Q2 execution forbids --wd-scope all "
            "(frozen Decision-22 parameter-group semantics)."
        )
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable — frozen Q2 contract requires CUDA.")
    data_root = data_root.resolve()
    verified_digests = verify_q1_inputs(data_root)
    runtime_snapshot_path = repo_root / "docs" / "experiments" / "e0" / "q2_q3_runtime.snapshot.json"
    lineage = {
        "code_git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root,
            capture_output=True, text=True, check=True,
        ).stdout.strip(),
        "working_tree_clean_at_start": subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_root,
            capture_output=True, text=True, check=True,
        ).stdout.strip() == "",
        "verified_q1_input_digests": verified_digests,
        "runtime_snapshot_sha256": sha256_file(runtime_snapshot_path),
    }
    print(f"P1 verified frozen Q1 input digests under {data_root}", flush=True)
    train_rows, _ = load_split(data_root, "train_ID")
    eval_id_rows, _ = load_split(data_root, "eval_ID")
    eval_struct_rows, _ = load_split(data_root, "eval_STRUCT")
    model = build_model(candidate, seed, device)
    optimizer = torch.optim.AdamW(
        parameter_groups(model, wd_scope),
        lr=PEAK_LR,
        betas=BETAS,
        eps=ADAM_EPS,
    )
    return {
        "device": device,
        "verified_digests": verified_digests,
        "lineage": lineage,
        "train_rows": train_rows,
        "eval_id_rows": eval_id_rows,
        "eval_struct_rows": eval_struct_rows,
        "model": model,
        "optimizer": optimizer,
    }


def run_training(
    candidate: str,
    seed: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_rows: list[dict],
    eval_id_rows: list[dict],
    eval_struct_rows: list[dict],
    device: torch.device,
    wd_scope: str,
    out_dir: Path,
    updates: int = UPDATES,
    eval_every: int = EVAL_EVERY,
    lineage: dict | None = None,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)

    stream = TrainStream(len(train_rows), seed)
    torch.cuda.reset_peak_memory_stats(device)

    best = {"update": 0, "sma": float("-inf"), "state": None}
    eval_history = []
    last_finite_eval_update: int | None = None
    started = time.time()
    try:
        for update in range(1, updates + 1):
            lr = learning_rate(update)
            for group in optimizer.param_groups:
                group["lr"] = lr
            batch_indices = stream.take(EFFECTIVE_BATCH)
            optimizer.zero_grad(set_to_none=True)
            for micro in range(GRAD_ACCUM):
                rows = [train_rows[i] for i in batch_indices[micro * MICROBATCH : (micro + 1) * MICROBATCH]]
                ids, decide, labels = collate(rows, device)
                logits = model(ids, decide)
                loss = F.cross_entropy(logits, labels, label_smoothing=0.0)
                _require_finite(loss, "nonfinite_microbatch_loss", update)
                (loss / GRAD_ACCUM).backward()
            for param in model.parameters():
                if param.grad is not None:
                    _require_finite(param.grad, "nonfinite_accumulated_gradient", update)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            _require_finite(grad_norm, "nonfinite_grad_norm", update)
            optimizer.step()
            for param in model.parameters():
                _require_finite(param.data, "nonfinite_parameters_after_step", update)

            if update % eval_every == 0:
                sma = evaluate(model, eval_id_rows, device, update)["cmdr_sma"]
                if not np.isfinite(sma):
                    raise Q2DivergenceError("nonfinite_eval_metric", update)
                eval_history.append({"update": update, "eval_ID_cmdr_sma": sma, "lr": lr})
                last_finite_eval_update = update
                if sma > best["sma"]:
                    best = {
                        "update": update,
                        "sma": sma,
                        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    }
                print(f"[{candidate} seed={seed}] update {update}/{updates} eval_ID SMA={sma:.6f} lr={lr:.3e}", flush=True)
    except Q2DivergenceError as exc:
        # P2: divergence cannot be rescued by an earlier checkpoint. Write the
        # machine-readable failure record and re-raise; no normal seed result.
        out_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema_id": "E0-Q2-M0-SCIENTIFIC-FAILURE-v0",
            "candidate": candidate,
            "seed": seed,
            "update": exc.update,
            "failure_code": exc.code,
            "detail": exc.detail,
            "last_finite_checkpoint_update": last_finite_eval_update,
            "non_scientific": seed not in FROZEN_SEEDS,
            **(lineage or {}),
        }
        (out_dir / "scientific_failure.json").write_text(
            json.dumps(failure, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        raise

    model.load_state_dict(best["state"])
    model.to(device)

    id_metrics = evaluate(model, eval_id_rows, device)
    struct_metrics = evaluate(model, eval_struct_rows, device)
    q_id, q_struct = id_metrics["cmdr_sma"], struct_metrics["cmdr_sma"]
    q = (q_id + q_struct) / 2.0

    pooled_true = np.asarray(
        [r["label_id"] for r in eval_id_rows] + [r["label_id"] for r in eval_struct_rows], dtype=np.int64
    )
    pooled_depths = np.asarray(
        [r["depth"] for r in eval_id_rows] + [r["depth"] for r in eval_struct_rows], dtype=np.int64
    )
    pooled_pred = np.concatenate(
        [predict(model, eval_id_rows, device), predict(model, eval_struct_rows, device)]
    )
    q_2_4 = macro_cell_accuracy_depths(pooled_true, pooled_pred, pooled_depths, (2, 3, 4))
    label_recall = {}
    for label_id, name in enumerate(LABELS):
        idx = pooled_true == label_id
        label_recall[name] = float(np.mean(pooled_pred[idx] == pooled_true[idx]))
    depth_accuracy = {
        f"d{depth}": float(np.mean(pooled_pred[pooled_depths == depth] == pooled_true[pooled_depths == depth]))
        for depth in range(1, 5)
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best.pt"
    torch.save(best["state"], ckpt_path)

    result = {
        "schema_id": "E0-Q2-M0-SEED-RESULT-v0",
        "candidate": candidate,
        "seed": seed,
        "updates": updates,
        "eval_every": eval_every,
        "wd_scope": wd_scope,
        **(lineage or {}),
        "param_count": CANDIDATES[candidate]["params"],
        "selected_update": best["update"],
        "selected_eval_ID_cmdr_sma": best["sma"],
        "Q_ID": q_id,
        "Q_STRUCT": q_struct,
        "Q": q,
        "Q_2_4_pooled": q_2_4,
        "Q_2_4_ID": id_metrics["sma_depth_2_4"],
        "Q_2_4_STRUCT": struct_metrics["sma_depth_2_4"],
        "label_recall_pooled": label_recall,
        "min_label_recall": min(label_recall.values()),
        "per_depth_accuracy_pooled": depth_accuracy,
        "eval_history": eval_history,
        "best_checkpoint_sha256": sha256_file(ckpt_path),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "host_peak_memory_bytes": host_peak_memory_bytes(),
        "wall_seconds": round(time.time() - started, 3),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    return result


def evaluate_candidate_gate(results: list[dict]) -> dict:
    seeds_q = sorted(r["Q"] for r in results)
    med = lambda key: float(np.median([r[key] for r in results]))
    median_q = med("Q")
    median_q24 = med("Q_2_4_pooled")
    median_qstruct = med("Q_STRUCT")
    median_min_recall = med("min_label_recall")
    seeds_in_range = all(GATE["seed_Q_min"] <= r["Q"] <= GATE["seed_Q_max"] for r in results)
    checks = {
        "median_Q_in_band": GATE["median_Q_min"] <= median_q <= GATE["median_Q_max"],
        "median_Q_2_4_ge_0_50": median_q24 >= GATE["median_Q24_min"],
        "median_Q_STRUCT_ge_0_50": median_qstruct >= GATE["median_Q_STRUCT_min"],
        "median_min_label_recall_ge_0_45": median_min_recall >= GATE["median_min_label_recall_min"],
        "every_seed_Q_in_0_50_0_75": seeds_in_range,
    }
    if all(checks.values()):
        decision = "PASS_SELECT_CANDIDATE"
    elif median_q > GATE["median_Q_max"]:
        decision = "ABOVE_CEILING_STOP"
    else:
        decision = "BELOW_FLOOR_NEXT_CANDIDATE_OR_UNQUALIFIED"
    return {
        "schema_id": "E0-Q2-M0-CANDIDATE-GATE-v0",
        "gate": GATE,
        "checks": checks,
        "median_Q": median_q,
        "median_Q_2_4": median_q24,
        "median_Q_STRUCT": median_qstruct,
        "median_min_label_recall": median_min_recall,
        "per_seed_Q": seeds_q,
        "decision": decision,
    }


def write_summary(candidate: str, results: list[dict], gate: dict, repo_root: Path) -> None:
    summary_path = repo_root / "docs" / "experiments" / "e0" / "q2_m0_qualification_summary.json"
    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "schema_id": "E0-Q2-M0-QUALIFICATION-SUMMARY-v0",
            "candidates": summary.get("candidates", {}),
        "execution_notes": {
            "wd_scope": "exclude_norm_bias (frozen Decision-22 / R1)",
            "permutation_seed_rule": "sha256('ExpertForge-E0-Q2|M0|perm|<seed>|<epoch>')[:8] big-endian, zero-based epochs",
        },
        }
    )
    summary["candidates"][candidate] = {
        "per_seed_results": results,
        "gate_evaluation": gate,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")


# ---------------------------------------------------------------- smoke ----

def run_smoke(data_root: Path, out_path: Path, repo_root: Path) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    checks: dict[str, dict] = {}
    checks["deterministic_algorithms_enabled"] = {"ok": True}

    train_rows, rej_train = load_split(data_root, "train_ID")
    eval_id_rows, rej_id = load_split(data_root, "eval_ID")
    eval_struct_rows, rej_struct = load_split(data_root, "eval_STRUCT")
    total_rejected = rej_train + rej_id + rej_struct
    checks["data_load"] = {
        "train_ID": len(train_rows),
        "eval_ID": len(eval_id_rows),
        "eval_STRUCT": len(eval_struct_rows),
        "overlength_rejected": total_rejected,
        "ok": len(train_rows) == 24000 and len(eval_id_rows) == 6000 and len(eval_struct_rows) == 6000
        and total_rejected == 0,
    }

    stream_a = TrainStream(len(train_rows), FROZEN_SEEDS[0])
    stream_b = TrainStream(len(train_rows), FROZEN_SEEDS[0])
    stream_c = TrainStream(len(train_rows), FROZEN_SEEDS[1])
    idx_a = [stream_a.take(EFFECTIVE_BATCH) for _ in range(3)]
    idx_b = [stream_b.take(EFFECTIVE_BATCH) for _ in range(3)]
    idx_c = [stream_c.take(EFFECTIVE_BATCH) for _ in range(3)]
    epoch0_a = TrainStream._epoch_indices(stream_a, 0)
    checks["deterministic_data_order"] = {
        "same_seed_reproducible": idx_a == idx_b,
        "different_seed_differs": idx_a != idx_c,
        "epoch0_is_permutation": sorted(epoch0_a) == list(range(len(train_rows))),
        "ok": idx_a == idx_b and idx_a != idx_c and sorted(epoch0_a) == list(range(len(train_rows))),
    }

    lr_checks = {
        "lr_warmup_end": learning_rate(WARMUP) == PEAK_LR,
        "lr_final": abs(learning_rate(UPDATES) - PEAK_LR * FINAL_LR_FRACTION) < 1e-15,
        "lr_warmup_start": abs(learning_rate(1) - PEAK_LR / WARMUP) < 1e-15,
    }
    checks["lr_schedule"] = {**lr_checks, "ok": all(lr_checks.values())}

    y_true = np.asarray([r["label_id"] for r in eval_id_rows], dtype=np.int64)
    depths = np.asarray([r["depth"] for r in eval_id_rows], dtype=np.int64)
    sma_gold = macro_cell_accuracy(y_true, y_true, depths)
    checks["metric_sanity"] = {
        "gold_cmdr_sma": sma_gold,
        "ok": sma_gold == 1.0,
    }

    param_counts = {}
    for cand in CANDIDATES:
        model = build_model(cand, FROZEN_SEEDS[0], torch.device("cpu"))
        actual = sum(p.numel() for p in model.parameters())
        param_counts[cand] = {"actual": actual, "expected": CANDIDATES[cand]["params"]}
        del model
    checks["parameter_counts"] = {
        "counts": param_counts,
        "ok": all(v["actual"] == v["expected"] for v in param_counts.values()),
    }

    init_model = build_model("C0", FROZEN_SEEDS[0], torch.device("cpu"))
    norms_all_one = all(
        float(p.detach().min()) == 1.0 and float(p.detach().max()) == 1.0
        for n, p in init_model.named_parameters() if n.endswith("norm.weight")
    )
    bias_all_zero = bool(torch.all(init_model.classifier.bias == 0).item())
    checks["init_audit"] = {
        "rmsnorm_scales_exactly_1": norms_all_one,
        "classifier_bias_exactly_0": bias_all_zero,
        "ok": norms_all_one and bias_all_zero,
    }
    del init_model

    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None

    def one_step_loss() -> tuple[float, float]:
        torch.manual_seed(FROZEN_SEEDS[0])
        model = build_model("C0", FROZEN_SEEDS[0], device)
        optimizer = torch.optim.AdamW(
            parameter_groups(model, "exclude_norm_bias"), lr=learning_rate(1),
            betas=BETAS, eps=ADAM_EPS,
        )
        rows = [train_rows[i] for i in TrainStream(len(train_rows), FROZEN_SEEDS[0]).take(MICROBATCH)]
        ids, decide, labels = collate(rows, device)
        logits = model(ids, decide)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP))
        grads_finite = all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        return float(loss.item()), grad_norm, grads_finite, model

    loss1, norm1, finite1, model1 = one_step_loss()
    loss2, norm2, finite2, model2 = one_step_loss()
    checks["one_forward_backward_step"] = {
        "wd_scope": "exclude_norm_bias",
        "loss_finite": loss1 == loss1 and abs(loss1) < 1e6,
        "grads_finite": finite1,
        "loss_bitexact_repeatable": loss1 == loss2,
        "grad_norm_bitexact_repeatable": norm1 == norm2,
        "loss": loss1,
        "grad_norm": norm1,
        "ok": loss1 == loss1 and finite1 and loss1 == loss2 and norm1 == norm2,
    }
    del model1, model2
    if device.type == "cuda":
        checks["one_forward_backward_step"]["cuda_peak_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated(device)
        )
        torch.cuda.empty_cache()

    manifest = {
        "schema_id": "E0-Q2-SMOKE-v0",
        "git_commit": subprocess_git_head(repo_root),
        "device": str(device),
        "torch_version": torch.__version__,
        "checks": checks,
        "status": "PASS" if all(c.get("ok", False) for c in checks.values()) else "FAIL",
        "non_scientific": True,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(manifest, indent=2))
    return manifest


def subprocess_git_head(repo_root: Path) -> str:
    import subprocess

    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--smoke-train",
        action="store_true",
        help="non-scientific: 16-update C0 run exercising the full training/eval/checkpoint path",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="non-scientific: rehearse the exact scientific setup branch plus one disposable step",
    )
    parser.add_argument(
        "--preflight-seed",
        type=int,
        default=None,
        help="diagnostic override; defaults to the first frozen seed 1647674144",
    )
    parser.add_argument("--candidate", choices=list(CANDIDATES))
    parser.add_argument(
        "--wd-scope",
        choices=["exclude_norm_bias", "all"],
        default="exclude_norm_bias",
        help="exclude_norm_bias is the frozen Decision-22 semantics (R1). "
        "'all' is permitted only for non-scientific diagnostics and is "
        "rejected on scientific full-run invocations.",
    )
    parser.add_argument(
        "--data-root", type=Path, default=Path("local_data/e0_qualification_bootstrap/Q1")
    )
    parser.add_argument(
        "--out-root", type=Path, default=Path("local_data/e0_qualification_bootstrap/Q2")
    )
    args = parser.parse_args()
    repo_root = (Path(__file__).resolve().parent.parent.parent).resolve()

    if args.smoke:
        manifest = run_smoke(args.data_root.resolve(), repo_root / "docs/experiments/e0/q2_smoke_manifest.json", repo_root)
        raise SystemExit(0 if manifest["status"] == "PASS" else 1)

    if args.smoke_train:
        # Non-scientific short-path exercise of run_training (stream, accumulation,
        # clipping, periodic eval, best-checkpoint selection, reload, final metrics,
        # checkpoint digest). Results are intentionally NOT qualification evidence.
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device("cuda")
        if not torch.cuda.is_available():
            raise SystemExit("CUDA unavailable.")
        data_root = args.data_root.resolve()
        train_rows, _ = load_split(data_root, "train_ID")
        eval_id_rows, _ = load_split(data_root, "eval_ID")
        eval_struct_rows, _ = load_split(data_root, "eval_STRUCT")
        model = build_model("C0", 0, device)
        optimizer = torch.optim.AdamW(
            parameter_groups(model, args.wd_scope), lr=PEAK_LR, betas=BETAS, eps=ADAM_EPS,
        )
        out_dir = args.out_root.resolve() / "SMOKE_TRAIN" / "seed_0"
        result = run_training(
            "C0", 0, model, optimizer, train_rows, eval_id_rows, eval_struct_rows,
            device, args.wd_scope, out_dir, updates=16, eval_every=8,
        )
        result["non_scientific"] = True
        out_dir.joinpath("result.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        print(json.dumps({k: result[k] for k in ["candidate", "updates", "selected_update", "Q", "Q_ID", "Q_STRUCT", "wall_seconds"]}, indent=2))
        raise SystemExit(0)

    if not args.candidate and not args.preflight:
        parser.error("provide --candidate, --preflight, --smoke, or --smoke-train")

    if args.preflight:
        # Q2_C0_INFRASTRUCTURE_RETRY_RELEASE.md: rehearse the exact scientific
        # setup branch (shared scientific_setup) plus one disposable guarded
        # step. Emits NO scientific result.json/best.pt/summary — only the
        # non-scientific preflight manifest.
        candidate = args.candidate or "C0"
        seed = args.preflight_seed if args.preflight_seed is not None else FROZEN_SEEDS[0]
        ctx = scientific_setup(candidate, args.wd_scope, seed, args.data_root, repo_root)
        device, model, optimizer = ctx["device"], ctx["model"], ctx["optimizer"]
        torch.cuda.reset_peak_memory_stats(device)
        rows = [
            ctx["train_rows"][i]
            for i in TrainStream(len(ctx["train_rows"]), seed).take(MICROBATCH)
        ]
        ids, decide, labels = collate(rows, device)
        logits = model(ids, decide)
        loss = F.cross_entropy(logits, labels, label_smoothing=0.0)
        loss_finite = bool(torch.isfinite(loss).all())
        loss.backward()
        grads_finite = all(
            bool(torch.isfinite(p.grad).all())
            for p in model.parameters()
            if p.grad is not None
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        grad_norm_finite = bool(torch.isfinite(grad_norm).all())
        optimizer.step()
        params_finite = all(bool(torch.isfinite(p.data).all()) for p in model.parameters())
        cuda_peak = int(torch.cuda.max_memory_allocated(device))
        groups = parameter_groups(model, args.wd_scope)
        decay_elems = sum(
            p.numel() for g in groups if g["weight_decay"] > 0 for p in g["params"]
        )
        no_decay_elems = sum(
            p.numel() for g in groups if g["weight_decay"] == 0 for p in g["params"]
        )
        checks_pass = loss_finite and grads_finite and grad_norm_finite and params_finite
        parameter_count = sum(p.numel() for p in model.parameters())
        # The preflight never constructs seed output dirs, checkpoints, result
        # records, or the candidate summary; no scientific artifacts are
        # produced by this process.
        del model, optimizer
        if device.type == "cuda":
            torch.cuda.empty_cache()
        manifest = {
            "schema_id": "E0-Q2-SCIENTIFIC-PREFLIGHT-v0",
            "non_scientific": True,
            "candidate": candidate,
            "seed": seed,
            "code_git_commit": ctx["lineage"]["code_git_commit"],
            "working_tree_clean_at_start": ctx["lineage"]["working_tree_clean_at_start"],
            "verified_q1_input_digests": ctx["verified_digests"],
            "runtime_snapshot_sha256": ctx["lineage"]["runtime_snapshot_sha256"],
            "wd_scope": args.wd_scope,
            "parameter_count": parameter_count,
            "decay_parameter_count": decay_elems,
            "no_decay_parameter_count": no_decay_elems,
            "loss_finite": loss_finite,
            "gradients_finite": grads_finite,
            "grad_norm_finite": grad_norm_finite,
            "parameters_finite_after_step": params_finite,
            "cuda_peak_allocated_bytes": cuda_peak,
            "microbatch_source": "TrainStream(<seed>).take(MICROBATCH) — identical to the scientific first microbatch",
            "emits_scientific_artifacts": False,
            "status": "PASS" if checks_pass else "FAIL",
        }
        out_path = repo_root / "docs" / "experiments" / "e0" / "q2_c0_scientific_preflight.json"
        out_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps(manifest, indent=2))
        raise SystemExit(0 if manifest["status"] == "PASS" else 1)

    if args.wd_scope == "all":
        # R1 (Q2_FULL_RUN_RELEASE.md): scientific Q2 must use exclude_norm_bias.
        # (scientific_setup enforces this too; fail before touching anything.)
        raise SystemExit(
            "Scientific Q2 execution forbids --wd-scope all "
            "(frozen Decision-22 parameter-group semantics)."
        )

    results = []
    try:
        for seed in FROZEN_SEEDS:
            ctx = scientific_setup(args.candidate, args.wd_scope, seed, args.data_root, repo_root)
            out_dir = args.out_root.resolve() / args.candidate / f"seed_{seed}"
            results.append(
                run_training(
                    args.candidate, seed, ctx["model"], ctx["optimizer"],
                    ctx["train_rows"], ctx["eval_id_rows"], ctx["eval_struct_rows"],
                    ctx["device"], args.wd_scope, out_dir, lineage=ctx["lineage"],
                )
            )
    except Q2DivergenceError as exc:
        print(f"P2 fail-closed: {exc}", flush=True)
        raise SystemExit(2) from exc
    gate = evaluate_candidate_gate(results)
    write_summary(args.candidate, results, gate, repo_root)
    print(json.dumps(gate, indent=2))
    if gate["decision"] == "BELOW_FLOOR_NEXT_CANDIDATE_OR_UNQUALIFIED" and args.candidate == "C2":
        raise SystemExit("Q2 UNQUALIFIED: C2 remains below floor.")


if __name__ == "__main__":
    main()
