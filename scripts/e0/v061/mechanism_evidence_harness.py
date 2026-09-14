"""E0 v0.6.1 mechanism-selection evidence harness (MECHANISM_SELECTION_EXECUTION_RELEASED).

One invocation = one arm/cell × one master seed × the full frozen recipe.

Cells: R1 | R4 | R16 | P-FROZEN | P-FT (P-FT@R1) | P-RANDOM (P-RANDOM@R1)
Conditional cells (P-RANDOM-FROZEN, P-FT@R16) are NOT runnable here; they
require authority-verified trigger states.

Frozen execution contract (v0.6.1 candidate + release record 90365137…):
  - Training stream: MSEL-ExampleStream-v1 with the MASTER SEED exactly as
    frozen (§5.3). The v0.5 TrainStream and the data_order substream are
    bootstrap-only and are NOT used here.
  - Corpus: frozen CMDR-MSEL-v0; split digests hard-verified against
    MSEL_MANIFEST.json at startup; rung membership from MSEL_RUNGS.json.
  - R arms (§5): exact v0.5 C0, AdamW β=(0.9,0.95) ε=1e-8 peak 5e-4, WD 0.05
    exclude_norm_bias, clip 1.0, warmup 400 → cosine to 10% at 8000,
    microbatch 16 × accumulation 8 = 128, 8,000 updates, 1,024,000
    presentations, FP32, no early stopping.
  - P-FT/P-RANDOM (§6.6/6.7): P0 backbone (44,670,976) + 512→3 bias
    classifier (1,539), native tokenizer, AdamW peak 5e-5, WD 0.01
    matrix-only, same schedule/batching.
  - P-FROZEN (§6.4): pretrained backbone FROZEN; classifier-only AdamW
    lr 1e-3 constant, WD 0, β=(0.9,0.95) ε=1e-8, clip 1.0; same stream,
    batching, checkpoint cadence.
  - RNG substreams (§8.3) govern backbone_init / classifier_init; the DATA
    stream uses the master seed via MSEL-ExampleStream-v1.
  - Checkpoint candidates every 400 updates; selection = dev_ID CMDR-SMA
    strictly-greater (earliest-update tie-break); eval_ID/eval_STRUCT never
    influence selection; no early termination.
  - Run status per §10.1; divergence fails closed (DIVERGED_SCIENTIFIC).
  - Full §19 telemetry per run.

Usage (MUST run under the frozen .venv):
  .venv/Scripts/python.exe scripts/e0/v061/mechanism_evidence_harness.py \
      --cell R1 --seed 806915476
  Rehearsal (diagnostic only, never evidence):
  ... --cell R1 --seed 806915476 --updates 400 --eval-every 200
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

import numpy as np
import torch
import torch.nn.functional as F

from msel_stream import PRESENTATIONS, MSELExampleStream
import msel_metrics
from q2_m0_qualify import (
    ADAM_EPS,
    BETAS,
    EFFECTIVE_BATCH,
    GRAD_ACCUM,
    GRAD_CLIP,
    MICROBATCH,
    collate,
    learning_rate,
    parameter_groups,
)
from gpu_smoke_harness import (
    EXPECTED_P0_BACKBONE,
    EXPECTED_P0_CLASSIFIER,
    P0_SNAP,
    build_p0_classification,
    collate_p0,
    p0_lr_at,
    p0_parameter_groups,
    p0_readout_logits,
    rng_substream,
)

REPO_ROOT = HERE.parent.parent.parent
CORPUS = REPO_ROOT / "local_data" / "e0_v061_msel"
DOCS = REPO_ROOT / "docs" / "experiments" / "e0" / "v061"
EVIDENCE_DIR = DOCS / "evidence"
MANIFEST = DOCS / "MSEL_MANIFEST.json"
RUNGS = CORPUS / "MSEL_RUNGS.json"
RUNTIME_FREEZE = DOCS / "GPU_RUNTIME_FREEZE.json"
RELEASE_RECORD = DOCS / "E0_v0.6.1_MECHANISM_SELECTION_EXECUTION_RELEASE.json"

# Frozen digests (release-record bindings)
CONTRACT_SHA = "a70a910e2781fe5c37d625f5474032efaba32855d16446412eb590d55caf7e5b"
ADDENDUM_SHA = "eecccdc3b6cb1e818d5084323c3101e5ce441bd9669ba97e0d91d74a038a950b"
METRICS_SHA = "dde9e4e9bae2237dcef45568d325e6bccd4c66d2845a43da33ed17143c4c4118"
RELEASE_SHA = "90365137f372b001324b2811d59f9277915dc7b64891e1f8a6620d176d6a07b8"
P0_WEIGHTS_SHA = "ebfa4e2f18696ebd83716a0d39fe2c025f2ff8483f72a83ca59c475692fc9d15"

UPDATES = 8_000
EVAL_EVERY = 400
LABELS = ["ENTAILED", "CONTRADICTED", "UNKNOWN"]
LABEL_TO_ID = {n: i for i, n in enumerate(LABELS)}
DEPTHS = (1, 2, 3, 4)
EVAL_BATCH = 128

P_FROZEN_LR = 1.0e-3


class DivergenceError(RuntimeError):
    """§10.1 DIVERGED_SCIENTIFIC — fail closed, no rescue."""

    def __init__(self, code: str, update: int, detail: str | None = None):
        super().__init__(f"{code} at update {update}" + (f" ({detail})" if detail else ""))
        self.code = code
        self.update = update


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def cjson(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def state_digest(named_tensors) -> str:
    h = hashlib.sha256()
    for name, tensor in named_tensors:
        t = tensor.detach().to("cpu", torch.float32).contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def require_finite(tensor: torch.Tensor, code: str, update: int) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise DivergenceError(code, update)


def verify_frozen_inputs() -> dict:
    """Hard-stop verification of every frozen input digest before model work."""
    verified = {}
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    split_digests = manifest["splits"]  # keys: "<split>_d<dep>" (no .jsonl suffix)
    for split in ("train_pool", "dev_ID", "eval_ID", "eval_STRUCT"):
        for dep in DEPTHS:
            key = f"{split}_d{dep}"
            path = CORPUS / f"{key}.jsonl"
            if not path.is_file():
                raise SystemExit(f"fail-closed: corpus file missing: {path}")
            actual = sha256_file(path)
            expected = split_digests[key]["sha256"]
            if actual != expected:
                raise SystemExit(f"fail-closed: {key} digest {actual} != frozen {expected}")
            verified[key] = actual
    rel = sha256_file(RELEASE_RECORD)
    if rel != RELEASE_SHA:
        raise SystemExit(f"fail-closed: release record digest {rel} != frozen {RELEASE_SHA}")
    verified["_release_record"] = rel
    return verified


def load_split(split: str, rung_families: dict | None = None) -> list[dict]:
    """Load one split across depths; optional rung family filter (train_pool)."""
    rows = []
    for dep in DEPTHS:
        with (CORPUS / f"{split}_d{dep}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if rung_families is not None and row["family_id"] not in rung_families[f"d{dep}"]:
                    continue
                rows.append(row)
    return rows


def load_rung_families(rung: str) -> dict:
    rungs_doc = json.loads(RUNGS.read_text(encoding="utf-8"))
    return {f"d{dep}": set(rungs_doc["family_ids"][f"d{dep}"][rung]) for dep in DEPTHS}


# ---------------------------------------------------------------- R-path data

def encode_lex(rows: list[dict]) -> list[dict]:
    from q1_cmdr_bootstrap import LEX
    from m0_model import MAX_LEN
    out = []
    rejected = 0
    for row in rows:
        ids = LEX.encode(row["rendered"], append_decide=True)
        if len(ids) > MAX_LEN:
            rejected += 1
            continue
        out.append({
            "sample_id": row["sample_id"],
            "token_ids": ids,
            "label_id": LABEL_TO_ID[row["gold_label"]],
            "depth": row["reasoning_depth_stratum"],
            "family_id": row["family_id"],
        })
    if rejected:
        raise SystemExit(f"fail-closed: {rejected} train examples over MAX_LEN — corpus contract violation")
    return out


def encode_lex_eval(rows: list[dict]) -> list[dict]:
    from q1_cmdr_bootstrap import LEX
    out = []
    for row in rows:
        ids = __import__("q1_cmdr_bootstrap").LEX.encode(row["rendered"], append_decide=True)
        out.append({
            "sample_id": row["sample_id"],
            "token_ids": ids,
            "label_id": LABEL_TO_ID[row["gold_label"]],
            "depth": row["reasoning_depth_stratum"],
            "family_id": row["family_id"],
        })
    return out


def raw_rows(rows: list[dict]) -> list[dict]:
    return [{
        "sample_id": row["sample_id"],
        "rendered_text": row["rendered"],
        "label_id": LABEL_TO_ID[row["gold_label"]],
        "depth": row["reasoning_depth_stratum"],
        "family_id": row["family_id"],
    } for row in rows]


# ------------------------------------------------------------------- runners

def predict_r(model, rows: list[dict], device) -> list[int]:
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(rows), EVAL_BATCH):
            b = rows[start:start + EVAL_BATCH]
            ids, decide, _ = collate(b, device)
            preds.extend(int(x) for x in model(ids, decide).argmax(-1).tolist())
    model.train()
    return preds


def predict_p(backbone, classifier, rows: list[dict], tok, device) -> list[int]:
    backbone.eval(); classifier.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(rows), EVAL_BATCH):
            b = rows[start:start + EVAL_BATCH]
            input_ids, attention_mask, _ = collate_p0(b, tok, device)
            logits = p0_readout_logits(backbone, classifier, input_ids, attention_mask)
            preds.extend(int(x) for x in logits.argmax(-1).tolist())
    backbone.train(); classifier.train()
    return preds


def run_r_cell(cell: str, seed: int, updates: int, eval_every: int, device) -> dict:
    from m0_model import build_model

    rung = cell  # R1 | R4 | R16
    train_rows = encode_lex(load_split("train_pool", load_rung_families(rung)))
    dev_rows = encode_lex_eval(load_split("dev_ID"))
    eval_id_rows = encode_lex_eval(load_split("eval_ID"))
    eval_struct_rows = encode_lex_eval(load_split("eval_STRUCT"))

    sample_ids = [r["sample_id"] for r in train_rows]
    by_sid = {r["sample_id"]: r for r in train_rows}
    stream = MSELExampleStream(sample_ids, seed)
    assert stream.total_updates(EFFECTIVE_BATCH) == 8_000

    backbone_seed = rng_substream(seed, "backbone_init")
    model = build_model("C0", backbone_seed, device)
    optimizer = torch.optim.AdamW(
        parameter_groups(model, "exclude_norm_bias"),
        lr=learning_rate(1), betas=BETAS, eps=ADAM_EPS)

    return train_loop_r(
        model, optimizer, stream, by_sid, dev_rows, eval_id_rows, eval_struct_rows,
        cell, seed, updates, eval_every, device,
        param_count=53_232_643, trainable_count=53_232_643,
        backbone_seed=backbone_seed, model_family="C0/M0-CausalDense-v1")


def train_loop_r(model, optimizer, stream, by_sid, dev_rows, eval_id_rows,
                 eval_struct_rows, cell, seed, updates, eval_every, device,
                 param_count, trainable_count, backbone_seed, model_family) -> dict:
    torch.cuda.reset_peak_memory_stats(device)
    history = []
    best = {"update": 0, "sma": float("-inf"), "state": None}
    nonpad_tokens = 0
    forward_tokens = 0
    divergence = None
    started = time.time()

    y_true_dev = [r["label_id"] for r in dev_rows]
    depths_dev = [r["depth"] for r in dev_rows]

    try:
        for update in range(1, updates + 1):
            lr = learning_rate(update)
            for g in optimizer.param_groups:
                g["lr"] = lr
            batch_sids = stream.take(EFFECTIVE_BATCH)
            optimizer.zero_grad(set_to_none=True)
            for micro in range(GRAD_ACCUM):
                rows = [by_sid[sid] for sid in batch_sids[micro * MICROBATCH:(micro + 1) * MICROBATCH]]
                ids, decide, labels = collate(rows, device)
                nonpad_tokens += sum(len(r["token_ids"]) for r in rows)
                forward_tokens += int(ids.numel())
                logits = model(ids, decide)
                loss = F.cross_entropy(logits, labels, label_smoothing=0.0)
                require_finite(loss, "nonfinite_microbatch_loss", update)
                (loss / GRAD_ACCUM).backward()
            for p in model.parameters():
                if p.grad is not None:
                    require_finite(p.grad, "nonfinite_accumulated_gradient", update)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            require_finite(gn, "nonfinite_grad_norm", update)
            optimizer.step()
            for p in model.parameters():
                require_finite(p.data, "nonfinite_parameters_after_step", update)

            if update % eval_every == 0:
                preds = predict_r(model, dev_rows, device)
                sma = msel_metrics.cmdr_sma(y_true_dev, preds, depths_dev)
                if not np.isfinite(sma):
                    raise DivergenceError("nonfinite_eval_metric", update)
                history.append({"update": update, "dev_ID_cmdr_sma": sma, "lr": lr})
                if sma > best["sma"]:
                    best = {"update": update, "sma": sma,
                            "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
                print(f"  [{cell} seed={seed}] update {update}/{updates} dev_ID SMA={sma:.6f} lr={lr:.3e}", flush=True)
    except DivergenceError as exc:
        divergence = {"code": exc.code, "update": exc.update}

    wall = time.time() - started
    final_state_digest = state_digest(model.state_dict().items())

    result = {
        "cell": cell, "seed": seed, "model_family": model_family,
        "param_count": param_count, "trainable_param_count": trainable_count,
        "train_pool": {"examples": len(by_sid)},
        "updates_requested": updates,
        "history": history,
        "nonpadding_token_presentations": nonpad_tokens,
        "forward_token_count": forward_tokens,
        "wall_seconds": round(wall, 1),
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "final_update_state_digest": final_state_digest,
        "divergence": divergence,
    }

    if divergence is not None or best["state"] is None:
        result["run_status"] = "DIVERGED_SCIENTIFIC" if divergence is not None else "INVALID_INFRASTRUCTURE"
        result["selected_checkpoint"] = None
        result["metrics"] = None
        return result

    # Restore the SELECTED checkpoint for all final evaluation.
    model.load_state_dict(best["state"])
    result["run_status"] = "VALID"
    result["selected_checkpoint"] = {"update": best["update"], "dev_ID_cmdr_sma": best["sma"]}
    result["selected_state_digest"] = state_digest(model.state_dict().items())
    result["metrics"] = final_metrics_r(model, eval_id_rows, eval_struct_rows, device)
    return result


def final_metrics_r(model, eval_id_rows, eval_struct_rows, device) -> dict:
    pred_id = predict_r(model, eval_id_rows, device)
    pred_struct = predict_r(model, eval_struct_rows, device)
    return assemble_metrics(eval_id_rows, pred_id, eval_struct_rows, pred_struct)


def assemble_metrics(id_rows, pred_id, struct_rows, pred_struct) -> dict:
    qm = msel_metrics.q_metrics(
        [r["label_id"] for r in id_rows], pred_id, [r["depth"] for r in id_rows],
        [r["label_id"] for r in struct_rows], pred_struct, [r["depth"] for r in struct_rows])
    q24 = msel_metrics.q_2_4(
        [r["label_id"] for r in id_rows], pred_id, [r["depth"] for r in id_rows],
        [r["label_id"] for r in struct_rows], pred_struct, [r["depth"] for r in struct_rows])
    recalls = msel_metrics.per_label_recall(
        [r["label_id"] for r in id_rows], pred_id, [r["depth"] for r in id_rows],
        [r["label_id"] for r in struct_rows], pred_struct, [r["depth"] for r in struct_rows])
    fec_id = msel_metrics.family_exact_consistency(
        [r["label_id"] for r in id_rows], pred_id, [r["family_id"] for r in id_rows])
    fec_struct = msel_metrics.family_exact_consistency(
        [r["label_id"] for r in struct_rows], pred_struct, [r["family_id"] for r in struct_rows])
    per_depth = {}
    for surface, rows, preds in (("ID", id_rows, pred_id), ("STRUCT", struct_rows, pred_struct)):
        for dep in DEPTHS:
            idx = [i for i, r in enumerate(rows) if r["depth"] == dep]
            acc = sum(1 for i in idx if preds[i] == rows[i]["label_id"]) / max(1, len(idx))
            per_depth[f"{surface}_d{dep}"] = round(acc, 6)
    return {
        "Q": qm["Q"], "Q_ID": qm["Q_ID"], "Q_STRUCT": qm["Q_STRUCT"], "Q_2_4": q24,
        "per_label_recall": recalls,
        "family_exact_consistency": {
            "ID": fec_id, "STRUCT": fec_struct,
            "aggregate": msel_metrics.family_exact_consistency_aggregate(fec_id, fec_struct)},
        "per_depth_accuracy": per_depth,
        "predictions": {"eval_ID_argmax": pred_id, "eval_STRUCT_argmax": pred_struct},
    }


def run_p_cell(cell: str, seed: int, updates: int, eval_every: int, device) -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(P0_SNAP))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rung = "R1"
    train_rows = raw_rows(load_split("train_pool", load_rung_families(rung)))
    dev_rows = raw_rows(load_split("dev_ID"))
    eval_id_rows = raw_rows(load_split("eval_ID"))
    eval_struct_rows = raw_rows(load_split("eval_STRUCT"))

    sample_ids = [r["sample_id"] for r in train_rows]
    by_sid = {r["sample_id"]: r for r in train_rows}
    stream = MSELExampleStream(sample_ids, seed)

    if cell == "P-FROZEN":
        backbone, classifier = build_p0_classification("P-FT", seed, device)
        for p in backbone.parameters():
            p.requires_grad_(False)
        backbone.eval()
        optimizer = torch.optim.AdamW(
            [{"params": list(classifier.parameters()), "weight_decay": 0.0}],
            lr=P_FROZEN_LR, betas=BETAS, eps=ADAM_EPS)  # constant schedule (§6.4)
        param_count = EXPECTED_P0_BACKBONE + EXPECTED_P0_CLASSIFIER
        trainable_count = EXPECTED_P0_CLASSIFIER
    else:
        arm = "P-FT" if cell == "P-FT" else "P-RANDOM"
        backbone, classifier = build_p0_classification(arm, seed, device)
        optimizer = torch.optim.AdamW(
            p0_parameter_groups(backbone, classifier),
            lr=p0_lr_at(1), betas=BETAS, eps=ADAM_EPS)
        param_count = EXPECTED_P0_BACKBONE + EXPECTED_P0_CLASSIFIER
        trainable_count = param_count

    return train_loop_p(
        backbone, classifier, optimizer, tok, cell, stream, by_sid, dev_rows,
        eval_id_rows, eval_struct_rows, seed, updates, eval_every, device,
        param_count=param_count, trainable_count=trainable_count)


def train_loop_p(backbone, classifier, optimizer, tok, cell, stream, by_sid,
                 dev_rows, eval_id_rows, eval_struct_rows, seed, updates,
                 eval_every, device, param_count, trainable_count) -> dict:
    frozen = cell == "P-FROZEN"
    torch.cuda.reset_peak_memory_stats(device)
    history = []
    best = {"update": 0, "sma": float("-inf"), "state": None}
    nonpad_tokens = 0
    forward_tokens = 0
    divergence = None
    started = time.time()

    y_true_dev = [r["label_id"] for r in dev_rows]
    depths_dev = [r["depth"] for r in dev_rows]

    def lr_at(update: int) -> float:
        return P_FROZEN_LR if frozen else p0_lr_at(update)

    try:
        for update in range(1, updates + 1):
            lr = lr_at(update)
            for g in optimizer.param_groups:
                g["lr"] = lr
            batch_sids = stream.take(EFFECTIVE_BATCH)
            optimizer.zero_grad(set_to_none=True)
            for micro in range(GRAD_ACCUM):
                rows = [by_sid[sid] for sid in batch_sids[micro * MICROBATCH:(micro + 1) * MICROBATCH]]
                input_ids, attention_mask, labels = collate_p0(rows, tok, device)
                nonpad_tokens += int(attention_mask.sum().item())
                forward_tokens += int(input_ids.numel())
                if frozen:
                    with torch.no_grad():
                        hidden = backbone(input_ids=input_ids,
                                          attention_mask=attention_mask).last_hidden_state
                    seq_lens = attention_mask.sum(dim=1) - 1
                    idx = torch.arange(hidden.size(0), device=device)
                    logits = classifier(hidden[idx, seq_lens])
                else:
                    logits = p0_readout_logits(backbone, classifier, input_ids, attention_mask)
                loss = F.cross_entropy(logits, labels, label_smoothing=0.0)
                require_finite(loss, "nonfinite_microbatch_loss", update)
                (loss / GRAD_ACCUM).backward()
            trainable = [p for p in backbone.parameters() if p.requires_grad] + list(classifier.parameters())
            for p in trainable:
                if p.grad is not None:
                    require_finite(p.grad, "nonfinite_accumulated_gradient", update)
            gn = torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
            require_finite(gn, "nonfinite_grad_norm", update)
            optimizer.step()
            for p in trainable:
                require_finite(p.data, "nonfinite_parameters_after_step", update)

            if update % eval_every == 0:
                preds = predict_p(backbone, classifier, dev_rows, tok, device)
                sma = msel_metrics.cmdr_sma(y_true_dev, preds, depths_dev)
                if not np.isfinite(sma):
                    raise DivergenceError("nonfinite_eval_metric", update)
                history.append({"update": update, "dev_ID_cmdr_sma": sma, "lr": lr})
                if sma > best["sma"]:
                    best = {"update": update, "sma": sma, "state": {
                        "backbone": {k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()},
                        "classifier": {k: v.detach().cpu().clone() for k, v in classifier.state_dict().items()},
                    }}
                print(f"  [{cell} seed={seed}] update {update}/{updates} dev_ID SMA={sma:.6f} lr={lr:.3e}", flush=True)
    except DivergenceError as exc:
        divergence = {"code": exc.code, "update": exc.update}

    wall = time.time() - started
    named_final = ([(f"backbone.{n}", p) for n, p in backbone.state_dict().items()]
                   + [(f"classifier.{n}", p) for n, p in classifier.state_dict().items()])
    final_digest = state_digest(named_final)

    result = {
        "cell": cell, "seed": seed, "model_family": "P0-70M/GPTNeoX-classification",
        "param_count": param_count, "trainable_param_count": trainable_count,
        "backbone_params": EXPECTED_P0_BACKBONE, "classifier_params": EXPECTED_P0_CLASSIFIER,
        "train_pool": {"examples": len(by_sid), "rung": "R1"},
        "updates_requested": updates,
        "history": history,
        "nonpadding_token_presentations": nonpad_tokens,
        "forward_token_count": forward_tokens,
        "wall_seconds": round(wall, 1),
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "final_update_state_digest": final_digest,
        "divergence": divergence,
    }

    if divergence is not None or best["state"] is None:
        result["run_status"] = "DIVERGED_SCIENTIFIC" if divergence is not None else "INVALID_INFRASTRUCTURE"
        result["selected_checkpoint"] = None
        result["metrics"] = None
        return result

    backbone.load_state_dict(best["state"]["backbone"])
    classifier.load_state_dict(best["state"]["classifier"])
    result["run_status"] = "VALID"
    result["selected_checkpoint"] = {"update": best["update"], "dev_ID_cmdr_sma": best["sma"]}
    named_sel = ([(f"backbone.{n}", p) for n, p in backbone.state_dict().items()]
                 + [(f"classifier.{n}", p) for n, p in classifier.state_dict().items()])
    result["selected_state_digest"] = state_digest(named_sel)
    pred_id = predict_p(backbone, classifier, eval_id_rows, tok, device)
    pred_struct = predict_p(backbone, classifier, eval_struct_rows, tok, device)
    result["metrics"] = assemble_metrics(eval_id_rows, pred_id, eval_struct_rows, pred_struct)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", required=True,
                        choices=["R1", "R4", "R16", "P-FROZEN", "P-FT", "P-RANDOM"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--updates", type=int, default=UPDATES,
                        help="rehearsal override; anything but 8000 is diagnostic-only")
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    args = parser.parse_args()

    if not args.seed in (806915476, 1031646469, 128439691, 555223894,
                         454204619, 1678768041):
        raise SystemExit(f"seed {args.seed} is not one of the six frozen primary seeds")
    rehearsal = args.updates != UPDATES or args.eval_every != EVAL_EVERY

    venv_ok = ".venv" in str(Path(sys.executable).resolve())
    if not venv_ok:
        raise SystemExit("fail-closed: run under the frozen .venv interpreter "
                         "(.venv/Scripts/python.exe) per GPU_RUNTIME_FREEZE.json")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable — frozen contract requires CUDA")
    device = torch.device("cuda")

    verified = verify_frozen_inputs()
    print(f"corpus digests verified ({len([k for k in verified if not k.startswith('_')])} files); "
          f"release record verified", flush=True)

    if args.cell.startswith("R"):
        result = run_r_cell(args.cell, args.seed, args.updates, args.eval_every, device)
    else:
        result = run_p_cell(args.cell, args.seed, args.updates, args.eval_every, device)

    from q2_m0_qualify import host_peak_memory_bytes
    result["peak_host_rss_bytes"] = host_peak_memory_bytes()
    result["deterministic_state"] = query_det()
    result["deterministic_state_verified"] = bool(verify_applied(DET_STATE))
    result["rng_substreams"] = {
        name: rng_substream(args.seed, name)
        for name in ("backbone_init", "classifier_init", "data_order", "dataloader_workers")}
    result["training_stream"] = {
        "algorithm": "MSEL-ExampleStream-v1",
        "seed_input": "master_seed",
        "presentations": PRESENTATIONS if not rehearsal else args.updates * EFFECTIVE_BATCH,
    }
    result["bindings"] = {
        "contract_sha256": CONTRACT_SHA,
        "spec_bound_addendum_sha256": ADDENDUM_SHA,
        "metric_implementation_sha256": METRICS_SHA,
        "release_record_sha256": RELEASE_SHA,
        "p0_weights_sha256": P0_WEIGHTS_SHA if not args.cell.startswith("R") else None,
        "runtime_freeze_sha256": sha256_file(RUNTIME_FREEZE),
        "code_git_commit": git_head(),
        "verified_corpus_digests": verified,
    }
    result["rehearsal"] = rehearsal
    if rehearsal:
        result["diagnostic_only_not_evidence"] = True

    out_dir = EVIDENCE_DIR / args.cell
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"seed{args.seed}" + ("_rehearsal" if rehearsal else "")
    out = out_dir / f"{args.cell}_{suffix}.json"
    out.write_text(json.dumps(result, indent=2, default=str) + "\n",
                   encoding="utf-8", newline="\n")
    if rehearsal:
        alt = REPO_ROOT / "local_data" / "rehearsal" / out.name
        alt.parent.mkdir(parents=True, exist_ok=True)
        alt.write_text(json.dumps(result, indent=2, default=str) + "\n",
                       encoding="utf-8", newline="\n")
        out.unlink()  # rehearsals never live in docs/
        out = alt
    print(f"\nrun_status: {result['run_status']}  report: {out}", flush=True)
    sys.exit(0 if result["run_status"] in ("VALID", "DIVERGED_SCIENTIFIC") else 1)


if __name__ == "__main__":
    main()
