"""E0 v0.6.1 mechanism-selection evidence harness (MECHANISM_SELECTION_EXECUTION_RELEASED).

REMEDIATED under V06-EVIDENCE-HARNESS-REMEDIATION-1:
  - emits the frozen E0-v0.6.1-MSEL-REPORT-v0 schema literally, validated
    against the frozen schema file before writing (fail closed);
  - selected-checkpoint train-surface diagnostics (§9.6/§19) over the unique
    frozen training membership (single-surface SMA / depth-2:4 / per-depth /
    per-label recalls / family exact consistency under metrics.train_surface);
  - predict_r/predict_p preserve caller module modes; P-FROZEN asserts the
    frozen backbone stays eval() (training == False) around every checkpoint
    evaluation;
  - startup provenance: exact starting commit captured with a REQUIRED clean
    tracked worktree; hard byte-level verification of rung artifact, report
    schema, stream implementation, metric implementation, runtime snapshot
    (pip-freeze binding), release record, 16 corpus splits, and P0 snapshot
    files for P cells; deterministic state verified BEFORE model construction;
  - code_sha256 = canonical executable-code manifest over the harness and the
    imported scientific implementation files (component hashes preserved);
  - wall/peak/RSS/token telemetry finalized only after eval_ID, eval_STRUCT,
    and train-surface diagnostics complete; estimated_flops: null.

Scientific core unchanged from the audited d09fde1 (combined all-depth
master-seed MSEL-ExampleStream-v1; dev_ID-only selection at 400-update
cadence with strict-greater/earliest tie-break; selected-state restore before
final scoring; §6.4/6.6/6.7 recipes; §10.1 statuses; fail-closed divergence).

Usage (MUST run under the frozen .venv):
  .venv/Scripts/python.exe scripts/e0/v061/mechanism_evidence_harness.py \
      --cell R1 --seed 806915476 [--attempt-id ...] [--incident-parent ...]
  Rehearsal (diagnostic only, never evidence):
  ... --updates 400 --eval-every 200
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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
REPORT_SCHEMA = DOCS / "E0_v0.6.1_MSEL_REPORT_SCHEMA.json"
P0_VERIFICATION = DOCS / "P0_SNAPSHOT_VERIFICATION.json"

# ---- frozen digests (authority/release-record + closure bindings) ----
CONTRACT_SHA = "a70a910e2781fe5c37d625f5474032efaba32855d16446412eb590d55caf7e5b"
ADDENDUM_SHA = "eecccdc3b6cb1e818d5084323c3101e5ce441bd9669ba97e0d91d74a038a950b"
METRICS_SHA = "dde9e4e9bae2237dcef45568d325e6bccd4c66d2845a43da33ed17143c4c4118"  # release record + closure
STREAM_SHA = "71d5233721dcff8542a0fa8bac8af6fd440a69d3646bb965df61cedb6a848a77"  # PREEXEC_BUNDLE modules map
RUNGS_SHA = "e83e6849e5e0db189e24df354c8d5ca8ade69de07a222cd126946ae07858db74"  # postmaterialization audit
SCHEMA_SHA = "933731003536c4fb91318ad44abb830e917eb3e1ed00b2f71ef83bac67ce8100"  # design-artifacts r1 manifest
RELEASE_SHA = "90365137f372b001324b2811d59f9277915dc7b64891e1f8a6620d176d6a07b8"
P0_WEIGHTS_SHA = "ebfa4e2f18696ebd83716a0d39fe2c025f2ff8483f72a83ca59c475692fc9d15"
RUNTIME_PIP_FREEZE_SHA = "ef6d062c81daf54a4a7a98aca1d2f4204476433e299c345d893a2f4bf0627011"  # release-record binding

REPORT_SCHEMA_ID = "E0-v0.6.1-MSEL-REPORT-v0"

UPDATES = 8_000
EVAL_EVERY = 400
LABELS = ["ENTAILED", "CONTRADICTED", "UNKNOWN"]
LABEL_TO_ID = {n: i for i, n in enumerate(LABELS)}
DEPTHS = (1, 2, 3, 4)
EVAL_BATCH = 128
P_FROZEN_LR = 1.0e-3

# Executable-code manifest: this harness + imported scientific implementation.
CODE_MANIFEST_FILES = (
    "scripts/e0/v061/mechanism_evidence_harness.py",
    "scripts/e0/v061/gpu_smoke_harness.py",
    "scripts/e0/v061/msel_stream.py",
    "scripts/e0/v061/msel_metrics.py",
    "scripts/e0/v061/deterministic_preamble.py",
    "scripts/e0/q2_m0_qualify.py",
    "scripts/e0/m0_model.py",
    "scripts/e0/q1_cmdr_bootstrap.py",
)

PRIMARY_SEEDS = (806915476, 1031646469, 128439691, 555223894, 454204619, 1678768041)


class DivergenceError(RuntimeError):
    """§10.1 DIVERGED_SCIENTIFIC — fail closed, no rescue."""

    def __init__(self, code: str, update: int, detail: str | None = None):
        super().__init__(f"{code} at update {update}" + (f" ({detail})" if detail else ""))
        self.code = code
        self.update = update


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_output(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                          text=True, check=True).stdout.strip()


def startup_provenance() -> dict:
    """Capture the exact starting commit; REQUIRE a clean tracked worktree."""
    head = git_output("rev-parse", "HEAD")
    porcelain = git_output("status", "--porcelain")
    if porcelain:
        raise SystemExit(f"fail-closed: tracked worktree not clean at start:\n{porcelain}")
    components = {}
    canon = hashlib.sha256()
    for rel in sorted(CODE_MANIFEST_FILES):
        digest = sha256_file(REPO_ROOT / rel)
        components[rel] = digest
        canon.update(rel.encode("utf-8"))
        canon.update(digest.encode("utf-8"))
    return {
        "repository_commit": head,
        "working_tree_clean_at_start": True,
        "code_sha256": canon.hexdigest(),
        "code_components": components,
    }


# ------------------------------------------------ schema validation (no deps)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_TYPE_MAP = {"object": dict, "array": list, "string": str, "boolean": bool,
             "number": (int, float), "integer": int, "null": type(None)}


def _check_type(value, tname: str) -> bool:
    py = _TYPE_MAP[tname]
    if tname == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if tname == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if tname == "boolean":
        return isinstance(value, bool)
    return isinstance(value, py)


def validate_against_schema(value, schema: dict, path: str = "$") -> list[str]:
    """Minimal JSON-Schema validator covering exactly the keywords the frozen
    report schema uses: type (incl. unions), const, enum, pattern, minimum,
    maximum, required, properties, additionalProperties, items."""
    errors: list[str] = []
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: const mismatch ({value!r} != {schema['const']!r})")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in enum {schema['enum']}")
    tdecl = schema.get("type")
    if tdecl is not None:
        tnames = tdecl if isinstance(tdecl, list) else [tdecl]
        if not any(_check_type(value, t) for t in tnames):
            errors.append(f"{path}: type {type(value).__name__} not in {tnames}")
            return errors
    if isinstance(value, str) and "pattern" in schema:
        if not re.search(schema["pattern"], value):
            errors.append(f"{path}: pattern mismatch {schema['pattern']}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value} > maximum {schema['maximum']}")
    if isinstance(value, dict):
        for req in schema.get("required", []):
            if req not in value:
                errors.append(f"{path}: missing required property {req!r}")
        props = schema.get("properties", {})
        addl = schema.get("additionalProperties", True)
        for k, v in value.items():
            if k in props:
                errors.extend(validate_against_schema(v, props[k], f"{path}.{k}"))
            elif isinstance(addl, dict):
                errors.extend(validate_against_schema(v, addl, f"{path}.{k}"))
            elif addl is False:
                errors.append(f"{path}: additional property {k!r} not allowed")
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            errors.extend(validate_against_schema(item, schema["items"], f"{path}[{i}]"))
    return errors


# ---------------------------------------------------- frozen-input verification

def verify_frozen_inputs(cell: str, provenance: dict) -> dict:
    """Hard byte-level verification of every frozen executable/data input."""
    verified: dict[str, str] = {}
    problems: list[str] = []

    def check_file(path: Path, expected: str, label: str):
        if not path.is_file():
            problems.append(f"{label}: missing {path}")
            return
        actual = sha256_file(path)
        if actual != expected:
            problems.append(f"{label}: {actual} != frozen {expected}")
        else:
            verified[label] = actual

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    verified["manifest"] = sha256_file(MANIFEST)
    for split in ("train_pool", "dev_ID", "eval_ID", "eval_STRUCT"):
        for dep in DEPTHS:
            key = f"{split}_d{dep}"
            check_file(CORPUS / f"{key}.jsonl", manifest["splits"][key]["sha256"], key)

    check_file(RUNGS, RUNGS_SHA, "rungs")
    check_file(REPORT_SCHEMA, SCHEMA_SHA, "report_schema")
    check_file(HERE / "msel_stream.py", STREAM_SHA, "msel_stream.py")
    check_file(HERE / "msel_metrics.py", METRICS_SHA, "msel_metrics.py")
    check_file(RELEASE_RECORD, RELEASE_SHA, "release_record")

    freeze = json.loads(RUNTIME_FREEZE.read_text(encoding="utf-8"))
    if freeze.get("runtime", {}).get("pip_freeze_sha256") != RUNTIME_PIP_FREEZE_SHA:
        problems.append("runtime_freeze: pip_freeze_sha256 != release-record binding")
    verified["runtime_freeze"] = sha256_file(RUNTIME_FREEZE)

    if not cell.startswith("R"):
        p0 = json.loads(P0_VERIFICATION.read_text(encoding="utf-8"))
        verified["p0_verification"] = sha256_file(P0_VERIFICATION)
        for name, digest in p0["file_digests"].items():
            check_file(P0_SNAP / name, digest, f"p0/{name}")

    if problems:
        raise SystemExit("fail-closed frozen-input verification:\n  " + "\n  ".join(problems))
    return verified


# ------------------------------------------------------------ corpus and data

def load_split(split: str, rung_families: dict | None = None) -> list[dict]:
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


def encode_lex(rows: list[dict], enforce_max_len: bool) -> list[dict]:
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
    if rejected and enforce_max_len:
        raise SystemExit(f"fail-closed: {rejected} TRAIN examples over MAX_LEN — corpus contract violation")
    if rejected:
        print(f"  WARNING: {rejected} eval examples over MAX_LEN excluded", flush=True)
    return out


def raw_rows(rows: list[dict]) -> list[dict]:
    return [{
        "sample_id": row["sample_id"],
        "rendered_text": row["rendered"],
        "label_id": LABEL_TO_ID[row["gold_label"]],
        "depth": row["reasoning_depth_stratum"],
        "family_id": row["family_id"],
    } for row in rows]


# ------------------------------------------------------------- mode-preserving

def predict_r(model, rows: list[dict], device) -> list[int]:
    was_training = model.training
    model.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(rows), EVAL_BATCH):
            b = rows[start:start + EVAL_BATCH]
            ids, decide, _ = collate(b, device)
            preds.extend(int(x) for x in model(ids, decide).argmax(-1).tolist())
    model.train(was_training)
    return preds


def predict_p(backbone, classifier, rows: list[dict], tok, device) -> list[int]:
    was_bb = backbone.training
    was_cls = classifier.training
    backbone.eval(); classifier.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, len(rows), EVAL_BATCH):
            b = rows[start:start + EVAL_BATCH]
            input_ids, attention_mask, _ = collate_p0(b, tok, device)
            logits = p0_readout_logits(backbone, classifier, input_ids, attention_mask)
            preds.extend(int(x) for x in logits.argmax(-1).tolist())
    backbone.train(was_bb); classifier.train(was_cls)
    return preds


# ----------------------------------------------------------------- prediction

def require_finite(tensor: torch.Tensor, code: str, update: int) -> None:
    if not bool(torch.isfinite(tensor).all()):
        raise DivergenceError(code, update)


def state_digest(named_tensors) -> str:
    h = hashlib.sha256()
    for name, tensor in named_tensors:
        t = tensor.detach().to("cpu", torch.float32).contiguous()
        h.update(name.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def assemble_metrics(id_rows, pred_id, struct_rows, pred_struct) -> dict:
    y_i = [r["label_id"] for r in id_rows]; d_i = [r["depth"] for r in id_rows]
    y_s = [r["label_id"] for r in struct_rows]; d_s = [r["depth"] for r in struct_rows]
    qm = msel_metrics.q_metrics(y_i, pred_id, d_i, y_s, pred_struct, d_s)
    q24 = msel_metrics.q_2_4(y_i, pred_id, d_i, y_s, pred_struct, d_s)
    recalls = msel_metrics.per_label_recall(y_i, pred_id, d_i, y_s, pred_struct, d_s)
    fec_id = msel_metrics.family_exact_consistency(y_i, pred_id, [r["family_id"] for r in id_rows])
    fec_s = msel_metrics.family_exact_consistency(y_s, pred_struct, [r["family_id"] for r in struct_rows])
    per_depth = {}
    for surface, rows, preds in (("ID", id_rows, pred_id), ("STRUCT", struct_rows, pred_struct)):
        for dep in DEPTHS:
            idx = [i for i, r in enumerate(rows) if r["depth"] == dep]
            per_depth[f"{surface}_d{dep}"] = round(
                sum(1 for i in idx if preds[i] == rows[i]["label_id"]) / max(1, len(idx)), 6)
    return {
        "Q": qm["Q"], "Q_ID": qm["Q_ID"], "Q_STRUCT": qm["Q_STRUCT"], "Q_2_4": q24,
        "minimum_label_recall": recalls["minimum_label_recall"],
        "per_label_recall": recalls,
        "family_exact_consistency": {
            "ID": fec_id, "STRUCT": fec_s,
            "aggregate": msel_metrics.family_exact_consistency_aggregate(fec_id, fec_s)},
        "per_depth_accuracy": per_depth,
        "predictions": {"eval_ID_argmax": pred_id, "eval_STRUCT_argmax": pred_struct},
    }


def train_surface_diagnostics(train_rows, preds) -> dict:
    """§9.6: same behavioral metrics on the unique training-membership surface.
    Two-surface helpers are applied with the single surface duplicated on both
    arguments — the duplicated cells leave the macro means unchanged, yielding
    exactly the single-surface statistic. Interpretation-only; cannot satisfy
    a transfer gate."""
    y = [r["label_id"] for r in train_rows]
    d = [r["depth"] for r in train_rows]
    sma = msel_metrics.cmdr_sma(y, preds, d)
    q24 = msel_metrics.q_2_4(y, preds, d, y, preds, d)
    recalls = msel_metrics.per_label_recall(y, preds, d, y, preds, d)
    fec = msel_metrics.family_exact_consistency(y, preds, [r["family_id"] for r in train_rows])
    per_depth = {}
    for dep in DEPTHS:
        idx = [i for i, r in enumerate(train_rows) if r["depth"] == dep]
        per_depth[f"d{dep}"] = round(
            sum(1 for i in idx if preds[i] == train_rows[i]["label_id"]) / max(1, len(idx)), 6)
    return {
        "note": "interpretation-only; unique frozen training membership; no transfer-gate value",
        "membership_examples": len(train_rows),
        "cmdr_sma": sma,
        "depth_2_4": q24,
        "per_depth_accuracy": per_depth,
        "per_label_recall": recalls,
        "minimum_label_recall": recalls["minimum_label_recall"],
        "family_exact_consistency": fec,
        "train_argmax": preds,
    }


def zeros_metrics() -> dict:
    """§10.1 divergence aggregation values."""
    return {
        "Q": 0.0, "Q_ID": 0.0, "Q_STRUCT": 0.0, "Q_2_4": 0.0,
        "minimum_label_recall": 0.0,
        "per_label_recall": {name: 0.0 for name in LABELS} | {"minimum_label_recall": 0.0},
        "family_exact_consistency": {"ID": 0.0, "STRUCT": 0.0, "aggregate": 0.0},
        "per_depth_accuracy": {},
        "predictions": None,
    }


# --------------------------------------------------------------- R-path runner

def run_r_cell(cell: str, seed: int, updates: int, eval_every: int, device) -> dict:
    from m0_model import build_model

    train_rows = encode_lex(load_split("train_pool", load_rung_families(cell)), enforce_max_len=True)
    dev_rows = encode_lex(load_split("dev_ID"), enforce_max_len=False)
    eval_id_rows = encode_lex(load_split("eval_ID"), enforce_max_len=False)
    eval_struct_rows = encode_lex(load_split("eval_STRUCT"), enforce_max_len=False)

    sample_ids = [r["sample_id"] for r in train_rows]
    by_sid = {r["sample_id"]: r for r in train_rows}
    stream = MSELExampleStream(sample_ids, seed)
    assert stream.total_updates(EFFECTIVE_BATCH) == 8_000

    backbone_seed = rng_substream(seed, "backbone_init")
    model = build_model("C0", backbone_seed, device)
    optimizer = torch.optim.AdamW(
        parameter_groups(model, "exclude_norm_bias"),
        lr=learning_rate(1), betas=BETAS, eps=ADAM_EPS)

    torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    history = []
    best = {"update": 0, "sma": float("-inf"), "state": None}
    nonpad_tokens = 0
    forward_tokens = 0
    divergence = None

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

    final_digest = state_digest(model.state_dict().items())

    result = {
        "cell": cell, "seed": seed, "updates_requested": updates, "history": history,
        "final_checkpoint_sha256": final_digest,
        "_counters": {"nonpad": nonpad_tokens, "forward": forward_tokens},
        "_started": started,
        "_device": device,
    }

    if divergence is not None or best["state"] is None:
        result["run_status"] = "DIVERGED_SCIENTIFIC" if divergence is not None else "INVALID_INFRASTRUCTURE"
        result["divergence"] = divergence
        result["metrics"] = zeros_metrics()
        result["_train_rows"] = None
        result["_selected"] = None
        return result

    model.load_state_dict(best["state"])
    pred_id = predict_r(model, eval_id_rows, device)
    pred_struct = predict_r(model, eval_struct_rows, device)
    metrics = assemble_metrics(eval_id_rows, pred_id, eval_struct_rows, pred_struct)
    train_preds = predict_r(model, train_rows, device)
    metrics["train_surface"] = train_surface_diagnostics(train_rows, train_preds)
    result["run_status"] = "VALID"
    result["metrics"] = metrics
    result["_train_rows"] = train_rows
    result["_selected"] = {"update": best["update"], "sma": best["sma"],
                           "sha256": state_digest(model.state_dict().items())}
    return result


# --------------------------------------------------------------- P-path runner

def run_p_cell(cell: str, seed: int, updates: int, eval_every: int, device) -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(P0_SNAP))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_rows = raw_rows(load_split("train_pool", load_rung_families("R1")))
    dev_rows = raw_rows(load_split("dev_ID"))
    eval_id_rows = raw_rows(load_split("eval_ID"))
    eval_struct_rows = raw_rows(load_split("eval_STRUCT"))

    sample_ids = [r["sample_id"] for r in train_rows]
    by_sid = {r["sample_id"]: r for r in train_rows}
    stream = MSELExampleStream(sample_ids, seed)

    frozen = cell == "P-FROZEN"
    if frozen:
        backbone, classifier = build_p0_classification("P-FT", seed, device)
        for p in backbone.parameters():
            p.requires_grad_(False)
        backbone.eval()
        optimizer = torch.optim.AdamW(
            [{"params": list(classifier.parameters()), "weight_decay": 0.0}],
            lr=P_FROZEN_LR, betas=BETAS, eps=ADAM_EPS)
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

    torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    history = []
    best = {"update": 0, "sma": float("-inf"), "state": None}
    nonpad_tokens = 0
    forward_tokens = 0
    divergence = None

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
                if frozen:
                    # frozen-backbone invariant (V06-EVIDENCE-HARNESS-REMEDIATION-1 directive 4)
                    assert backbone.training is False, "frozen backbone left training mode"
                preds = predict_p(backbone, classifier, dev_rows, tok, device)
                if frozen:
                    assert backbone.training is False, "frozen backbone entered training mode during eval"
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

    named_final = ([(f"backbone.{n}", p) for n, p in backbone.state_dict().items()]
                   + [(f"classifier.{n}", p) for n, p in classifier.state_dict().items()])
    final_digest = state_digest(named_final)

    result = {
        "cell": cell, "seed": seed, "updates_requested": updates, "history": history,
        "final_checkpoint_sha256": final_digest,
        "_counters": {"nonpad": nonpad_tokens, "forward": forward_tokens},
        "_started": started,
        "_device": device,
        "_param_count": param_count, "_trainable_count": trainable_count,
        "_tokenizer_len": len(tok),
    }

    if divergence is not None or best["state"] is None:
        result["run_status"] = "DIVERGED_SCIENTIFIC" if divergence is not None else "INVALID_INFRASTRUCTURE"
        result["divergence"] = divergence
        result["metrics"] = zeros_metrics()
        result["_train_rows"] = None
        result["_selected"] = None
        return result

    backbone.load_state_dict(best["state"]["backbone"])
    classifier.load_state_dict(best["state"]["classifier"])
    if frozen:
        backbone.eval()
    pred_id = predict_p(backbone, classifier, eval_id_rows, tok, device)
    pred_struct = predict_p(backbone, classifier, eval_struct_rows, tok, device)
    metrics = assemble_metrics(eval_id_rows, pred_id, eval_struct_rows, pred_struct)
    train_preds = predict_p(backbone, classifier, train_rows, tok, device)
    metrics["train_surface"] = train_surface_diagnostics(train_rows, train_preds)
    result["run_status"] = "VALID"
    result["metrics"] = metrics
    result["_train_rows"] = train_rows
    named_sel = ([(f"backbone.{n}", p) for n, p in backbone.state_dict().items()]
                 + [(f"classifier.{n}", p) for n, p in classifier.state_dict().items()])
    result["_selected"] = {"update": best["update"], "sma": best["sma"],
                           "sha256": state_digest(named_sel)}
    return result


# ------------------------------------------------------------------ report

def build_report(result: dict, cell: str, seed: int, provenance: dict,
                 verified: dict, rehearsal: bool, attempt_id: str,
                 incident_parent) -> dict:
    from q2_m0_qualify import host_peak_memory_bytes

    # Resource telemetry finalized AFTER all mandatory evaluations (directive 6)
    wall = time.time() - result["_started"]
    device = result["_device"]

    metrics = result["metrics"]
    train_rows = result.get("_train_rows")
    selected = result.get("_selected")

    if cell.startswith("R"):
        param_count = 53_232_643
        trainable_count = 53_232_643
        model_identity = {
            "family": "C0/M0-CausalDense-v1",
            "architecture": "15 layers, width 512, SwiGLU 1536, RMSNorm, RoPE",
            "tokenizer": "CMDR-Lex-v1",
        }
        train_membership = {"rung": cell, "surface": "ID", "examples": len(train_rows) if train_rows else None}
    else:
        param_count = result["_param_count"]
        trainable_count = result["_trainable_count"]
        model_identity = {
            "family": "P0-70M/GPTNeoX-classification",
            "backbone_params": EXPECTED_P0_BACKBONE,
            "classifier_params": EXPECTED_P0_CLASSIFIER,
            "tokenizer": "P0 native (pinned snapshot)",
            "tokenizer_len": result.get("_tokenizer_len"),
            "backbone_init": ("frozen_pretrained" if cell == "P-FROZEN"
                              else "pretrained" if cell == "P-FT" else "random_from_frozen_config"),
            "backbone_frozen": cell == "P-FROZEN",
        }
        train_membership = {"rung": "R1", "surface": "ID", "examples": len(train_rows) if train_rows else None}

    report = {
        "schema_id": REPORT_SCHEMA_ID,
        "contract_sha256": CONTRACT_SHA,
        "code_sha256": provenance["code_sha256"],
        "runtime_snapshot_sha256": verified["runtime_freeze"],
        "regime_id": cell,
        "master_seed": seed,
        "rng_substreams": {
            name: rng_substream(seed, name)
            for name in ("backbone_init", "classifier_init", "data_order", "dataloader_workers")},
        "run_status": result["run_status"],
        "attempt_id": attempt_id,
        "incident_parent": incident_parent,
        "working_tree_clean_at_start": provenance["working_tree_clean_at_start"],
        "repository_commit": provenance["repository_commit"],
        "dataset_digests": {
            k: v for k, v in verified.items() if not k.startswith("p0/")},
        "model_identity": model_identity,
        "parameter_count": param_count,
        "trainable_parameter_count": trainable_count,
        "selected_checkpoint_update": selected["update"] if selected else None,
        "selected_checkpoint_sha256": selected.get("sha256") if selected else None,
        "final_checkpoint_sha256": result["final_checkpoint_sha256"],
        "metrics": metrics,
        "resource": {
            "wall_seconds": round(wall, 1),
            "peak_accelerator_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "host_peak_rss_bytes": host_peak_memory_bytes(),
            "nonpadding_token_presentations": result["_counters"]["nonpad"],
            "forward_token_count": result["_counters"]["forward"],
            "estimated_flops": None,
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
        "training_stream": {
            "algorithm": "MSEL-ExampleStream-v1",
            "seed_input": "master_seed",
            "presentations_contract": PRESENTATIONS,
            "presentations_executed": result["updates_requested"] * EFFECTIVE_BATCH,
        },
        "train_membership": train_membership,
        "selection": {
            "metric": "dev_ID CMDR-SMA",
            "cadence_updates": EVAL_EVERY,
            "tie_break": "earliest_update",
            "history": result["history"],
        },
        "divergence": result.get("divergence"),
        "bindings": {
            "spec_bound_addendum_sha256": ADDENDUM_SHA,
            "metric_implementation_sha256": METRICS_SHA,
            "stream_implementation_sha256": STREAM_SHA,
            "rungs_sha256": RUNGS_SHA,
            "report_schema_sha256": SCHEMA_SHA,
            "release_record_sha256": RELEASE_SHA,
            "runtime_pip_freeze_sha256": RUNTIME_PIP_FREEZE_SHA,
        },
        "provenance_code_components": provenance["code_components"],
        "probe_geometry": None,
        "notes": [],
    }
    if selected:
        report["notes"].append(
            f"selected checkpoint update {selected['update']} dev_ID_cmdr_sma {selected['sma']}")
    if rehearsal:
        report["diagnostic_only_not_evidence"] = True
        report["notes"].append("REHEARSAL: reduced updates/eval cadence; not §17 evidence")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", required=True,
                        choices=["R1", "R4", "R16", "P-FROZEN", "P-FT", "P-RANDOM"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--updates", type=int, default=UPDATES)
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--attempt-id", default=None)
    parser.add_argument("--incident-parent", default=None)
    args = parser.parse_args()

    if args.seed not in PRIMARY_SEEDS:
        raise SystemExit(f"seed {args.seed} is not one of the six frozen primary seeds")
    rehearsal = args.updates != UPDATES or args.eval_every != EVAL_EVERY

    if ".venv" not in str(Path(sys.executable).resolve()):
        raise SystemExit("fail-closed: run under the frozen .venv interpreter "
                         "(.venv/Scripts/python.exe) per GPU_RUNTIME_FREEZE.json")

    # Directive 5: provenance captured at process start; clean tree REQUIRED.
    provenance = startup_provenance()
    print(f"start commit {provenance['repository_commit'][:12]} (clean tree) "
          f"code_sha256 {provenance['code_sha256'][:12]}", flush=True)

    if not verify_applied(DET_STATE):
        raise SystemExit("fail-closed: deterministic preamble not verified BEFORE model construction")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable — frozen contract requires CUDA")
    device = torch.device("cuda")

    verified = verify_frozen_inputs(args.cell, provenance)
    print(f"frozen inputs verified: {len(verified)} digests OK", flush=True)

    if args.cell.startswith("R"):
        result = run_r_cell(args.cell, args.seed, args.updates, args.eval_every, device)
    else:
        result = run_p_cell(args.cell, args.seed, args.updates, args.eval_every, device)

    attempt_id = args.attempt_id or f"{args.cell}|{args.seed}|{int(time.time())}"
    report = build_report(result, args.cell, args.seed, provenance, verified,
                          rehearsal, attempt_id, args.incident_parent)

    # Directive 2: fail closed unless the frozen report schema is satisfied.
    schema = json.loads(REPORT_SCHEMA.read_text(encoding="utf-8"))
    errors = validate_against_schema(report, schema)
    if errors:
        for e in errors:
            print(f"SCHEMA VIOLATION: {e}", flush=True)
        out = REPO_ROOT / "local_data" / "schema_rejected" / f"{args.cell}_seed{args.seed}_rejected.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str) + "\n",
                       encoding="utf-8", newline="\n")
        raise SystemExit(f"fail-closed: report violates frozen schema ({len(errors)} errors); "
                          f"rejected draft preserved at {out}")

    if rehearsal:
        out = REPO_ROOT / "local_data" / "rehearsal" / f"{args.cell}_seed{args.seed}_rehearsal.json"
        out.parent.mkdir(parents=True, exist_ok=True)
    else:
        out = EVIDENCE_DIR / args.cell / f"{args.cell}_seed{args.seed}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str) + "\n",
                   encoding="utf-8", newline="\n")
    print(f"\nrun_status: {report['run_status']}  report: {out}", flush=True)
    sys.exit(0 if report["run_status"] in ("VALID", "DIVERGED_SCIENTIFIC") else 1)


if __name__ == "__main__":
    main()
