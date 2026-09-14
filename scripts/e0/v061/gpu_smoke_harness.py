"""V06-GPU-BOOTSTRAP resource smokes — R16/C0, P-FT@R1/P0, P-RANDOM@R1/P0.

Three 400-update resource smokes on the burned smoke corpus
(local_data/e0_v061_smoke, namespace ExpertForge-E0-v061-smoke-burn,
validated disjoint from the frozen MSEL burn index by
generate_smoke_replay_corpora.py).

Frozen scientific configuration (v0.6.1 contract candidate §§5.1-5.2, 6.1-6.7,
8.1, 8.3):
  - R arm: exact v0.5 C0 implementation, AdamW β=(0.9,0.95) ε=1e-8,
    peak LR 5e-4, WD 0.05 exclude_norm_bias, clip 1.0, warmup 400,
    linear-then-cosine to 10% at update 8000, microbatch 16, accumulation 8,
    effective batch 128, label smoothing 0, no early stopping, FP32 (v0.5
    execution contract — no autocast anywhere in v0.5 or v0.6.1).
  - P arms: P0 backbone = GPTNeoX backbone without LM head (44,670,976),
    bias-enabled linear 512→3 classifier (1,539), native tokenizer,
    AdamW peak LR 5e-5, WD 0.01 matrix_weights_only_no_norm_or_bias,
    same schedule/batching.
  - RNG substreams: uint31(SHA256("ExpertForge-E0-v061-rng|<master>|<name>")).
  - Deterministic preamble applied before any CUDA operation.

The dev accuracy recorded here is a resource/trajectory sanity readout on a
burned namespace. It is NOT capability evidence and must not be cited as
such (authority constraint: no MSEL eval metric used to characterize model
capability during smokes).

Usage:
  python scripts/e0/v061/gpu_smoke_harness.py --arm all
  python scripts/e0/v061/gpu_smoke_harness.py --arm r16
  python scripts/e0/v061/gpu_smoke_harness.py --arm p-ft
  python scripts/e0/v061/gpu_smoke_harness.py --arm p-random
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
sys.path.insert(0, str(HERE))            # v061: deterministic_preamble
sys.path.insert(0, str(HERE.parent))     # e0: m0_model, q2_m0_qualify, q1_cmdr_bootstrap

from deterministic_preamble import apply as apply_det, query as query_det, verify_applied

DET_STATE = apply_det()  # BEFORE torch import / CUDA probe

import numpy as np
import torch
import torch.nn.functional as F

# v0.5 frozen helpers, imported verbatim — no reimplementation drift.
from q2_m0_qualify import (
    BETAS,
    ADAM_EPS,
    GRAD_CLIP,
    MICROBATCH,
    GRAD_ACCUM,
    EFFECTIVE_BATCH,
    TrainStream,
    collate,
    host_peak_memory_bytes,
    learning_rate,
    parameter_groups,
)

REPO_ROOT = HERE.parent.parent.parent
SMOKE_DIR = REPO_ROOT / "local_data" / "e0_v061_smoke"
DOCS = REPO_ROOT / "docs" / "experiments" / "e0" / "v061"
P0_SNAP = Path(
    r"C:\huggingface_cache\hub\models--EleutherAI--pythia-70m\snapshots"
    r"\a39f36b100fe8a5377810d56c3f4789b9c53ac42"
)

FIRST_PRIMARY_SEED = 806915476  # §8.1 index 0
UPDATES_SMOKE = 400
P0_PEAK_LR = 5.0e-5
P0_WEIGHT_DECAY = 0.01
P0_NATIVE_MAX_LEN = 384  # §6.2 interface bound (truncation = 0 required)
EXPECTED_P0_BACKBONE = 44_670_976
EXPECTED_P0_CLASSIFIER = 1_539
LABELS = ["ENTAILED", "CONTRADICTED", "UNKNOWN"]
LABEL_TO_ID = {n: i for i, n in enumerate(LABELS)}

# Resource gates (V06-GPU-BOOTSTRAP-EXECUTION-AUTHORIZED):
#   t400 ≤ 19.2 min  (1.25 × 20 × t400 ≤ 8 h projection)
#   peak CUDA memory ≤ 12 GiB; host RSS ≤ 64 GiB
GATE_T400_MIN = 19.2
GATE_VRAM_GIB = 12.0
GATE_RSS_GIB = 64.0


def cjson(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def rng_substream(master_seed: int, name: str) -> int:
    """§8.3 frozen derivation: uint31(SHA256('ExpertForge-E0-v061-rng|<master>|<name>'))."""
    payload = f"ExpertForge-E0-v061-rng|{master_seed}|{name}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


def load_smoke_rows(data_dir: Path = SMOKE_DIR) -> tuple[list[dict], list[dict]]:
    """Load train/dev JSONL, encode with CMDR-Lex-v1 (v0.5 load_split path)."""
    from q1_cmdr_bootstrap import LEX
    from m0_model import MAX_LEN

    def load(name: str) -> list[dict]:
        rows = []
        rejected = 0
        with (data_dir / f"{name}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                ids = LEX.encode(row["rendered"], append_decide=True)
                if len(ids) > MAX_LEN:
                    rejected += 1
                    continue
                rows.append({
                    "sample_id": row["sample_id"],
                    "token_ids": ids,
                    "label_id": LABEL_TO_ID[row["gold_label"]],
                    "depth": row["reasoning_depth_stratum"],
                })
        if rejected:
            print(f"  WARNING: {name}: {rejected} examples over MAX_LEN={MAX_LEN} skipped", flush=True)
        return rows

    return load("train"), load("dev")


def percentile(values: list[int], q: float) -> int:
    arr = np.sort(np.asarray(values, dtype=np.int64))
    idx = min(len(arr) - 1, max(0, int(np.ceil(q * len(arr))) - 1))
    return int(arr[idx])


def run_r16_smoke(seed: int) -> dict:
    """C0/R-path 400-update smoke (exact v0.5 recipe, burned corpus)."""
    from m0_model import build_model
    from q2_m0_qualify import evaluate

    device = torch.device("cuda")
    train_rows, dev_rows = load_smoke_rows()
    print(f"  train={len(train_rows)} dev={len(dev_rows)} examples", flush=True)

    backbone_seed = rng_substream(seed, "backbone_init")
    data_seed = rng_substream(seed, "data_order")
    model = build_model("C0", backbone_seed, device)  # asserts 53,232,643 params
    optimizer = torch.optim.AdamW(
        parameter_groups(model, "exclude_norm_bias"),
        lr=learning_rate(1), betas=BETAS, eps=ADAM_EPS,
    )
    stream = TrainStream(len(train_rows), data_seed)

    torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    eval_record = None
    for update in range(1, UPDATES_SMOKE + 1):
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
        if update % UPDATES_SMOKE == 0:
            # cadence eval INSIDE the timed window (production: one eval per 400 updates)
            ev = evaluate(model, dev_rows, device, update)
            eval_record = {"update": update, "cmdr_sma": ev["cmdr_sma"], "lr": lr}
            print(f"    update {update}: dev SMA={ev['cmdr_sma']:.4f} lr={lr:.3e}", flush=True)
    wall = time.time() - started

    peak_alloc = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    result = {
        "arm": "R16_C0",
        "seed": seed,
        "backbone_init_substream_seed": backbone_seed,
        "data_order_substream_seed": data_seed,
        "param_count": 53_232_643,
        "trainable_param_count": 53_232_643,
        "updates": UPDATES_SMOKE,
        "train_examples": len(train_rows),
        "dev_examples": len(dev_rows),
        "train_wall_seconds": round(wall, 1),
        "train_wall_minutes": round(wall / 60, 2),
        "projected_8000_updates_minutes": round(wall / 60 * 25, 1),  # 20x duration + 25% safety
        "peak_cuda_allocated_bytes": peak_alloc,
        "peak_cuda_reserved_bytes": peak_reserved,
        "peak_cuda_allocated_gib": round(peak_alloc / 2**30, 3),
        "peak_cuda_reserved_gib": round(peak_reserved / 2**30, 3),
        "peak_host_rss_bytes": host_peak_memory_bytes(),
        "smoke_dev_readout_NOT_capability_evidence": eval_record,
        "deterministic_state": query_det(),
    }
    result["gate_t400"] = result["train_wall_minutes"] <= GATE_T400_MIN
    result["gate_vram"] = result["peak_cuda_reserved_gib"] <= GATE_VRAM_GIB
    rss = result["peak_host_rss_bytes"]
    result["gate_rss"] = rss is None or rss / 2**30 <= GATE_RSS_GIB
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def load_smoke_raw(data_dir: Path = SMOKE_DIR) -> tuple[list[dict], list[dict]]:
    """Raw rendered text rows for the P0 native tokenizer path."""
    def load(name: str) -> list[dict]:
        rows = []
        with (data_dir / f"{name}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                rows.append({
                    "sample_id": row["sample_id"],
                    "rendered_text": row["rendered"],
                    "label_id": LABEL_TO_ID[row["gold_label"]],
                    "depth": row["reasoning_depth_stratum"],
                })
        return rows
    return load("train"), load("dev")


def p0_token_audit(rows_raw: list[dict], tok) -> dict:
    """§6.2-style token audit over the smoke corpus for the P0 tokenizer."""
    lengths = []
    unk_id = tok.unk_token_id
    unk_count = 0
    truncation = 0
    for row in rows_raw:
        ids = tok.encode(row["rendered_text"], add_special_tokens=False)
        lengths.append(len(ids))
        if unk_id is not None:
            unk_count += sum(1 for t in ids if t == unk_id)
        if len(ids) > P0_NATIVE_MAX_LEN:
            truncation += 1
    return {
        "count": len(lengths),
        "min": int(min(lengths)),
        "median": float(np.median(lengths)),
        "p95": percentile(lengths, 0.95),
        "p99": percentile(lengths, 0.99),
        "max": int(max(lengths)),
        "truncation_count": truncation,
        "unknown_token_count": unk_count,
    }


def collate_p0(rows: list[dict], tok, device):
    texts = [r["rendered_text"] for r in rows]
    labels = torch.tensor([r["label_id"] for r in rows], dtype=torch.long)
    enc = tok(texts, return_tensors="pt", padding=True, truncation=False,
              add_special_tokens=False)
    return (enc["input_ids"].to(device), enc["attention_mask"].to(device),
            labels.to(device))


def p0_parameter_groups(backbone: torch.nn.Module, classifier: torch.nn.Module) -> list[dict]:
    """§6.6 WD scope: matrix_weights_only_no_norm_or_bias — decay iff weight dim > 1."""
    decay, no_decay = [], []
    for _, param in backbone.named_parameters():
        (decay if param.dim() > 1 else no_decay).append(param)
    for _, param in classifier.named_parameters():
        (decay if param.dim() > 1 else no_decay).append(param)
    return [
        {"params": decay, "weight_decay": P0_WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_p0_classification(arm: str, seed: int, device):
    """Return (backbone, classifier) per §6.1: GPTNeoX backbone sans LM head
    (44,670,976) + bias-enabled 512→3 classifier (1,539)."""
    from transformers import AutoConfig, AutoModelForCausalLM

    if arm == "P-FT":
        full = AutoModelForCausalLM.from_pretrained(str(P0_SNAP), torch_dtype=torch.float32)
    elif arm == "P-RANDOM":
        cfg = AutoConfig.from_pretrained(str(P0_SNAP))
        torch.manual_seed(rng_substream(seed, "backbone_init"))
        # pythia config.json carries torch_dtype float16; override to the
        # frozen FP32 execution contract (no silent FP16 substitution).
        full = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.float32)
    else:
        raise ValueError(arm)
    backbone = full.gpt_neox  # embed_in + layers + final_layer_norm
    backbone_params = sum(p.numel() for p in backbone.parameters())
    if backbone_params != EXPECTED_P0_BACKBONE:
        del full
        raise AssertionError(f"P0 backbone params {backbone_params} != frozen {EXPECTED_P0_BACKBONE}")
    # transformers 4.50 names the untied LM head `embed_out`; later versions
    # rename it `lm_head`. Remove whichever exists.
    for head_name in ("lm_head", "embed_out"):
        if hasattr(full, head_name):
            delattr(full, head_name)
    torch.manual_seed(rng_substream(seed, "classifier_init"))
    classifier = torch.nn.Linear(backbone.config.hidden_size, 3, bias=True)
    cls_params = sum(p.numel() for p in classifier.parameters())
    if cls_params != EXPECTED_P0_CLASSIFIER:
        raise AssertionError(f"classifier params {cls_params} != frozen {EXPECTED_P0_CLASSIFIER}")
    backbone = backbone.to(device)
    classifier = classifier.to(device)
    dtypes = {p.dtype for p in backbone.parameters()} | {p.dtype for p in classifier.parameters()}
    if dtypes != {torch.float32}:
        raise AssertionError(f"FP32 execution contract violated, dtypes={dtypes}")
    return backbone, classifier


def p0_readout_logits(backbone, classifier, input_ids, attention_mask):
    """Final hidden state (post final_layer_norm) at the last non-pad token → classifier."""
    hidden = backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    seq_lens = attention_mask.sum(dim=1) - 1
    idx = torch.arange(hidden.size(0), device=hidden.device)
    return classifier(hidden[idx, seq_lens])


def p0_lr_at(update: int) -> float:
    """v0.5 schedule shape (warmup 400, cosine to 10% at 8000) at the P0 peak."""
    return learning_rate(update) * (P0_PEAK_LR / 5.0e-4)


def run_p0_smoke(arm: str, seed: int) -> dict:
    """P-FT@R1 or P-RANDOM@R1 400-update smoke (§6.6/6.7 recipe, burned corpus)."""
    from transformers import AutoTokenizer

    device = torch.device("cuda")
    train_raw, dev_raw = load_smoke_raw()
    tok = AutoTokenizer.from_pretrained(str(P0_SNAP))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    audit_train = p0_token_audit(train_raw, tok)
    print(f"  tokens[train] min={audit_train['min']} med={audit_train['median']:.0f} "
          f"p95={audit_train['p95']} p99={audit_train['p99']} max={audit_train['max']} "
          f"trunc={audit_train['truncation_count']} unk={audit_train['unknown_token_count']}", flush=True)
    if audit_train["truncation_count"] or audit_train["unknown_token_count"]:
        raise SystemExit(f"P0 interface bound violated on smoke corpus: {audit_train}")

    backbone, classifier = build_p0_classification(arm, seed, device)
    backbone_params = sum(p.numel() for p in backbone.parameters())
    cls_params = sum(p.numel() for p in classifier.parameters())
    optimizer = torch.optim.AdamW(
        p0_parameter_groups(backbone, classifier),
        lr=p0_lr_at(1), betas=BETAS, eps=ADAM_EPS,
    )

    data_seed = rng_substream(seed, "data_order")
    stream = TrainStream(len(train_raw), data_seed)

    torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    eval_record = None
    nonpad_tokens = 0
    for update in range(1, UPDATES_SMOKE + 1):
        lr = p0_lr_at(update)
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch_indices = stream.take(EFFECTIVE_BATCH)
        optimizer.zero_grad(set_to_none=True)
        for micro in range(GRAD_ACCUM):
            rows = [train_raw[i] for i in batch_indices[micro * MICROBATCH:(micro + 1) * MICROBATCH]]
            input_ids, attention_mask, labels = collate_p0(rows, tok, device)
            nonpad_tokens += int(attention_mask.sum().item())
            logits = p0_readout_logits(backbone, classifier, input_ids, attention_mask)
            loss = F.cross_entropy(logits, labels, label_smoothing=0.0)
            (loss / GRAD_ACCUM).backward()
        params = list(backbone.parameters()) + list(classifier.parameters())
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        optimizer.step()
        if update % UPDATES_SMOKE == 0:
            backbone.eval(); classifier.eval()
            correct = 0
            with torch.no_grad():
                for start in range(0, len(dev_raw), 64):
                    b = dev_raw[start:start + 64]
                    input_ids, attention_mask, labels = collate_p0(b, tok, device)
                    logits = p0_readout_logits(backbone, classifier, input_ids, attention_mask)
                    correct += int((logits.argmax(-1) == labels).sum().item())
            acc = correct / len(dev_raw)
            eval_record = {"update": update, "dev_acc": acc, "lr": lr}
            print(f"    update {update}: dev acc={acc:.4f} lr={lr:.3e}", flush=True)
            backbone.train(); classifier.train()
    wall = time.time() - started

    peak_alloc = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    result = {
        "arm": arm,
        "seed": seed,
        "backbone_init": (
            "pretrained_ebfa4e2f18696ebd83716a0d39fe2c025f2ff8483f72a83ca59c475692fc9d15"
            if arm == "P-FT"
            else f"random_from_frozen_config_substream_{rng_substream(seed, 'backbone_init')}"
        ),
        "classifier_init_substream_seed": rng_substream(seed, "classifier_init"),
        "data_order_substream_seed": data_seed,
        "param_count": backbone_params + cls_params,
        "trainable_param_count": backbone_params + cls_params,
        "backbone_params": backbone_params,
        "classifier_params": cls_params,
        "updates": UPDATES_SMOKE,
        "train_examples": len(train_raw),
        "dev_examples": len(dev_raw),
        "token_audit_train": audit_train,
        "nonpadding_token_presentations": nonpad_tokens,
        "train_wall_seconds": round(wall, 1),
        "train_wall_minutes": round(wall / 60, 2),
        "projected_8000_updates_minutes": round(wall / 60 * 25, 1),
        "peak_cuda_allocated_bytes": peak_alloc,
        "peak_cuda_reserved_bytes": peak_reserved,
        "peak_cuda_allocated_gib": round(peak_alloc / 2**30, 3),
        "peak_cuda_reserved_gib": round(peak_reserved / 2**30, 3),
        "peak_host_rss_bytes": host_peak_memory_bytes(),
        "smoke_dev_readout_NOT_capability_evidence": eval_record,
        "deterministic_state": query_det(),
    }
    result["gate_t400"] = result["train_wall_minutes"] <= GATE_T400_MIN
    result["gate_vram"] = result["peak_cuda_reserved_gib"] <= GATE_VRAM_GIB
    rss = result["peak_host_rss_bytes"]
    result["gate_rss"] = rss is None or rss / 2**30 <= GATE_RSS_GIB
    del backbone, classifier, optimizer
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["r16", "p-ft", "p-random", "all"], default="all")
    parser.add_argument("--seed", type=int, default=FIRST_PRIMARY_SEED)
    args = parser.parse_args()

    if not SMOKE_DIR.is_dir():
        raise SystemExit(f"smoke corpus missing: {SMOKE_DIR} (run generate_smoke_replay_corpora.py)")
    corpus_digests = {
        "train.jsonl": sha256_file(SMOKE_DIR / "train.jsonl"),
        "dev.jsonl": sha256_file(SMOKE_DIR / "dev.jsonl"),
    }
    gen_evidence = DOCS / "SMOKE_REPLAY_CORPORA_EVIDENCE.json"
    if gen_evidence.is_file():
        gen = json.loads(gen_evidence.read_text(encoding="utf-8"))
        bound = gen.get("smoke_corpus", {})
        if (bound.get("train_sha256") != corpus_digests["train.jsonl"]
                or bound.get("dev_sha256") != corpus_digests["dev.jsonl"]):
            raise SystemExit("smoke corpus digests do not match generator evidence — fail closed")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable — GPU bootstrap requires CUDA")

    results = []
    if args.arm in ("r16", "all"):
        print("=== R16/C0 smoke (400 updates) ===", flush=True)
        r = run_r16_smoke(args.seed)
        results.append(r)
        print(f"  t400={r['train_wall_minutes']}min peak_reserved={r['peak_cuda_reserved_gib']}GiB "
              f"gates t400/vram/rss = {r['gate_t400']}/{r['gate_vram']}/{r['gate_rss']}", flush=True)
    if args.arm in ("p-ft", "all"):
        print("=== P-FT@R1 smoke (400 updates) ===", flush=True)
        r = run_p0_smoke("P-FT", args.seed)
        results.append(r)
        print(f"  t400={r['train_wall_minutes']}min peak_reserved={r['peak_cuda_reserved_gib']}GiB "
              f"gates t400/vram/rss = {r['gate_t400']}/{r['gate_vram']}/{r['gate_rss']}", flush=True)
    if args.arm in ("p-random", "all"):
        print("=== P-RANDOM@R1 smoke (400 updates) ===", flush=True)
        r = run_p0_smoke("P-RANDOM", args.seed)
        results.append(r)
        print(f"  t400={r['train_wall_minutes']}min peak_reserved={r['peak_cuda_reserved_gib']}GiB "
              f"gates t400/vram/rss = {r['gate_t400']}/{r['gate_vram']}/{r['gate_rss']}", flush=True)

    evidence = {
        "schema_id": "E0-V061-GPU-SMOKE-EVIDENCE-v0",
        "authority": "V06-GPU-BOOTSTRAP-EXECUTION-AUTHORIZED",
        "code_git_commit": git_head(),
        "seed": args.seed,
        "arms": [r["arm"] for r in results],
        "corpus_digests": corpus_digests,
        "gates": {
            "t400_max_minutes": GATE_T400_MIN,
            "peak_vram_max_gib": GATE_VRAM_GIB,
            "host_rss_max_gib": GATE_RSS_GIB,
            "projection": "1.25 x 20 x t400 <= 8h",
        },
        "smokes": results,
        "deterministic_state_at_start": DET_STATE,
        "deterministic_state_verified": verify_applied(DET_STATE),
        "status": None,
    }
    all_pass = bool(results) and all(r["gate_t400"] and r["gate_vram"] and r["gate_rss"] for r in results)
    evidence["status"] = "PASS" if all_pass else "FAIL"

    out = DOCS / "GPU_SMOKE_EVIDENCE.json"
    out.write_text(json.dumps(evidence, indent=2, default=str) + "\n",
                   encoding="utf-8", newline="\n")
    print(f"\nSTATUS: {evidence['status']}  evidence: {out}", flush=True)
    sys.exit(0 if evidence["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
