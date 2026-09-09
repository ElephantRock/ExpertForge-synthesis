"""E0 Q2-UNQUALIFIED diagnostics — D0/D1/D2 only (issue #3, non-scientific).

Branch: e0/q2-unqualified-diagnostics
Authority: issue #3 "E0 Q2 UNQUALIFIED — root-cause diagnostics before any
v0.6 proposal". Every output carries diagnostic_only/non_scientific flags.
Nothing in this module writes to, edits, or overloads v0.5 qualification
records; results go to docs/experiments/e0/diagnostics/.

D0  training-dynamics instrumentation on the exact production path
D1  tiny balanced counterfactual-family memorization test (key falsifier)
D2  independent label / <DECIDE> / loss-transport audit
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from m0_model import CANDIDATES
from q2_m0_qualify import (
    BETAS,
    EVAL_BATCH,
    FROZEN_SEEDS,
    GRAD_ACCUM,
    GRAD_CLIP,
    LABEL_TO_ID,
    LABELS,
    MICROBATCH,
    PEAK_LR,
    UPDATES,
    FINAL_LR_FRACTION,
    WARMUP,
    TrainStream,
    collate,
    evaluate,
    learning_rate,
    load_split,
    parameter_groups,
    scientific_setup,
    sha256_file,
    subprocess_git_head,
)

REPO_ROOT = (Path(__file__).resolve().parent.parent.parent).resolve()
DIAG_DIR = REPO_ROOT / "docs" / "experiments" / "e0" / "diagnostics"
DIAG_SEED = FROZEN_SEEDS[0]  # 1647674144 — first frozen qualification seed


def diag_header(extra: dict | None = None) -> dict:
    header = {
        "diagnostic_only": True,
        "non_scientific": True,
        "git_commit": subprocess_git_head(REPO_ROOT),
        "branch": subprocess.run(
            ["git", "branch", "--show-current"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip(),
        "runtime_snapshot_sha256": sha256_file(
            REPO_ROOT / "docs" / "experiments" / "e0" / "q2_q3_runtime.snapshot.json"
        ),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "authority": "issue #3 D0-D2; no v0.5 record is modified",
    }
    if extra:
        header.update(extra)
    return header


def write_diag(name: str, payload: dict) -> Path:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    path = DIAG_DIR / name
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {path}")
    return path


# --------------------------------------------------------------------- D0 ----

def run_d0(updates: int = 2000, eval_every: int = 400) -> dict:
    """Instrumented replay of the exact production training path (C0, frozen
    seed, exclude_norm_bias). Instrumentation is read-only with respect to the
    training math: it records tensors the production loop already computes."""
    ctx = scientific_setup("C0", "exclude_norm_bias", DIAG_SEED,
                           REPO_ROOT / "local_data/e0_qualification_bootstrap/Q1", REPO_ROOT)
    device, model, optimizer = ctx["device"], ctx["model"], ctx["optimizer"]
    train_rows = ctx["train_rows"]
    stream = TrainStream(len(train_rows), DIAG_SEED)
    prev_flat = torch.cat([p.detach().flatten() for p in model.parameters()])

    records = []
    evals = []
    started = time.time()
    for update in range(1, updates + 1):
        lr = learning_rate(update)
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch_indices = stream.take(128)
        optimizer.zero_grad(set_to_none=True)
        micro_losses, micro_argmaxes, micro_logstats = [], [], []
        for micro in range(GRAD_ACCUM):
            rows = [train_rows[i] for i in batch_indices[micro * MICROBATCH : (micro + 1) * MICROBATCH]]
            ids, decide, labels = collate(rows, device)
            logits = model(ids, decide)
            loss = F.cross_entropy(logits, labels, label_smoothing=0.0)
            micro_losses.append(loss.detach())
            with torch.no_grad():
                micro_argmaxes.append(logits.detach().argmax(dim=-1))
                micro_logstats.append((
                    float(logits.detach().float().mean()),
                    float(logits.detach().float().std()),
                    float(logits.detach().float().abs().max()),
                ))
            (loss / GRAD_ACCUM).backward()

        with torch.no_grad():
            embed_grad_norm = float(model.embed_tokens.weight.grad.norm())
            classifier_grad_norm = float(model.classifier.weight.grad.norm())
            all_losses = torch.stack(micro_losses)
            batch_ce = float(all_losses.mean())
            argmax = torch.cat(micro_argmaxes).cpu()
            hist = torch.bincount(argmax, minlength=3).tolist()
            # entropy of the aggregate prediction histogram (proxy recorded per D0 spec)
            p = np.asarray(hist, dtype=np.float64) / max(1, sum(hist))
            nz = p[p > 0]
            hist_entropy = float(-(nz * np.log(nz)).sum())
            logits_mean = float(np.mean([s[0] for s in micro_logstats]))
            logits_std = float(np.mean([s[1] for s in micro_logstats]))
            logits_maxabs = float(np.max([s[2] for s in micro_logstats]))

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        with torch.no_grad():
            flat = torch.cat([p.detach().flatten() for p in model.parameters()])
            param_delta_norm = float((flat - prev_flat).norm())
            prev_flat = flat

        if update <= 200 or update % 10 == 0:
            records.append({
                "update": update, "lr": lr,
                "batch_ce_mean": batch_ce,
                "pred_hist": hist, "pred_hist_entropy": hist_entropy,
                "grad_norm_pre_clip": float(grad_norm),
                "classifier_w_grad_norm": classifier_grad_norm,
                "embed_grad_norm": embed_grad_norm,
                "param_delta_norm": param_delta_norm,
                "logits_mean": logits_mean, "logits_std": logits_std,
                "logits_max_abs": logits_maxabs,
            })
        if update % eval_every == 0:
            sma = evaluate(model, ctx["eval_id_rows"], device, update)["cmdr_sma"]
            evals.append({"update": update, "eval_ID_cmdr_sma": sma})
            print(f"[D0] update {update}/{updates} batch_ce={batch_ce:.4f} eval_SMA={sma:.6f}", flush=True)

    return diag_header({
        "schema_id": "E0-Q2-DIAG-D0-TRAINING-DYNAMICS-v0",
        "candidate": "C0", "seed": DIAG_SEED, "updates": updates,
        "wd_scope": "exclude_norm_bias",
        "instrumentation_cadence": "every update <=200, then every 10",
        "records": records,
        "eval_ID_checkpoints": evals,
        "wall_seconds": round(time.time() - started, 1),
    })


# --------------------------------------------------------------------- D1 ----

def d1_family_selection(train_rows_raw: list[dict], families_per_depth: int = 6) -> list[dict]:
    """Deterministic tiny set: complete counterfactual families, balanced across
    label (by family construction) and depth. Rank by SHA-256, take first N."""
    from q1_cmdr_bootstrap import LEX

    by_family: dict[str, list[dict]] = {}
    for row in train_rows_raw:
        by_family.setdefault(row["family_id"], []).append(row)
    selected_ids = []
    for depth in range(1, 5):
        fams = []
        for fid, variants in by_family.items():
            if len(variants) != 3:
                continue
            if any(v["reasoning_depth_stratum"] != depth for v in variants):
                continue
            if sorted(v["gold_label"] for v in variants) != sorted(LABELS):
                continue
            rank = hashlib.sha256(f"E0-Q2-DIAG|D1|{fid}".encode()).hexdigest()
            fams.append((rank, fid))
        fams.sort()
        selected_ids.extend(fid for _, fid in fams[:families_per_depth])

    rows = []
    for fid in selected_ids:
        for variant in sorted(by_family[fid], key=lambda v: LABELS.index(v["gold_label"])):
            ids = LEX.encode(variant["rendered"], append_decide=True)
            rows.append({
                "sample_id": variant["sample_id"], "family_id": fid,
                "token_ids": ids, "label_id": LABEL_TO_ID[variant["gold_label"]],
                "depth": variant["reasoning_depth_stratum"], "surface": variant["surface"],
            })
    return rows


def run_d1(updates: int = 2000, families_per_depth: int = 6, eval_every: int = 50) -> dict:
    """Tiny-set memorization through the exact production collate/model/loss/
    optimizer path. Discriminator: can training accuracy approach 1.0?"""
    ctx = scientific_setup("C0", "exclude_norm_bias", DIAG_SEED,
                           REPO_ROOT / "local_data/e0_qualification_bootstrap/Q1", REPO_ROOT)
    device, model, optimizer = ctx["device"], ctx["model"], ctx["optimizer"]
    raw_path = REPO_ROOT / "local_data/e0_qualification_bootstrap/Q1/train_ID.jsonl"
    raw_train = [json.loads(line) for line in raw_path.open(encoding="utf-8")]
    tiny = d1_family_selection(raw_train, families_per_depth)
    n = len(tiny)
    label_counts = {name: sum(1 for r in tiny if r["label_id"] == i) for i, name in enumerate(LABELS)}
    depth_counts = {f"d{d}": sum(1 for r in tiny if r["depth"] == d) for d in range(1, 5)}

    stream = TrainStream(n, DIAG_SEED)  # same stream semantics over the tiny corpus

    def tiny_metrics() -> dict:
        preds, losses = [], []
        model.eval()
        with torch.no_grad():
            for start in range(0, n, MICROBATCH):
                batch = tiny[start : start + MICROBATCH]
                ids, decide, labels = collate(batch, device)
                logits = model(ids, decide)
                losses.append(F.cross_entropy(logits, labels, reduction="sum"))
                preds.extend(logits.argmax(dim=-1).cpu().tolist())
        model.train()
        y_true = np.asarray([r["label_id"] for r in tiny])
        y_pred = np.asarray(preds)
        per_label = {name: float(np.mean(y_pred[y_true == i] == i)) for i, name in enumerate(LABELS)}
        return {
            "train_accuracy": float(np.mean(y_pred == y_true)),
            "train_ce_mean": float(torch.stack(losses).sum() / n),
            "per_label_accuracy": per_label,
            "pred_hist": [int(np.sum(y_pred == i)) for i in range(3)],
        }

    history = [dict(update=0, **tiny_metrics())]
    started = time.time()
    for update in range(1, updates + 1):
        lr = learning_rate(update)
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch_indices = stream.take(128)
        optimizer.zero_grad(set_to_none=True)
        for micro in range(GRAD_ACCUM):
            rows = [tiny[i] for i in batch_indices[micro * MICROBATCH : (micro + 1) * MICROBATCH]]
            ids, decide, labels = collate(rows, device)
            logits = model(ids, decide)
            loss = F.cross_entropy(logits, labels, label_smoothing=0.0)
            (loss / GRAD_ACCUM).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        if update % eval_every == 0 or update == updates:
            m = tiny_metrics()
            history.append(dict(update=update, **m))
            print(f"[D1] update {update}/{updates} acc={m['train_accuracy']:.4f} ce={m['train_ce_mean']:.4f}", flush=True)

    final = history[-1]
    return diag_header({
        "schema_id": "E0-Q2-DIAG-D1-MEMORIZATION-v0",
        "candidate": "C0", "seed": DIAG_SEED, "updates": updates,
        "wd_scope": "exclude_norm_bias",
        "tiny_set": {
            "families_per_depth": families_per_depth,
            "families": families_per_depth * 4,
            "examples": n,
            "label_counts": label_counts,
            "depth_counts": depth_counts,
            "selection_rule": "complete families ranked by sha256('E0-Q2-DIAG|D1|<family_id>'), first N per depth",
        },
        "history": history,
        "final_train_accuracy": final["train_accuracy"],
        "final_train_ce_mean": final["train_ce_mean"],
        "final_per_label_accuracy": final["per_label_accuracy"],
        "memorization_verdict": "MEMORIZED" if final["train_accuracy"] >= 0.98 else "NOT_MEMORIZED",
        "wall_seconds": round(time.time() - started, 1),
    })


# --------------------------------------------------------------------- D2 ----

def run_d2() -> dict:
    """Independent label / <DECIDE> / loss-transport audit on the production
    path. Verifies the training signal is what the contract says it is."""
    import q1_cmdr_bootstrap as q1
    from collections import Counter

    from m0_model import PAD_TOKEN_ID, build_model, build_rope_tables

    checks: dict[str, dict] = {}

    # 1. gold_label -> label_id mapping matches the Q1 generator's label order
    checks["label_mapping"] = {
        "q2_labels": LABELS,
        "q1_labels": q1.LABELS,
        "identical_order": LABELS == q1.LABELS,
        "ok": LABELS == q1.LABELS,
    }

    raw_path = REPO_ROOT / "local_data/e0_qualification_bootstrap/Q1/train_ID.jsonl"
    raw_rows = [json.loads(line) for line in raw_path.open(encoding="utf-8")]
    sampled = raw_rows[::2400][:10]

    # 2. <DECIDE> is the terminal token, its ID is the frozen lexicon entry, and
    #    collate's decide_index is exactly the terminal position
    decide_id = q1.LEX.vocab["<DECIDE>"]
    mismatches = []
    for row in sampled:
        ids = q1.LEX.encode(row["rendered"], append_decide=True)
        if ids[-1] != decide_id:
            mismatches.append([row["sample_id"], "terminal_token", ids[-1]])
        if len(ids) != row["student_token_count_with_DECIDE"]:
            mismatches.append([row["sample_id"], "token_count"])
        token_ids, decide, labels = collate(
            [{"token_ids": ids, "label_id": LABEL_TO_ID[row["gold_label"]]}], torch.device("cpu")
        )
        if int(decide[0]) != len(ids) - 1 or int(labels[0]) != LABEL_TO_ID[row["gold_label"]]:
            mismatches.append([row["sample_id"], "collate_decide_or_label"])
    checks["decide_token"] = {
        "decide_token_id": decide_id,
        "pad_token_id": PAD_TOKEN_ID,
        "sampled_examples": len(sampled),
        "mismatches": mismatches,
        "ok": not mismatches,
    }

    # 3. One production-path microbatch with a capturing loss: verify the exact
    #    labels tensor reaching cross_entropy against an independent mapping,
    #    and keep the training logits for the downstream identity checks.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("C0", DIAG_SEED, device)
    model.train()
    tiny = d1_family_selection(raw_rows, families_per_depth=6)
    stream = TrainStream(len(tiny), DIAG_SEED)
    batch_indices = stream.take(128)
    micro_rows = [tiny[i] for i in batch_indices[0:MICROBATCH]]
    ids, decide, labels = collate(micro_rows, device)
    logits = model(ids, decide)
    captured = {"logits": logits.detach().clone(), "labels": labels.detach().clone()}
    loss = F.cross_entropy(logits, labels, label_smoothing=0.0)
    loss.backward()

    expected_labels = torch.tensor([r["label_id"] for r in micro_rows], device=device)
    gold_by_sample = {r["sample_id"]: r["gold_label"] for r in raw_rows}
    expected_from_gold = torch.tensor(
        [LABEL_TO_ID[gold_by_sample[r["sample_id"]]] for r in micro_rows], device=device
    )
    checks["loss_transport"] = {
        "captured_labels_match_collate": bool(torch.equal(captured["labels"], labels)),
        "captured_labels_match_independent_mapping": bool(torch.equal(captured["labels"], expected_labels)),
        "independent_mapping_matches_gold_strings": bool(torch.equal(expected_labels, expected_from_gold)),
        "loss_finite": bool(torch.isfinite(loss)),
        "logits_shape": list(captured["logits"].shape),
        "ok": bool(torch.equal(captured["labels"], expected_labels))
        and bool(torch.equal(expected_labels, expected_from_gold))
        and bool(torch.isfinite(loss)),
    }

    # 4. Predictions come from the same logits used in the loss (bit-exact
    #    replay of the forward; no optimizer step has been taken).
    with torch.no_grad():
        replay_logits = model(ids, decide)
    checks["prediction_source"] = {
        "replay_bitexact": bool(torch.equal(replay_logits, captured["logits"])),
        "argmax_consistent": bool(torch.equal(replay_logits.argmax(-1), captured["logits"].argmax(-1))),
        "ok": bool(torch.equal(replay_logits, captured["logits"])),
    }

    # 5. Classifier reads the post-final-RMSNorm state at exactly decide_index.
    with torch.no_grad():
        x = model.embed_tokens(ids)
        seq_len = x.shape[1]
        cos_tab, sin_tab = build_rope_tables(seq_len, device)
        causal_mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        for layer in model.layers:
            x = layer(x, cos_tab, sin_tab, causal_mask)
        x = model.norm(x)
        idx = torch.arange(ids.shape[0], device=device)
        manual_logits = model.classifier(x[idx, decide])
    checks["decide_state_transport"] = {
        "manual_equals_forward_logits": bool(torch.equal(manual_logits, captured["logits"])),
        "max_abs_diff": float((manual_logits - captured["logits"]).abs().max()),
        "ok": bool(torch.equal(manual_logits, captured["logits"])),
    }

    # 6. Counterfactual variants are distinct inputs in sequence order while
    #    unigram/bigram multisets remain invariant (input-side signal exists).
    fam: dict[str, list[dict]] = {}
    for row in raw_rows:
        fam.setdefault(row["family_id"], []).append(row)
    distinct_ok, unigram_ok, bigram_ok, diff_positions = True, True, True, []
    audited = 0
    for fid, variants in fam.items():
        if len(variants) != 3 or audited >= 24:
            continue
        seqs = [q1.LEX.encode(v["rendered"], append_decide=True) for v in variants]
        if len({len(s) for s in seqs}) != 1:
            distinct_ok = False
            continue
        if seqs[0] == seqs[1] or seqs[1] == seqs[2] or seqs[0] == seqs[2]:
            distinct_ok = False
            continue
        diff_positions.append(next(i for i in range(len(seqs[0])) if len({s[i] for s in seqs}) > 1))
        uni = [Counter(s) for s in seqs]
        bi = [Counter(zip(s, s[1:])) for s in seqs]
        if not (uni[0] == uni[1] == uni[2]):
            unigram_ok = False
        if not (bi[0] == bi[1] == bi[2]):
            bigram_ok = False
        audited += 1
    checks["counterfactual_input_distinctness"] = {
        "families_audited": audited,
        "pairwise_distinct_sequences": distinct_ok,
        "unigram_multiset_invariant": unigram_ok,
        "bigram_multiset_invariant": bigram_ok,
        "first_difference_position": {
            "min": min(diff_positions), "median": float(np.median(diff_positions)), "max": max(diff_positions),
        },
        "ok": distinct_ok and unigram_ok and bigram_ok,
    }

    overall = all(v.get("ok", False) for v in checks.values())
    return diag_header({
        "schema_id": "E0-Q2-DIAG-D2-TRANSPORT-AUDIT-v0",
        "candidate": "C0", "seed": DIAG_SEED,
        "checks": checks,
        "status": "PASS" if overall else "FAIL",
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["d0", "d1", "d2", "all"])
    parser.add_argument("--d0-updates", type=int, default=2000)
    parser.add_argument("--d1-updates", type=int, default=2000)
    args = parser.parse_args()

    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if args.mode in ("d0", "all"):
        write_diag("d0_training_dynamics.json", run_d0(updates=args.d0_updates))
    if args.mode in ("d1", "all"):
        write_diag("d1_memorization.json", run_d1(updates=args.d1_updates))
    if args.mode in ("d2", "all"):
        write_diag("d2_label_transport_audit.json", run_d2())


if __name__ == "__main__":
    main()
