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

def _ranked_complete_families_by_depth(train_rows_raw: list[dict]) -> dict[int, list[str]]:
    """Depth -> ordered family IDs of complete counterfactual families (all 3
    labels, uniform depth), ranked by sha256('E0-Q2-DIAG|D1|<family_id>')."""
    by_family: dict[str, list[dict]] = {}
    for row in train_rows_raw:
        by_family.setdefault(row["family_id"], []).append(row)
    ranked: dict[int, list[tuple[str, str]]] = {}
    for fid, variants in by_family.items():
        if len(variants) != 3:
            continue
        if sorted(v["gold_label"] for v in variants) != sorted(LABELS):
            continue
        depths = {v["reasoning_depth_stratum"] for v in variants}
        if len(depths) != 1:
            continue
        depth = depths.pop()
        ranked.setdefault(depth, []).append(
            (hashlib.sha256(f"E0-Q2-DIAG|D1|{fid}".encode()).hexdigest(), fid)
        )
    return {d: [fid for _, fid in sorted(pairs)] for d, pairs in sorted(ranked.items())}


def _build_family_rows(train_rows_raw: list[dict], family_ids: list[str]) -> list[dict]:
    from q1_cmdr_bootstrap import LEX

    by_family: dict[str, list[dict]] = {}
    for row in train_rows_raw:
        by_family.setdefault(row["family_id"], []).append(row)
    rows = []
    for fid in family_ids:
        for variant in sorted(by_family[fid], key=lambda v: LABELS.index(v["gold_label"])):
            ids = LEX.encode(variant["rendered"], append_decide=True)
            rows.append({
                "sample_id": variant["sample_id"], "family_id": fid,
                "token_ids": ids, "label_id": LABEL_TO_ID[variant["gold_label"]],
                "depth": variant["reasoning_depth_stratum"], "surface": variant["surface"],
            })
    return rows


def d1_family_selection(train_rows_raw: list[dict], families_per_depth: int = 6) -> list[dict]:
    """D1 tiny set: first N ranked complete families per depth (unchanged rule)."""
    ranked = _ranked_complete_families_by_depth(train_rows_raw)
    selected = [fid for depth in sorted(ranked) for fid in ranked[depth][:families_per_depth]]
    return _build_family_rows(train_rows_raw, selected)


def d3_family_sets(train_rows_raw: list[dict], per_depth: int = 6):
    """D3 sets: TRAIN = first per_depth ranked families per depth (identical to
    D1); HOLDOUT = the NEXT per_depth ranked families per depth (disjoint)."""
    ranked = _ranked_complete_families_by_depth(train_rows_raw)
    train_ids, holdout_ids = [], []
    for depth in sorted(ranked):
        if len(ranked[depth]) < 2 * per_depth:
            raise AssertionError(f"depth {depth} has {len(ranked[depth])} complete families < {2*per_depth}")
        train_ids.extend(ranked[depth][:per_depth])
        holdout_ids.extend(ranked[depth][per_depth : 2 * per_depth])
    assert not (set(train_ids) & set(holdout_ids))
    train_rows = _build_family_rows(train_rows_raw, train_ids)
    holdout_rows = _build_family_rows(train_rows_raw, holdout_ids)
    meta = {
        "train_family_ids": train_ids,
        "holdout_family_ids": holdout_ids,
        "train_family_list_sha256": hashlib.sha256(json.dumps(train_ids).encode()).hexdigest(),
        "holdout_family_list_sha256": hashlib.sha256(json.dumps(holdout_ids).encode()).hexdigest(),
    }
    return train_rows, holdout_rows, meta


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


# --------------------------------------------------------------------- D3 ----

def _decide_states(model, rows: list[dict], device) -> torch.Tensor:
    """Post-final-RMSNorm 512-d state at <DECIDE> for each row (manual replay
    through the exact production modules; eval mode, no grad)."""
    from m0_model import build_rope_tables

    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(rows), MICROBATCH):
            batch = rows[start : start + MICROBATCH]
            ids, decide, _ = collate(batch, device)
            x = model.embed_tokens(ids)
            seq_len = x.shape[1]
            cos_t, sin_t = build_rope_tables(seq_len, device)
            mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
            )
            for layer in model.layers:
                x = layer(x, cos_t, sin_t, mask)
            x = model.norm(x)
            idx = torch.arange(ids.shape[0], device=device)
            out.append(x[idx, decide].float().cpu())
    model.train()
    return torch.cat(out)


def _classifier_metrics(model, rows: list[dict], device) -> dict:
    model.eval()
    preds, losses = [], []
    with torch.no_grad():
        for start in range(0, len(rows), MICROBATCH):
            batch = rows[start : start + MICROBATCH]
            ids, decide, labels = collate(batch, device)
            logits = model(ids, decide)
            losses.append(F.cross_entropy(logits, labels, reduction="sum"))
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
    model.train()
    y = np.asarray([r["label_id"] for r in rows])
    p = np.asarray(preds)
    per_label = {name: float(np.mean(p[y == i] == i)) for i, name in enumerate(LABELS)}
    return {
        "accuracy": float(np.mean(p == y)),
        "ce_mean": float(torch.stack(losses).sum() / len(rows)),
        "pred_hist": [int(np.sum(p == i)) for i in range(3)],
        "per_label_recall": per_label,
        "min_label_recall": min(per_label.values()),
    }


_LABEL_PAIRS = [(0, 1, "E_C"), (0, 2, "E_U"), (1, 2, "C_U")]
_DISPLACEMENTS = [(0, 1, "E_minus_C"), (0, 2, "E_minus_U"), (1, 2, "C_minus_U")]


def _geometry(states: torch.Tensor, n_families: int):
    """states ordered family-major with label order E/C/U (how _build_family_rows
    emits). Returns within-family distances/cosines, family-centered residuals,
    and cross-family displacement alignment."""
    h = states.view(n_families, 3, -1)
    within = {}
    for a, b, name in _LABEL_PAIRS:
        d = (h[:, a] - h[:, b]).norm(dim=-1)
        cos = F.cosine_similarity(h[:, a], h[:, b], dim=-1)
        within[name] = {
            "l2_mean": float(d.mean()), "l2_median": float(d.median()),
            "cos_mean": float(cos.mean()), "cos_median": float(cos.median()),
        }
    residuals = h - h.mean(dim=1, keepdim=True)
    displacement = {}
    for a, b, name in _DISPLACEMENTS:
        v = h[:, a] - h[:, b]
        vn = F.normalize(v, dim=-1)
        sim = vn @ vn.T
        iu = torch.triu_indices(n_families, n_families, offset=1)
        sims = sim[iu[0], iu[1]]
        displacement[name] = {
            "mean_pairwise_cos": float(sims.mean()),
            "median_pairwise_cos": float(sims.median()),
            "p10": float(sims.quantile(0.1)), "p90": float(sims.quantile(0.9)),
        }
    displacement["vectors"] = {
        name: F.normalize(h[:, a] - h[:, b], dim=-1) for a, b, name in _DISPLACEMENTS
    }
    return within, residuals, displacement


def _cross_set_alignment(disp_train: dict, disp_holdout: dict) -> dict:
    out = {}
    for _, _, name in _DISPLACEMENTS:
        sim = disp_train["vectors"][name] @ disp_holdout["vectors"][name].T
        out[name] = {"mean_cos": float(sim.mean()), "min": float(sim.min()), "max": float(sim.max())}
    return out


def _centroid_classification(train_residuals: torch.Tensor, holdout_residuals: torch.Tensor) -> dict:
    centroids = train_residuals.mean(dim=0)  # (3, D)
    cn = F.normalize(centroids, dim=-1)

    def classify(res: torch.Tensor) -> dict:
        sims = F.normalize(res.reshape(-1, res.shape[-1]), dim=-1) @ cn.T  # (N,3)
        pred = sims.argmax(dim=-1).numpy()
        y = np.tile(np.arange(3), res.shape[0])
        per_label = {name: float(np.mean(pred[y == i] == i)) for i, name in enumerate(LABELS)}
        return {
            "accuracy": float(np.mean(pred == y)),
            "per_label_recall": per_label,
            "min_label_recall": min(per_label.values()),
        }

    return {
        "train": classify(train_residuals),
        "holdout": classify(holdout_residuals),
        "centroid_pairwise_cos": {
            name: float(F.cosine_similarity(centroids[a], centroids[b], dim=-1))
            for a, b, name in _LABEL_PAIRS
        },
    }


def run_d3(updates: int = 2000, per_depth: int = 6) -> dict:
    """D3 representation discriminability: does the label-dependent <DECIDE>-state
    displacement align ACROSS independent counterfactual families (train and a
    disjoint holdout), at T0 / T100 / T2000?"""
    ctx = scientific_setup("C0", "exclude_norm_bias", DIAG_SEED,
                           REPO_ROOT / "local_data/e0_qualification_bootstrap/Q1", REPO_ROOT)
    device, model, optimizer = ctx["device"], ctx["model"], ctx["optimizer"]
    tree_clean_at_start = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip() == ""
    raw_path = REPO_ROOT / "local_data/e0_qualification_bootstrap/Q1/train_ID.jsonl"
    raw_rows = [json.loads(line) for line in raw_path.open(encoding="utf-8")]
    train_rows, holdout_rows, fam_meta = d3_family_sets(raw_rows, per_depth)
    n_fam = per_depth * 4

    probes: dict[str, dict] = {}

    def capture(tag: str) -> None:
        st_tr = _decide_states(model, train_rows, device)
        st_ho = _decide_states(model, holdout_rows, device)
        within_tr, resid_tr, disp_tr = _geometry(st_tr, n_fam)
        within_ho, resid_ho, disp_ho = _geometry(st_ho, n_fam)
        vectors_tr, vectors_ho = disp_tr.pop("vectors"), disp_ho.pop("vectors")
        probes[tag] = {
            "within_family": {"train": within_tr, "holdout": within_ho},
            "cross_family_displacement": {"train": disp_tr, "holdout": disp_ho},
            "cross_set_displacement": _cross_set_alignment(
                {"vectors": vectors_tr}, {"vectors": vectors_ho}
            ),
            "centroid_classification": _centroid_classification(resid_tr, resid_ho),
            "classifier": {
                "train": _classifier_metrics(model, train_rows, device),
                "holdout": _classifier_metrics(model, holdout_rows, device),
            },
        }
        print(f"[D3] captured probe {tag}", flush=True)

    capture("T0")
    memorization_history = [dict(update=0, **_classifier_metrics(model, train_rows, device))]
    started = time.time()
    stream = TrainStream(len(train_rows), DIAG_SEED)
    for update in range(1, updates + 1):
        lr = learning_rate(update)
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch_indices = stream.take(128)
        optimizer.zero_grad(set_to_none=True)
        for micro in range(GRAD_ACCUM):
            rows = [train_rows[i] for i in batch_indices[micro * MICROBATCH : (micro + 1) * MICROBATCH]]
            ids, decide, labels = collate(rows, device)
            logits = model(ids, decide)
            loss = F.cross_entropy(logits, labels, label_smoothing=0.0)
            (loss / GRAD_ACCUM).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        if update % 50 == 0:
            m = _classifier_metrics(model, train_rows, device)
            memorization_history.append(dict(update=update, **{k: m[k] for k in ("accuracy", "ce_mean", "pred_hist")}))
        if update == 100:
            capture("T100")
    capture("T2000")

    # Pattern classification (criteria stated explicitly; authority inspects raw
    # geometry regardless).
    def aligned(set_name: str, probe: str) -> bool:
        v = probes[probe]["cross_family_displacement"][set_name]["E_minus_C"]["mean_pairwise_cos"]
        base = abs(probes["T0"]["cross_family_displacement"][set_name]["E_minus_C"]["mean_pairwise_cos"])
        return v > max(0.20, 4.0 * base)

    final_clf_train = probes["T2000"]["classifier"]["train"]["accuracy"]
    train_discriminative = (
        aligned("train", "T2000")
        or probes["T2000"]["centroid_classification"]["train"]["accuracy"] > 0.9
    )
    if final_clf_train >= 0.99 and not train_discriminative:
        pattern = "CONTRADICTION_TRAIN_CLASSIFIER_VS_GEOMETRY"
    elif aligned("train", "T2000") and aligned("holdout", "T2000"):
        pattern = "TRAIN_ALIGNED_HOLDOUT_ALIGNED"
    elif aligned("train", "T2000") and not aligned("holdout", "T2000"):
        pattern = "TRAIN_ALIGNED_HOLDOUT_CHANCE"
    else:
        pattern = "TRAIN_NOT_DISCRIMINATIVE_GEOMETRICALLY"

    final_mem = memorization_history[-1]
    return diag_header({
        "schema_id": "E0-Q2-DIAG-D3-REPRESENTATION-v0",
        "candidate": "C0", "seed": DIAG_SEED, "updates": updates,
        "wd_scope": "exclude_norm_bias",
        "working_tree_clean_at_start": tree_clean_at_start,
        "code_git_commit": subprocess_git_head(REPO_ROOT),
        "verified_q1_input_digests": ctx["verified_digests"],
        "family_sets": {
            "families_per_set": n_fam,
            "train_examples": len(train_rows),
            "holdout_examples": len(holdout_rows),
            "selection_rule": "train = first 6 sha256-ranked complete families per depth (identical to D1); holdout = next 6 per depth (disjoint)",
            "train_family_list_sha256": fam_meta["train_family_list_sha256"],
            "holdout_family_list_sha256": fam_meta["holdout_family_list_sha256"],
        },
        "d1_replication": {
            "history": memorization_history,
            "final_train_accuracy": final_mem["accuracy"],
            "final_train_ce_mean": final_mem["ce_mean"],
            "note": "identical train-set selection, seed, stream, and production LR values as D1",
        },
        "probes": probes,
        "pattern_criteria": {
            "aligned": "mean_pairwise_cos(E_minus_C) > max(0.20, 4*|T0 value|)",
            "train_discriminative": "train aligned OR train residual-centroid accuracy > 0.9",
        },
        "observed_pattern": pattern,
        "wall_seconds": round(time.time() - started, 1),
    })


# ------------------------------------------------------------------- D4-A ----

FC_STREAM_NAMESPACE = "E0-Q2-DIAG|D4A|fc"
D4A_PROBE_NAMESPACE = "E0-Q2-DIAG|D4A|probe"


class FamilyCoherentStream:
    """D4-A FC arm stream: families shuffled as units, E/C/U variants emitted
    contiguously (fixed label order). Same take() semantics as the production
    TrainStream — cross-boundary batches, no example dropped. An effective
    batch of 128 contains 42 complete families (126 examples) plus one
    2-example boundary fragment of the next family. Yields indices into the
    production train_rows list (resolved by sample_id)."""

    def __init__(self, family_rows: list[dict], sid_to_idx: dict[str, int], seed: int) -> None:
        import random as _random

        self._random = _random
        self.seed = seed
        self.sid_to_idx = sid_to_idx
        by_family: dict[str, list[dict]] = {}
        for r in family_rows:
            by_family.setdefault(r["family_id"], []).append(r)
        self.blocks = [
            sorted(by_family[fid], key=lambda r: r["label_id"]) for fid in sorted(by_family)
        ]
        self._epoch = 0
        self._buffer: list[int] = []

    def _epoch_flat(self, epoch: int) -> list[int]:
        digest = hashlib.sha256(f"{FC_STREAM_NAMESPACE}|{self.seed}|{epoch}".encode()).digest()
        rng = self._random.Random(int.from_bytes(digest[:8], "big"))
        order = list(range(len(self.blocks)))
        rng.shuffle(order)
        return [self.sid_to_idx[row["sample_id"]] for i in order for row in self.blocks[i]]

    def take(self, count: int) -> list[int]:
        out: list[int] = []
        while len(out) < count:
            if not self._buffer:
                self._buffer = self._epoch_flat(self._epoch)
                self._epoch += 1
            need = count - len(out)
            out.extend(self._buffer[:need])
            self._buffer = self._buffer[need:]
        return out


def _eval_surface_metrics(model, rows: list[dict], device) -> dict:
    """Full-surface metrics through the production eval batching."""
    from q2_m0_qualify import EVAL_BATCH as _EB

    model.eval()
    preds, losses = [], []
    with torch.no_grad():
        for start in range(0, len(rows), _EB):
            batch = rows[start : start + _EB]
            ids, decide, labels = collate(batch, device)
            logits = model(ids, decide)
            losses.append(F.cross_entropy(logits, labels, reduction="sum"))
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
    model.train()
    y = np.asarray([r["label_id"] for r in rows])
    p = np.asarray(preds)
    depths = np.asarray([r["depth"] for r in rows])
    per_label = {name: float(np.mean(p[y == i] == i)) for i, name in enumerate(LABELS)}
    from q2_m0_qualify import macro_cell_accuracy, macro_cell_accuracy_depths

    return {
        "cmdr_sma": macro_cell_accuracy(y, p, depths),
        "sma_depth_2_4": macro_cell_accuracy_depths(y, p, depths, (2, 3, 4)),
        "accuracy": float(np.mean(p == y)),
        "ce_mean": float(torch.stack(losses).sum() / len(rows)),
        "pred_hist": [int(np.sum(p == i)) for i in range(3)],
        "per_label_recall": per_label,
        "min_label_recall": min(per_label.values()),
    }


def _d4a_probe_rows(raw_eval_rows: list[dict], per_depth: int = 6) -> tuple[list[dict], str]:
    by_family: dict[str, list[dict]] = {}
    for row in raw_eval_rows:
        by_family.setdefault(row["family_id"], []).append(row)
    ranked: dict[int, list[tuple[str, str]]] = {}
    for fid, variants in by_family.items():
        if len(variants) != 3 or len({v["gold_label"] for v in variants}) != 3:
            continue
        depths = {v["reasoning_depth_stratum"] for v in variants}
        if len(depths) != 1:
            continue
        ranked.setdefault(depths.pop(), []).append(
            (hashlib.sha256(f"{D4A_PROBE_NAMESPACE}|{fid}".encode()).hexdigest(), fid)
        )
    ids = [fid for depth in sorted(ranked) for _, fid in sorted(ranked[depth])[:per_depth]]
    digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    return _build_family_rows(raw_eval_rows, ids), digest


def run_d4a(updates: int = 2000) -> dict:
    """D4-A family-coherent batching falsifier: paired R (production example-
    level stream) vs FC (family-unit stream) arms on the full 24k corpus, same
    data/model/seed/loss/optimizer/LR/batch/compute; 2,000 updates each."""
    raw_train_path = REPO_ROOT / "local_data/e0_qualification_bootstrap/Q1/train_ID.jsonl"
    raw_train = [json.loads(line) for line in raw_train_path.open(encoding="utf-8")]
    raw_eval_path = REPO_ROOT / "local_data/e0_qualification_bootstrap/Q1/eval_ID.jsonl"
    raw_eval = [json.loads(line) for line in raw_eval_path.open(encoding="utf-8")]
    probe_rows, probe_digest = _d4a_probe_rows(raw_eval)

    production_seed1 = json.loads(
        (REPO_ROOT / "local_data/e0_qualification_bootstrap/Q2/C0/seed_1647674144/result.json")
        .read_text(encoding="utf-8")
    )
    prod_eval_history = {h["update"]: h["eval_ID_cmdr_sma"] for h in production_seed1["eval_history"]}

    stream_configs = {
        "R": {
            "stream": "production example-level TrainStream",
            "namespace": "ExpertForge-E0-Q2|M0|perm",
        },
        "FC": {
            "stream": "family-unit permutation; E/C/U variants contiguous (fixed label order)",
            "namespace": FC_STREAM_NAMESPACE,
        },
    }
    stream_config_digest = hashlib.sha256(json.dumps(stream_configs, sort_keys=True).encode()).hexdigest()

    arms: dict[str, dict] = {}
    for arm in ("R", "FC"):
        ctx = scientific_setup("C0", "exclude_norm_bias", DIAG_SEED,
                               REPO_ROOT / "local_data/e0_qualification_bootstrap/Q1", REPO_ROOT)
        device, model, optimizer = ctx["device"], ctx["model"], ctx["optimizer"]
        train_rows = ctx["train_rows"]
        tree_clean = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip() == ""

        fc_rows = None
        if arm == "FC":
            fc_rows = _build_family_rows(raw_train, sorted({r["family_id"] for r in raw_train}))
            sid_to_idx = {r["sample_id"]: i for i, r in enumerate(train_rows)}
            stream = FamilyCoherentStream(fc_rows, sid_to_idx, DIAG_SEED)
            take = stream.take
        else:
            stream = TrainStream(len(train_rows), DIAG_SEED)
            take = stream.take

        prev_flat = torch.cat([p.detach().flatten() for p in model.parameters()])
        telemetry: list[dict] = []
        probes: dict[str, dict] = {}

        def capture(tag: str, update: int) -> None:
            probes[tag] = {
                "update": update,
                "eval_ID": _eval_surface_metrics(model, ctx["eval_id_rows"], device),
                "eval_STRUCT": _eval_surface_metrics(model, ctx["eval_struct_rows"], device),
                "unseen_family_geometry": _d3_probe_geometry(model, probe_rows, device),
            }
            print(f"[D4A {arm}] probe {tag}: eval_ID SMA={probes[tag]['eval_ID']['cmdr_sma']:.6f}", flush=True)

        torch.cuda.reset_peak_memory_stats(device)
        capture("T0", 0)
        started = time.time()
        for update in range(1, updates + 1):
            lr = learning_rate(update)
            for group in optimizer.param_groups:
                group["lr"] = lr
            batch_indices = take(128)
            optimizer.zero_grad(set_to_none=True)
            micro_losses, micro_argmaxes = [], []
            for micro in range(GRAD_ACCUM):
                rows = [train_rows[i] for i in batch_indices[micro * MICROBATCH : (micro + 1) * MICROBATCH]]
                ids, decide, labels = collate(rows, device)
                logits = model(ids, decide)
                loss = F.cross_entropy(logits, labels, label_smoothing=0.0)
                micro_losses.append(loss.detach())
                with torch.no_grad():
                    micro_argmaxes.append(logits.detach().argmax(dim=-1))
                (loss / GRAD_ACCUM).backward()
            with torch.no_grad():
                embed_grad_norm = float(model.embed_tokens.weight.grad.norm())
                classifier_grad_norm = float(model.classifier.weight.grad.norm())
                batch_ce = float(torch.stack(micro_losses).mean())
                argmax = torch.cat(micro_argmaxes).cpu()
                hist = torch.bincount(argmax, minlength=3).tolist()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            with torch.no_grad():
                flat = torch.cat([p.detach().flatten() for p in model.parameters()])
                param_delta_norm = float((flat - prev_flat).norm())
                prev_flat = flat
            if update <= 200 or update % 10 == 0:
                telemetry.append({
                    "update": update, "lr": lr, "batch_ce_mean": batch_ce,
                    "pred_hist": hist, "grad_norm_pre_clip": float(grad_norm),
                    "classifier_w_grad_norm": classifier_grad_norm,
                    "embed_grad_norm": embed_grad_norm,
                    "param_delta_norm": param_delta_norm,
                })
            if update % 400 == 0:
                capture(f"T{update}", update)
        wall = time.time() - started

        r_vs_production = None
        if arm == "R":
            r_vs_production = {
                str(u): probes[f"T{u}"]["eval_ID"]["cmdr_sma"] == prod_eval_history[u]
                for u in (400, 800, 1200, 1600, 2000) if u in prod_eval_history
            }
        arms[arm] = {
            "stream_config": stream_configs[arm],
            "working_tree_clean_at_start": tree_clean,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "telemetry": telemetry,
            "probes": probes,
            "wall_seconds": round(wall, 1),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "r_matches_production_eval_history": r_vs_production,
        }
        del model, optimizer
        torch.cuda.empty_cache()

    def improved(arm_name: str) -> bool:
        return arms[arm_name]["probes"]["T2000"]["eval_ID"]["cmdr_sma"] >= 0.45

    def chance(arm_name: str) -> bool:
        return arms[arm_name]["probes"]["T2000"]["eval_ID"]["cmdr_sma"] <= 0.36

    fc_val = arms["FC"]["probes"]["T2000"]["eval_ID"]["cmdr_sma"]
    r_val = arms["R"]["probes"]["T2000"]["eval_ID"]["cmdr_sma"]
    if improved("FC") and chance("R"):
        pattern = "FC_IMPROVES_R_CHANCE"
    elif chance("FC") and chance("R"):
        pattern = "BOTH_REMAIN_CHANCE"
    elif improved("FC") and improved("R") and abs(fc_val - r_val) < 0.05:
        pattern = "BOTH_IMPROVE_SIMILARLY"
    else:
        pattern = "MIXED_OR_INCONCLUSIVE"

    return diag_header({
        "schema_id": "E0-Q2-DIAG-D4A-FAMILY-COHERENT-BATCHING-v0",
        "authority": "issue #3 D4-A (family-coherent batching falsifier); all other D4 factors held",
        "diagnostic_only": True,
        "non_scientific": True,
        "candidate": "C0", "seed": DIAG_SEED, "updates": updates,
        "wd_scope": "exclude_norm_bias",
        "working_tree_clean_at_start": tree_clean,
        "code_git_commit": subprocess_git_head(REPO_ROOT),
        "parameter_count": arms["R"]["parameter_count"],
        "verified_q1_input_digests": ctx["verified_digests"],
        "family_stream_config_sha256": stream_config_digest,
        "eval_probe_family_list_sha256": probe_digest,
        "batch_contract": {"effective_batch": 128, "microbatch": 16, "accumulation": 8,
                           "fc_complete_families_per_batch": 42, "fc_boundary_examples": 2},
        "pattern_criteria": {
            "improved": "eval_ID CMDR-SMA at T2000 >= 0.45",
            "chance": "eval_ID CMDR-SMA at T2000 <= 0.36",
            "both_improve_similarly": "both improved and |FC - R| < 0.05",
        },
        "arms": arms,
        "observed_pattern": pattern,
    })


def _d3_probe_geometry(model, probe_rows: list[dict], device) -> dict:
    """Unseen-family D3-style geometry: within-family distances and cross-family
    displacement alignment among the frozen probe families."""
    n_fam = len(probe_rows) // 3
    states = _decide_states(model, probe_rows, device)
    within, _residuals, displacement = _geometry(states, n_fam)
    displacement.pop("vectors", None)
    return {"within_family": within, "cross_family_displacement": displacement}


# ------------------------------------------------------------------- D4-B ----

D4B_LB_NAMESPACE = "E0-Q2-DIAG|D4B|lb"
LB_C_SHIFT = 2667
LB_U_SHIFT = 5334


class LabelBalancedFamilyDisjointStream:
    """D4-B LB arm: identical global E/C/U label sequence and identical per-128
    43/43/42 balance cycle to the FC stream (slot s -> variant s mod 3), but
    the families filling the E-, C-, and U-slots are drawn from three disjoint
    segments of the same epoch family permutation (shifts LB_C_SHIFT /
    LB_U_SHIFT). A 128-example batch spans a <=43-wide window of k, far below
    both shifts, so no family can appear twice within any batch. Every one of
    the 24,000 examples is consumed exactly once per stream epoch."""

    def __init__(self, family_rows: list[dict], sid_to_idx: dict[str, int], seed: int) -> None:
        import random as _random

        self._random = _random
        self.seed = seed
        self.sid_to_idx = sid_to_idx
        by_family: dict[str, list[dict]] = {}
        for r in family_rows:
            by_family.setdefault(r["family_id"], []).append(r)
        self.families = sorted(by_family)
        if len(self.families) <= LB_U_SHIFT + 48:
            raise AssertionError("family count too small for LB shifts")
        self.variant_of = {
            fid: {r["label_id"]: r["sample_id"] for r in vs} for fid, vs in by_family.items()
        }
        self._epoch = 0
        self._buffer: list[int] = []

    def _epoch_flat(self, epoch: int) -> list[int]:
        digest = hashlib.sha256(f"{D4B_LB_NAMESPACE}|{self.seed}|{epoch}".encode()).digest()
        rng = self._random.Random(int.from_bytes(digest[:8], "big"))
        fams = list(self.families)
        rng.shuffle(fams)
        n = len(fams)
        flat: list[int] = []
        for k in range(n):
            flat.append(self.sid_to_idx[self.variant_of[fams[k]][0]])
            flat.append(self.sid_to_idx[self.variant_of[fams[(k + LB_C_SHIFT) % n]][1]])
            flat.append(self.sid_to_idx[self.variant_of[fams[(k + LB_U_SHIFT) % n]][2]])
        return flat

    def take(self, count: int) -> list[int]:
        out: list[int] = []
        while len(out) < count:
            if not self._buffer:
                self._buffer = self._epoch_flat(self._epoch)
                self._epoch += 1
            need = count - len(out)
            out.extend(self._buffer[:need])
            self._buffer = self._buffer[need:]
        return out


def run_d4b(updates: int = 2000) -> dict:
    """D4-B label-balanced family-disjoint decomposition: FC (family-coherent)
    vs LB (same label sequence/balance, no family co-location) on the full 24k
    corpus, 2,000 updates, production LR function."""
    raw_train_path = REPO_ROOT / "local_data/e0_qualification_bootstrap/Q1/train_ID.jsonl"
    raw_train = [json.loads(line) for line in raw_train_path.open(encoding="utf-8")]
    raw_eval_path = REPO_ROOT / "local_data/e0_qualification_bootstrap/Q1/eval_ID.jsonl"
    raw_eval = [json.loads(line) for line in raw_eval_path.open(encoding="utf-8")]
    probe_rows, probe_digest = _d4a_probe_rows(raw_eval)

    d4a_path = DIAG_DIR / "d4a_family_coherent_batching.json"
    d4a_fc_history = None
    if d4a_path.exists():
        d4a = json.loads(d4a_path.read_text(encoding="utf-8"))
        d4a_fc_history = {
            int(tag[1:]): p["eval_ID"]["cmdr_sma"]
            for tag, p in d4a["arms"]["FC"]["probes"].items()
        }

    stream_configs = {
        "FC": {
            "stream": "family-unit permutation; E/C/U variants contiguous (fixed label order)",
            "namespace": FC_STREAM_NAMESPACE,
        },
        "LB": {
            "stream": "global E/C/U slot sequence identical to FC; E/C/U slot families drawn from disjoint permutation segments",
            "namespace": D4B_LB_NAMESPACE,
            "shifts": {"C": LB_C_SHIFT, "U": LB_U_SHIFT},
            "per_batch_family_disjoint": True,
        },
    }
    stream_config_digest = hashlib.sha256(json.dumps(stream_configs, sort_keys=True).encode()).hexdigest()

    arms: dict[str, dict] = {}
    for arm in ("FC", "LB"):
        ctx = scientific_setup("C0", "exclude_norm_bias", DIAG_SEED,
                               REPO_ROOT / "local_data/e0_qualification_bootstrap/Q1", REPO_ROOT)
        device, model, optimizer = ctx["device"], ctx["model"], ctx["optimizer"]
        train_rows = ctx["train_rows"]
        tree_clean = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip() == ""

        fc_rows = _build_family_rows(raw_train, sorted({r["family_id"] for r in raw_train}))
        sid_to_idx = {r["sample_id"]: i for i, r in enumerate(train_rows)}
        if arm == "FC":
            stream = FamilyCoherentStream(fc_rows, sid_to_idx, DIAG_SEED)
        else:
            stream = LabelBalancedFamilyDisjointStream(fc_rows, sid_to_idx, DIAG_SEED)
        take = stream.take

        prev_flat = torch.cat([p.detach().flatten() for p in model.parameters()])
        telemetry: list[dict] = []
        probes: dict[str, dict] = {}

        def capture(tag: str, update: int) -> None:
            probes[tag] = {
                "update": update,
                "train_ID": _eval_surface_metrics(model, train_rows, device),
                "eval_ID": _eval_surface_metrics(model, ctx["eval_id_rows"], device),
                "eval_STRUCT": _eval_surface_metrics(model, ctx["eval_struct_rows"], device),
                "unseen_family_geometry": _d3_probe_geometry(model, probe_rows, device),
            }
            print(
                f"[D4B {arm}] probe {tag}: train SMA={probes[tag]['train_ID']['cmdr_sma']:.4f} "
                f"eval_ID SMA={probes[tag]['eval_ID']['cmdr_sma']:.6f} "
                f"train_CE={probes[tag]['train_ID']['ce_mean']:.4f}",
                flush=True,
            )

        torch.cuda.reset_peak_memory_stats(device)
        capture("T0", 0)
        started = time.time()
        for update in range(1, updates + 1):
            lr = learning_rate(update)
            for group in optimizer.param_groups:
                group["lr"] = lr
            batch_indices = take(128)
            optimizer.zero_grad(set_to_none=True)
            micro_losses, micro_argmaxes = [], []
            for micro in range(GRAD_ACCUM):
                rows = [train_rows[i] for i in batch_indices[micro * MICROBATCH : (micro + 1) * MICROBATCH]]
                ids, decide, labels = collate(rows, device)
                logits = model(ids, decide)
                loss = F.cross_entropy(logits, labels, label_smoothing=0.0)
                micro_losses.append(loss.detach())
                with torch.no_grad():
                    micro_argmaxes.append(logits.detach().argmax(dim=-1))
                (loss / GRAD_ACCUM).backward()
            with torch.no_grad():
                embed_grad_norm = float(model.embed_tokens.weight.grad.norm())
                classifier_grad_norm = float(model.classifier.weight.grad.norm())
                batch_ce = float(torch.stack(micro_losses).mean())
                argmax = torch.cat(micro_argmaxes).cpu()
                hist = torch.bincount(argmax, minlength=3).tolist()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            with torch.no_grad():
                flat = torch.cat([p.detach().flatten() for p in model.parameters()])
                param_delta_norm = float((flat - prev_flat).norm())
                prev_flat = flat
            if update <= 200 or update % 10 == 0:
                telemetry.append({
                    "update": update, "lr": lr, "batch_ce_mean": batch_ce,
                    "pred_hist": hist, "grad_norm_pre_clip": float(grad_norm),
                    "classifier_w_grad_norm": classifier_grad_norm,
                    "embed_grad_norm": embed_grad_norm,
                    "param_delta_norm": param_delta_norm,
                })
            if update % 400 == 0:
                capture(f"T{update}", update)
        wall = time.time() - started

        fc_vs_d4a = None
        if arm == "FC" and d4a_fc_history:
            fc_vs_d4a = {
                str(u): probes[f"T{u}"]["eval_ID"]["cmdr_sma"] == d4a_fc_history[u]
                for u in (0, 400, 800, 1200, 1600, 2000) if u in d4a_fc_history
            }
        arms[arm] = {
            "stream_config": stream_configs[arm],
            "working_tree_clean_at_start": tree_clean,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "telemetry": telemetry,
            "probes": probes,
            "wall_seconds": round(wall, 1),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "fc_scalar_reproduces_d4a": fc_vs_d4a,
        }
        del model, optimizer
        torch.cuda.empty_cache()

    def engaged(arm_name: str) -> bool:
        final_ces = [t["batch_ce_mean"] for t in arms[arm_name]["telemetry"] if t["update"] >= 1950]
        return sum(final_ces) / len(final_ces) <= 1.05

    def transfers(arm_name: str) -> bool:
        return arms[arm_name]["probes"]["T2000"]["eval_ID"]["cmdr_sma"] >= 0.45

    fc_final_ce = arms["FC"]["probes"]["T2000"]["train_ID"]["ce_mean"]
    lb_final_ce = arms["LB"]["probes"]["T2000"]["train_ID"]["ce_mean"]
    if engaged("FC") and not engaged("LB"):
        pattern = "COLOCATION_CAUSES_ENGAGEMENT"
    elif engaged("FC") and engaged("LB") and abs(fc_final_ce - lb_final_ce) < 0.05 and not transfers("FC") and not transfers("LB"):
        pattern = "LABEL_BALANCE_EXPLAINS_ENGAGEMENT"
    elif transfers("LB") and not transfers("FC"):
        pattern = "LB_TRANSFERS_FC_NOT"
    elif transfers("FC") and not transfers("LB"):
        pattern = "FC_TRANSFERS_LB_NOT"
    else:
        pattern = "INCONCLUSIVE"

    return diag_header({
        "schema_id": "E0-Q2-DIAG-D4B-LABEL-BALANCED-DISJOINT-v0",
        "authority": "issue #3 D4-B (label-balanced, family-disjoint batching decomposition); all other factors held",
        "diagnostic_only": True,
        "non_scientific": True,
        "candidate": "C0", "seed": DIAG_SEED, "updates": updates,
        "wd_scope": "exclude_norm_bias",
        "working_tree_clean_at_start": tree_clean,
        "code_git_commit": subprocess_git_head(REPO_ROOT),
        "parameter_count": arms["FC"]["parameter_count"],
        "verified_q1_input_digests": ctx["verified_digests"],
        "family_stream_config_sha256": stream_config_digest,
        "eval_probe_family_list_sha256": probe_digest,
        "batch_contract": {"effective_batch": 128, "microbatch": 16, "accumulation": 8,
                           "label_balance_per_128": "43/43/42 (identical FC/LB global E/C/U slot sequence)",
                           "lb_family_disjoint_within_batch": True},
        "pattern_criteria": {
            "engaged": "mean batch CE over updates >=1950 <= 1.05 (ln3 = 1.0986)",
            "transfers": "eval_ID CMDR-SMA at T2000 >= 0.45",
            "label_balance_explains": "both engaged, |train_CE_FC - train_CE_LB| < 0.05, neither transfers",
        },
        "arms": arms,
        "observed_pattern": pattern,
    })


# ------------------------------------------------------------------ D4-B2 ----


class MatchedPermutationFamilyDisjointStream:
    """D4-B2 LB-MP arm: family permutation derived from the EXACT FC rule
    (namespace E0-Q2-DIAG|D4A|fc, same seed and per-epoch SHA-256 hashing),
    then E/C/U slots filled from that permutation at offsets 0 / 2667 / 5334.
    Removes co-location while holding the family presentation permutation,
    the global label sequence, and the 43/43/42 balance identical to FC."""

    def __init__(self, family_rows: list[dict], sid_to_idx: dict[str, int], seed: int) -> None:
        import random as _random

        self._random = _random
        self.seed = seed
        self.sid_to_idx = sid_to_idx
        by_family: dict[str, list[dict]] = {}
        for r in family_rows:
            by_family.setdefault(r["family_id"], []).append(r)
        self.families = sorted(by_family)
        self.variant_of = {
            fid: {r["label_id"]: r["sample_id"] for r in vs} for fid, vs in by_family.items()
        }
        self._epoch = 0
        self._buffer: list[int] = []

    def _fc_epoch_permutation(self, epoch: int) -> list[str]:
        digest = hashlib.sha256(f"{FC_STREAM_NAMESPACE}|{self.seed}|{epoch}".encode()).digest()
        rng = self._random.Random(int.from_bytes(digest[:8], "big"))
        fams = list(self.families)
        rng.shuffle(fams)
        return fams

    def _epoch_flat(self, epoch: int) -> list[int]:
        fams = self._fc_epoch_permutation(epoch)
        n = len(fams)
        flat: list[int] = []
        for k in range(n):
            flat.append(self.sid_to_idx[self.variant_of[fams[k]][0]])
            flat.append(self.sid_to_idx[self.variant_of[fams[(k + LB_C_SHIFT) % n]][1]])
            flat.append(self.sid_to_idx[self.variant_of[fams[(k + LB_U_SHIFT) % n]][2]])
        return flat

    def take(self, count: int) -> list[int]:
        out: list[int] = []
        while len(out) < count:
            if not self._buffer:
                self._buffer = self._epoch_flat(self._epoch)
                self._epoch += 1
            need = count - len(out)
            out.extend(self._buffer[:need])
            self._buffer = self._buffer[need:]
        return out


def _verify_lbmp(lbmp: MatchedPermutationFamilyDisjointStream,
                 fc: FamilyCoherentStream,
                 raw_rows: list[dict], n_batches: int) -> dict:
    """Fail-closed pre-training verification per the D4-B2 release."""
    from collections import Counter

    n = len(lbmp.families)

    perm_digest_checks, e_slot_checks, epoch_coverage_checks = [], [], []
    for epoch in range(0, (n_batches * 128) // (3 * n) + 2):
        lbmp_perm = lbmp._fc_epoch_permutation(epoch)
        # FC's flat stream is family blocks (3 indices each) in presentation
        # order; production index i corresponds to raw_rows[i] (same file order)
        fc_flat = fc._epoch_flat(epoch)
        fc_perm = [raw_rows[fc_flat[pos]]["family_id"] for pos in range(0, len(fc_flat), 3)]
        d1 = hashlib.sha256(json.dumps(lbmp_perm).encode()).hexdigest()
        d2 = hashlib.sha256(json.dumps(fc_perm).encode()).hexdigest()
        perm_digest_checks.append(d1 == d2)
        e_slot_checks.append(all(lbmp_perm[k] == fc_perm[k] for k in range(n)))
        flat = lbmp._epoch_flat(epoch)
        epoch_coverage_checks.append(sorted(flat) == list(range(len(raw_rows))))

    # Fresh streams for the horizon-level checks (label sequence identity and
    # family multiplicity in every batch, including epoch-boundary crossings).
    lbmp2 = MatchedPermutationFamilyDisjointStream.__new__(MatchedPermutationFamilyDisjointStream)
    lbmp2.__dict__.update({k: v for k, v in lbmp.__dict__.items() if k not in ("_buffer", "_epoch")})
    lbmp2._epoch, lbmp2._buffer = 0, []
    fc2 = FamilyCoherentStream.__new__(FamilyCoherentStream)
    fc2.__dict__.update({k: v for k, v in fc.__dict__.items() if k not in ("_buffer", "_epoch")})
    fc2._epoch, fc2._buffer = 0, []

    lid = {r["sample_id"]: LABELS.index(r["gold_label"]) for r in raw_rows}
    seq_ok, mult_violations, max_mult = True, [], 0
    for b in range(n_batches):
        fb = fc2.take(128)
        lb = lbmp2.take(128)
        if [lid[raw_rows[i]["sample_id"]] for i in fb] != [lid[raw_rows[i]["sample_id"]] for i in lb]:
            seq_ok = False
        c = Counter(raw_rows[i]["family_id"] for i in lb)
        m = max(c.values())
        max_mult = max(max_mult, m)
        if m > 1:
            mult_violations.append({"batch": b, "max_multiplicity": m})

    return {
        "permutation_digest_matches_fc_all_epochs": all(perm_digest_checks),
        "epochs_checked": len(perm_digest_checks),
        "e_slot_family_matches_fc_all_epochs": all(e_slot_checks),
        "epoch_coverage_exact_all_epochs": all(epoch_coverage_checks),
        "label_sequence_identical_to_fc_all_batches": seq_ok,
        "batches_checked": n_batches,
        "max_family_multiplicity_any_batch_including_boundaries": max_mult,
        "multiplicity_violations": mult_violations[:10],
        "ok": (
            all(perm_digest_checks) and all(e_slot_checks) and all(epoch_coverage_checks)
            and seq_ok and not mult_violations
        ),
    }


def run_d4b2(updates: int = 2000) -> dict:
    """D4-B2 matched-permutation family-disjoint control: a single LB-MP arm —
    same FC permutation, same label sequence/balance, no co-location."""
    raw_train_path = REPO_ROOT / "local_data/e0_qualification_bootstrap/Q1/train_ID.jsonl"
    raw_train = [json.loads(line) for line in raw_train_path.open(encoding="utf-8")]
    raw_eval_path = REPO_ROOT / "local_data/e0_qualification_bootstrap/Q1/eval_ID.jsonl"
    raw_eval = [json.loads(line) for line in raw_eval_path.open(encoding="utf-8")]
    probe_rows, probe_digest = _d4a_probe_rows(raw_eval)

    d4b = json.loads((DIAG_DIR / "d4b_label_balanced_disjoint.json").read_text(encoding="utf-8"))
    d4b_fc_final_ce = d4b["arms"]["FC"]["probes"]["T2000"]["train_ID"]["ce_mean"]

    ctx = scientific_setup("C0", "exclude_norm_bias", DIAG_SEED,
                           REPO_ROOT / "local_data/e0_qualification_bootstrap/Q1", REPO_ROOT)
    device, model, optimizer = ctx["device"], ctx["model"], ctx["optimizer"]
    train_rows = ctx["train_rows"]
    tree_clean = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip() == ""

    fc_rows = _build_family_rows(raw_train, sorted({r["family_id"] for r in raw_train}))
    sid_to_idx = {r["sample_id"]: i for i, r in enumerate(train_rows)}
    fc = FamilyCoherentStream(fc_rows, sid_to_idx, DIAG_SEED)
    stream = MatchedPermutationFamilyDisjointStream(fc_rows, sid_to_idx, DIAG_SEED)

    n_batches = updates
    verification = _verify_lbmp(stream, fc, raw_train, n_batches)
    if not verification["ok"]:
        (DIAG_DIR / "d4b2_verification_failure.json").write_text(
            json.dumps(verification, indent=2), encoding="utf-8"
        )
        raise SystemExit(f"D4-B2 verification failed: {json.dumps({k: v for k, v in verification.items() if k != 'ok'})[:400]}")

    prev_flat = torch.cat([p.detach().flatten() for p in model.parameters()])
    telemetry: list[dict] = []
    probes: dict[str, dict] = {}

    def capture(tag: str, update: int) -> None:
        probes[tag] = {
            "update": update,
            "train_ID": _eval_surface_metrics(model, train_rows, device),
            "eval_ID": _eval_surface_metrics(model, ctx["eval_id_rows"], device),
            "eval_STRUCT": _eval_surface_metrics(model, ctx["eval_struct_rows"], device),
            "unseen_family_geometry": _d3_probe_geometry(model, probe_rows, device),
        }
        print(
            f"[D4B2 LB-MP] probe {tag}: train SMA={probes[tag]['train_ID']['cmdr_sma']:.4f} "
            f"eval_ID SMA={probes[tag]['eval_ID']['cmdr_sma']:.6f} "
            f"train_CE={probes[tag]['train_ID']['ce_mean']:.4f}",
            flush=True,
        )

    torch.cuda.reset_peak_memory_stats(device)
    capture("T0", 0)
    started = time.time()
    for update in range(1, updates + 1):
        lr = learning_rate(update)
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch_indices = stream.take(128)
        optimizer.zero_grad(set_to_none=True)
        micro_losses, micro_argmaxes = [], []
        for micro in range(GRAD_ACCUM):
            rows = [train_rows[i] for i in batch_indices[micro * MICROBATCH : (micro + 1) * MICROBATCH]]
            ids, decide, labels = collate(rows, device)
            logits = model(ids, decide)
            loss = F.cross_entropy(logits, labels, label_smoothing=0.0)
            micro_losses.append(loss.detach())
            with torch.no_grad():
                micro_argmaxes.append(logits.detach().argmax(dim=-1))
            (loss / GRAD_ACCUM).backward()
        with torch.no_grad():
            embed_grad_norm = float(model.embed_tokens.weight.grad.norm())
            classifier_grad_norm = float(model.classifier.weight.grad.norm())
            batch_ce = float(torch.stack(micro_losses).mean())
            argmax = torch.cat(micro_argmaxes).cpu()
            hist = torch.bincount(argmax, minlength=3).tolist()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        with torch.no_grad():
            flat = torch.cat([p.detach().flatten() for p in model.parameters()])
            param_delta_norm = float((flat - prev_flat).norm())
            prev_flat = flat
        if update <= 200 or update % 10 == 0:
            telemetry.append({
                "update": update, "lr": lr, "batch_ce_mean": batch_ce,
                "pred_hist": hist, "grad_norm_pre_clip": float(grad_norm),
                "classifier_w_grad_norm": classifier_grad_norm,
                "embed_grad_norm": embed_grad_norm,
                "param_delta_norm": param_delta_norm,
            })
        if update % 400 == 0:
            capture(f"T{update}", update)
    wall = time.time() - started

    final_telemetry_ce = [t["batch_ce_mean"] for t in telemetry if t["update"] >= 1950]
    mean_final_ce = sum(final_telemetry_ce) / len(final_telemetry_ce)
    final_train_ce = probes["T2000"]["train_ID"]["ce_mean"]
    if mean_final_ce >= 1.08 and final_train_ce >= 1.05:
        pattern = "COLOCATION_CAUSES_ENGAGEMENT_CONFIRMED"
    elif final_train_ce <= d4b_fc_final_ce + 0.05:
        pattern = "ORDER_CONFOUND_MATERIAL"
    else:
        pattern = "INCONCLUSIVE"

    return diag_header({
        "schema_id": "E0-Q2-DIAG-D4B2-MATCHED-PERMUTATION-CONTROL-v0",
        "authority": "issue #3 D4-B2 (matched-permutation family-disjoint control); D4-C and all other factors held",
        "diagnostic_only": True,
        "non_scientific": True,
        "candidate": "C0", "seed": DIAG_SEED, "updates": updates,
        "wd_scope": "exclude_norm_bias",
        "working_tree_clean_at_start": tree_clean,
        "code_git_commit": subprocess_git_head(REPO_ROOT),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "verified_q1_input_digests": ctx["verified_digests"],
        "eval_probe_family_list_sha256": probe_digest,
        "stream_verification": verification,
        "d4b_fc_reference_final_train_ce": d4b_fc_final_ce,
        "pattern_criteria": {
            "confirmed": "mean batch CE (u>=1950) >= 1.08 AND T2000 train CE >= 1.05 (pinned at ln3)",
            "order_confound": "T2000 train CE <= D4-B FC T2000 train CE + 0.05",
        },
        "arm": {
            "stream_config": {
                "stream": "FC-rule family permutation (namespace E0-Q2-DIAG|D4A|fc) + E/C/U offsets 0/2667/5334",
                "offsets": {"E": 0, "C": LB_C_SHIFT, "U": LB_U_SHIFT},
                "family_disjoint_within_batch": True,
            },
            "telemetry": telemetry,
            "probes": probes,
            "wall_seconds": round(wall, 1),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
        "observed_pattern": pattern,
    })


# ------------------------------------------------------------------ D4-B3 ----


def _probe_metric(probe: dict, path: tuple[str, ...]):
    value = probe
    for key in path:
        value = value[key]
    return value


_D4B_PREFIX_METRICS = [
    ("train_SMA", ("train_ID", "cmdr_sma")),
    ("train_CE", ("train_ID", "ce_mean")),
    ("eval_ID_SMA", ("eval_ID", "cmdr_sma")),
    ("eval_STRUCT_SMA", ("eval_STRUCT", "cmdr_sma")),
    ("unseen_family_E_minus_C_alignment",
     ("unseen_family_geometry", "cross_family_displacement", "E_minus_C", "mean_pairwise_cos")),
]
_D4B_PREFIX_TAGS = ["T0", "T400", "T800", "T1200", "T1600", "T2000"]


def run_d4b3(updates: int = 8000, prefix_updates: int = 2000) -> dict:
    """D4-B3 / FC-8K: single family-coherent arm at the full 8,000-update
    horizon. Fail-closed prefix gate at T{prefix_updates}: the run must exactly
    reproduce the D4-B FC arm on five metrics at six probes before the
    remaining updates are spent."""
    raw_train_path = REPO_ROOT / "local_data/e0_qualification_bootstrap/Q1/train_ID.jsonl"
    raw_train = [json.loads(line) for line in raw_train_path.open(encoding="utf-8")]
    raw_eval_path = REPO_ROOT / "local_data/e0_qualification_bootstrap/Q1/eval_ID.jsonl"
    raw_eval = [json.loads(line) for line in raw_eval_path.open(encoding="utf-8")]
    probe_rows, probe_digest = _d4a_probe_rows(raw_eval)

    d4b = json.loads((DIAG_DIR / "d4b_label_balanced_disjoint.json").read_text(encoding="utf-8"))
    d4b_fc_probes = d4b["arms"]["FC"]["probes"]

    ctx = scientific_setup("C0", "exclude_norm_bias", DIAG_SEED,
                           REPO_ROOT / "local_data/e0_qualification_bootstrap/Q1", REPO_ROOT)
    device, model, optimizer = ctx["device"], ctx["model"], ctx["optimizer"]
    train_rows = ctx["train_rows"]
    tree_clean = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip() == ""

    fc_rows = _build_family_rows(raw_train, sorted({r["family_id"] for r in raw_train}))
    sid_to_idx = {r["sample_id"]: i for i, r in enumerate(train_rows)}
    stream = FamilyCoherentStream(fc_rows, sid_to_idx, DIAG_SEED)

    telemetry: list[dict] = []
    probes: dict[str, dict] = {}
    prefix_batch_hasher = hashlib.sha256()

    def capture(tag: str, update: int) -> None:
        probes[tag] = {
            "update": update,
            "train_ID": _eval_surface_metrics(model, train_rows, device),
            "eval_ID": _eval_surface_metrics(model, ctx["eval_id_rows"], device),
            "eval_STRUCT": _eval_surface_metrics(model, ctx["eval_struct_rows"], device),
            "unseen_family_geometry": _d3_probe_geometry(model, probe_rows, device),
        }
        print(
            f"[D4B3 FC-8K] probe {tag}: train SMA={probes[tag]['train_ID']['cmdr_sma']:.4f} "
            f"eval_ID SMA={probes[tag]['eval_ID']['cmdr_sma']:.6f} "
            f"eval_STRUCT SMA={probes[tag]['eval_STRUCT']['cmdr_sma']:.6f} "
            f"train_CE={probes[tag]['train_ID']['ce_mean']:.4f}",
            flush=True,
        )

    torch.cuda.reset_peak_memory_stats(device)
    capture("T0", 0)
    started = time.time()
    for update in range(1, updates + 1):
        lr = learning_rate(update)
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch_indices = stream.take(128)
        if update <= prefix_updates:
            prefix_batch_hasher.update(json.dumps(batch_indices).encode())
        optimizer.zero_grad(set_to_none=True)
        micro_losses, micro_argmaxes = [], []
        for micro in range(GRAD_ACCUM):
            rows = [train_rows[i] for i in batch_indices[micro * MICROBATCH : (micro + 1) * MICROBATCH]]
            ids, decide, labels = collate(rows, device)
            logits = model(ids, decide)
            loss = F.cross_entropy(logits, labels, label_smoothing=0.0)
            micro_losses.append(loss.detach())
            with torch.no_grad():
                micro_argmaxes.append(logits.detach().argmax(dim=-1))
            (loss / GRAD_ACCUM).backward()
        with torch.no_grad():
            embed_grad_norm = float(model.embed_tokens.weight.grad.norm())
            classifier_grad_norm = float(model.classifier.weight.grad.norm())
            batch_ce = float(torch.stack(micro_losses).mean())
            argmax = torch.cat(micro_argmaxes).cpu()
            hist = torch.bincount(argmax, minlength=3).tolist()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        if update <= 200 or update % 10 == 0:
            telemetry.append({
                "update": update, "lr": lr, "batch_ce_mean": batch_ce,
                "pred_hist": hist, "grad_norm_pre_clip": float(grad_norm),
                "classifier_w_grad_norm": classifier_grad_norm,
                "embed_grad_norm": embed_grad_norm,
            })
        if update % 400 == 0:
            capture(f"T{update}", update)
        if update == prefix_updates:
            prefix_hash = prefix_batch_hasher.hexdigest()
            mismatches = []
            for tag in _D4B_PREFIX_TAGS:
                for name, path in _D4B_PREFIX_METRICS:
                    mine = _probe_metric(probes[tag], path)
                    ref = _probe_metric(d4b_fc_probes[tag], path)
                    if mine != ref:
                        mismatches.append({"probe": tag, "metric": name, "new": mine, "d4b_fc": ref})
            prefix_report = {
                "prefix_updates": prefix_updates,
                "first_batch_sha256": prefix_hash,
                "probes_compared": _D4B_PREFIX_TAGS,
                "metrics_compared": [name for name, _ in _D4B_PREFIX_METRICS],
                "mismatches": mismatches,
                "reproduced": not mismatches,
            }
            print(f"[D4B3] prefix gate at T{prefix_updates}: reproduced={prefix_report['reproduced']} "
                  f"batch-hash={prefix_hash[:16]}", flush=True)
            if mismatches:
                DIAG_DIR.mkdir(parents=True, exist_ok=True)
                (DIAG_DIR / "d4b3_prefix_failure.json").write_text(
                    json.dumps(prefix_report, indent=2), encoding="utf-8"
                )
                raise SystemExit(
                    f"D4-B3 prefix gate FAILED at T{prefix_updates}: "
                    f"{len(mismatches)} metric mismatches vs D4-B FC; run stopped."
                )
    wall = time.time() - started

    final = probes[f"T{updates}"]
    eid = final["eval_ID"]["cmdr_sma"]
    estr = final["eval_STRUCT"]["cmdr_sma"]
    tr = final["train_ID"]["cmdr_sma"]
    if eid >= 0.45 and estr >= 0.45:
        endpoint = "FULL_HORIZON_TRANSFER"
    elif eid >= 0.45 or estr >= 0.45 or eid > 0.36 or estr > 0.36:
        endpoint = "ID_ONLY_OR_PARTIAL_TRANSFER"
    elif tr >= 0.90:
        endpoint = "FIT_NO_TRANSFER"
    else:
        endpoint = "NO_TRANSFER_UNDERFIT"

    return diag_header({
        "schema_id": "E0-Q2-DIAG-D4B3-FC-8K-v0",
        "authority": "issue #3 D4-B3 / FC-8K (8,000-update family-coherent extension); D4-C and all other factors held; result does not reopen v0.5 Q2",
        "diagnostic_only": True,
        "non_scientific": True,
        "candidate": "C0", "seed": DIAG_SEED, "updates": updates,
        "wd_scope": "exclude_norm_bias",
        "working_tree_clean_at_start": tree_clean,
        "code_git_commit": subprocess_git_head(REPO_ROOT),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "verified_q1_input_digests": ctx["verified_digests"],
        "eval_probe_family_list_sha256": probe_digest,
        "prefix_reproduction_gate": prefix_report,
        "endpoint_criteria": {
            "FULL_HORIZON_TRANSFER": "eval_ID >= 0.45 AND eval_STRUCT >= 0.45",
            "ID_ONLY_OR_PARTIAL_TRANSFER": "one surface >= 0.45, or either > 0.36 but < 0.45",
            "FIT_NO_TRANSFER": "both <= 0.36 AND train_ID >= 0.90",
            "NO_TRANSFER_UNDERFIT": "both <= 0.36 AND train_ID < 0.90",
        },
        "final_diagnostics": {
            "Q": (eid + estr) / 2.0,
            "Q_ID": eid,
            "Q_STRUCT": estr,
            "train_ID_SMA": tr,
            "train_ID_CE": final["train_ID"]["ce_mean"],
            "Q_2_4_ID": final["eval_ID"]["sma_depth_2_4"],
            "Q_2_4_STRUCT": final["eval_STRUCT"]["sma_depth_2_4"],
            "min_label_recall_ID": final["eval_ID"]["min_label_recall"],
            "min_label_recall_STRUCT": final["eval_STRUCT"]["min_label_recall"],
            "unseen_family_E_minus_C_alignment":
                final["unseen_family_geometry"]["cross_family_displacement"]["E_minus_C"]["mean_pairwise_cos"],
        },
        "probes": probes,
        "telemetry": telemetry,
        "observed_endpoint": endpoint,
        "wall_seconds": round(wall, 1),
    })


# ------------------------------------------------------------------- D4-C ----

FRA_LAMBDA = 1.0
FRA_EPS = 1e-6
FRA_MIN_FAMILIES = 3


class ClassifierDecideStateHook:
    """Forward pre-hook on model.classifier capturing the post-final-RMSNorm
    <DECIDE> state (the classifier's input). m0_model.py is NOT modified; the
    captured tensor keeps its autograd graph so the FRA loss shares the task
    graph and both losses get one combined backward per microbatch."""

    def __init__(self, classifier: torch.nn.Module) -> None:
        self.captured: dict = {"tensor": None, "active": False}
        classifier.register_forward_pre_hook(self._hook)

    def _hook(self, module, args):
        if self.captured["active"]:
            self.captured["tensor"] = args[0]
        return None


def _fra_loss_from_grouped(h: torch.Tensor) -> tuple[torch.Tensor, int]:
    """FRA loss for h grouped family-major with per-family label order E/C/U
    (shape (F,3,D)). Leave-one-family-out normalized prototypes; cross-entropy
    of auxiliary logits against each slot's own label; no temperature."""
    n_fam = h.shape[0]
    r = h - h.mean(dim=1, keepdim=True)
    z = r / torch.clamp(r.norm(dim=-1, keepdim=True), min=FRA_EPS)
    total = z.sum(dim=0)  # (3, D)
    mu = (total.unsqueeze(0) - z) / (n_fam - 1)  # (F, 3, D)
    mu = F.normalize(mu, dim=-1)
    logits = torch.einsum("fld,fkd->flk", z, mu)  # (F, 3, 3)
    targets = torch.arange(3, device=h.device).unsqueeze(0).expand(n_fam, 3).reshape(-1)
    return F.cross_entropy(logits.reshape(-1, 3), targets), n_fam


def _fra_loss_microbatch(h: torch.Tensor, rows: list[dict], fam_of_sid: dict[str, str]) -> tuple[torch.Tensor, int]:
    """Group a microbatch's captured <DECIDE> states by family (resolved via
    fam_of_sid: production rows carry sample_id/label_id but not family_id);
    use only complete E/C/U families wholly present. Fails closed on malformed
    groups or fewer than FRA_MIN_FAMILIES contributing families."""
    positions: dict[str, dict[int, int]] = {}
    for pos, row in enumerate(rows):
        positions.setdefault(fam_of_sid[row["sample_id"]], {})[row["label_id"]] = pos
    ordered: list[tuple[str, list[int]]] = []
    for fid, slots in positions.items():
        if set(slots.keys()) == {0, 1, 2}:
            ordered.append((fid, [slots[0], slots[1], slots[2]]))
        elif len(slots) == 3:
            raise AssertionError(f"malformed family group {fid}: 3 members, labels {sorted(slots)}")
    if len(ordered) < FRA_MIN_FAMILIES:
        raise AssertionError(
            f"fail-closed: only {len(ordered)} complete families in microbatch (< {FRA_MIN_FAMILIES})"
        )
    idx = torch.tensor([p for _, slots in ordered for p in slots], device=h.device)
    loss, _ = _fra_loss_from_grouped(h.index_select(0, idx).view(len(ordered), 3, -1))
    assert torch.is_tensor(loss) and loss.dim() == 0 and bool(torch.isfinite(loss)), (
        "FRA loss must be a finite scalar tensor"
    )
    return loss, len(ordered)


def _fc_fra_forward(model, hook: ClassifierDecideStateHook, ids, decide, labels,
                    rows: list[dict], fam_of_sid: dict[str, str]) -> dict:
    """One FC-FRA forward/loss computation, shared verbatim by the D4-C
    training loop and the D4-C preflight (so the preflight rehearses the exact
    production branch). Does NOT backward — call sites own that."""
    hook.captured["active"] = True
    logits = model(ids, decide)
    hook.captured["active"] = False
    h = hook.captured["tensor"]
    task_loss = F.cross_entropy(logits, labels, label_smoothing=0.0)
    fra, n_fam = _fra_loss_microbatch(h, rows, fam_of_sid)
    return {
        "task_loss": task_loss,
        "fra_loss": fra,
        "n_families": n_fam,
        "combined": task_loss + FRA_LAMBDA * fra,
        "captured_state_attached": h is not None and h.requires_grad,
    }


def run_d4c_preflight() -> dict:
    """D4-C preflight (remediation for d4c_launch_incident_001): execute the
    real FC-FRA training branch end-to-end for ONE effective update on a
    freshly constructed C0, then discard it. Non-scientific; fail closed."""
    raw_train = [json.loads(line) for line in
                 (REPO_ROOT / "local_data/e0_qualification_bootstrap/Q1/train_ID.jsonl").open(encoding="utf-8")]
    ctx = scientific_setup("C0", "exclude_norm_bias", DIAG_SEED,
                           REPO_ROOT / "local_data/e0_qualification_bootstrap/Q1", REPO_ROOT)
    device, model, optimizer = ctx["device"], ctx["model"], ctx["optimizer"]
    train_rows = ctx["train_rows"]
    fc_rows = _build_family_rows(raw_train, sorted({r["family_id"] for r in raw_train}))
    sid_to_idx = {r["sample_id"]: i for i, r in enumerate(train_rows)}
    fam_of_sid = {r["sample_id"]: r["family_id"] for r in raw_train}
    stream = FamilyCoherentStream(fc_rows, sid_to_idx, DIAG_SEED)
    hook = ClassifierDecideStateHook(model.classifier)

    micro_reports = []
    checks = {"captured_state_exists_and_attached": True}
    optimizer.zero_grad(set_to_none=True)
    batch_indices = stream.take(128)
    for micro in range(GRAD_ACCUM):
        rows = [train_rows[i] for i in batch_indices[micro * MICROBATCH : (micro + 1) * MICROBATCH]]
        ids, decide, labels = collate(rows, device)
        step = _fc_fra_forward(model, hook, ids, decide, labels, rows, fam_of_sid)
        checks["captured_state_exists_and_attached"] = (
            checks["captured_state_exists_and_attached"] and step["captured_state_attached"]
        )
        (step["combined"] / GRAD_ACCUM).backward()
        micro_reports.append({
            "micro": micro,
            "complete_families": step["n_families"],
            "task_ce": float(step["task_loss"].detach()),
            "fra_loss": float(step["fra_loss"].detach()),
            "combined": float(step["combined"].detach()),
            "losses_finite": bool(torch.isfinite(step["combined"])),
        })
    checks["min_complete_families"] = min(m["complete_families"] for m in micro_reports)
    checks["all_microbatches_ge_3_families"] = checks["min_complete_families"] >= FRA_MIN_FAMILIES
    checks["all_losses_finite"] = all(m["losses_finite"] for m in micro_reports)
    checks["backward_succeeded"] = all(
        p.grad is not None and bool(torch.isfinite(p.grad).all())
        for p in model.parameters() if p.grad is not None
    )
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    checks["grad_norm_finite"] = bool(torch.isfinite(grad_norm))
    optimizer.step()
    checks["optimizer_step_params_finite"] = all(
        bool(torch.isfinite(p.data).all()) for p in model.parameters()
    )
    ok = all([
        checks["captured_state_exists_and_attached"],
        checks["all_microbatches_ge_3_families"],
        checks["all_losses_finite"],
        checks["backward_succeeded"],
        checks["grad_norm_finite"],
        checks["optimizer_step_params_finite"],
    ])
    manifest = diag_header({
        "schema_id": "E0-Q2-DIAG-D4C-PREFLIGHT-v0",
        "authority": "issue #3 D4-C remediation (DIAGNOSTIC_INVALID_PRE_UPDATE / d4c_launch_incident_001); rehearses the real FC-FRA branch",
        "diagnostic_only": True,
        "non_scientific": True,
        "candidate": "C0", "seed": DIAG_SEED, "effective_updates": 1,
        "wd_scope": "exclude_norm_bias",
        "working_tree_clean_at_start": subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip() == "",
        "code_git_commit": subprocess_git_head(REPO_ROOT),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "verified_q1_input_digests": ctx["verified_digests"],
        "fra_config_reference": "frozen in d4c manifest; lambda=1.0, eps=1e-6, no temperature",
        "checks": checks,
        "microbatch_reports": micro_reports,
        "model_discarded": True,
        "status": "PASS" if ok else "FAIL",
    })
    del model, optimizer
    torch.cuda.empty_cache()
    return manifest


def _fra_loss_probe(model, hook: ClassifierDecideStateHook, rows: list[dict], device) -> float:
    """Probe-time FRA loss over a complete 24-family set (single no-grad forward)."""
    model.eval()
    ids, decide, _ = collate(rows, device)
    with torch.no_grad():
        hook.captured["active"] = True
        model(ids, decide)
        hook.captured["active"] = False
        h = hook.captured["tensor"]
        loss, _ = _fra_loss_from_grouped(h.view(len(rows) // 3, 3, -1))
    model.train()
    return float(loss)


def run_d4c(updates: int = 2000) -> dict:
    """D4-C Family-Residual Alignment: paired FC-CE (control) vs FC-FRA
    (task CE + lambda * FRA) on the family-coherent stream, 2,000 updates."""
    raw_train = [json.loads(line) for line in
                 (REPO_ROOT / "local_data/e0_qualification_bootstrap/Q1/train_ID.jsonl").open(encoding="utf-8")]
    raw_eval = [json.loads(line) for line in
                (REPO_ROOT / "local_data/e0_qualification_bootstrap/Q1/eval_ID.jsonl").open(encoding="utf-8")]
    unseen_rows, unseen_digest = _d4a_probe_rows(raw_eval)
    d3_train_rows, _d3_hold, d3_meta = d3_family_sets(raw_train)
    d3_train_digest = d3_meta["train_family_list_sha256"]

    fra_config = {
        "lambda": FRA_LAMBDA,
        "residual_epsilon": FRA_EPS,
        "prototype": "leave-one-family-out normalized mean of normalized residuals",
        "loss": "mean cross-entropy of auxiliary cosine logits a_{f,l,k}=z_{f,l}.mu_{-f,k} against slot label l",
        "temperature": None,
        "min_complete_families_per_microbatch": FRA_MIN_FAMILIES,
        "state_capture": "forward pre-hook on classifier input (post-final-RMSNorm <DECIDE> state)",
        "m0_model_modified": False,
    }

    arms: dict[str, dict] = {}
    for arm in ("FC-CE", "FC-FRA"):
        ctx = scientific_setup("C0", "exclude_norm_bias", DIAG_SEED,
                               REPO_ROOT / "local_data/e0_qualification_bootstrap/Q1", REPO_ROOT)
        device, model, optimizer = ctx["device"], ctx["model"], ctx["optimizer"]
        train_rows = ctx["train_rows"]
        tree_clean = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip() == ""
        fc_rows = _build_family_rows(raw_train, sorted({r["family_id"] for r in raw_train}))
        sid_to_idx = {r["sample_id"]: i for i, r in enumerate(train_rows)}
        fam_of_sid = {r["sample_id"]: r["family_id"] for r in raw_train}
        stream = FamilyCoherentStream(fc_rows, sid_to_idx, DIAG_SEED)
        hook = ClassifierDecideStateHook(model.classifier)

        telemetry: list[dict] = []
        probes: dict[str, dict] = {}
        fra_family_counts: list[int] = []

        def capture(tag: str, update: int) -> None:
            probes[tag] = {
                "update": update,
                "train_ID": _eval_surface_metrics(model, train_rows, device),
                "eval_ID": _eval_surface_metrics(model, ctx["eval_id_rows"], device),
                "eval_STRUCT": _eval_surface_metrics(model, ctx["eval_struct_rows"], device),
                "train_family_geometry": _d3_probe_geometry(model, d3_train_rows, device),
                "unseen_family_geometry": _d3_probe_geometry(model, unseen_rows, device),
                "fra_loss_train_probe": _fra_loss_probe(model, hook, d3_train_rows, device),
                "fra_loss_unseen_probe": _fra_loss_probe(model, hook, unseen_rows, device),
            }
            print(
                f"[D4C {arm}] probe {tag}: train SMA={probes[tag]['train_ID']['cmdr_sma']:.4f} "
                f"eval_ID={probes[tag]['eval_ID']['cmdr_sma']:.6f} "
                f"FRA(train)={probes[tag]['fra_loss_train_probe']:.4f} "
                f"FRA(unseen)={probes[tag]['fra_loss_unseen_probe']:.4f}",
                flush=True,
            )

        torch.cuda.reset_peak_memory_stats(device)
        capture("T0", 0)
        started = time.time()
        for update in range(1, updates + 1):
            lr = learning_rate(update)
            for group in optimizer.param_groups:
                group["lr"] = lr
            batch_indices = stream.take(128)
            optimizer.zero_grad(set_to_none=True)
            for micro in range(GRAD_ACCUM):
                rows = [train_rows[i] for i in batch_indices[micro * MICROBATCH : (micro + 1) * MICROBATCH]]
                ids, decide, labels = collate(rows, device)
                if arm == "FC-FRA":
                    step = _fc_fra_forward(model, hook, ids, decide, labels, rows, fam_of_sid)
                    task_loss, fra, n_fam = step["task_loss"], step["fra_loss"], step["n_families"]
                    fra_family_counts.append(n_fam)
                    combined = step["combined"]
                else:
                    logits = model(ids, decide)
                    task_loss = F.cross_entropy(logits, labels, label_smoothing=0.0)
                    fra = None
                    combined = task_loss
                (combined / GRAD_ACCUM).backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            if update <= 200 or update % 10 == 0:
                entry = {
                    "update": update, "lr": lr,
                    "batch_task_ce_mean": float(task_loss),
                    "grad_norm_pre_clip": float(grad_norm),
                }
                if arm == "FC-FRA":
                    entry["batch_fra_loss"] = float(fra)
                    entry["fra_complete_families"] = n_fam
                telemetry.append(entry)
            if update % 400 == 0:
                capture(f"T{update}", update)
        wall = time.time() - started

        arms[arm] = {
            "working_tree_clean_at_start": tree_clean,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "telemetry": telemetry,
            "probes": probes,
            "wall_seconds": round(wall, 1),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "fra_contributing_family_counts": (
                {"min": min(fra_family_counts), "max": max(fra_family_counts),
                 "mean": sum(fra_family_counts) / len(fra_family_counts),
                 "microbatches": len(fra_family_counts)}
                if fra_family_counts else None
            ),
        }
        del model, optimizer
        torch.cuda.empty_cache()

    def align(arm_name: str, probe: str, which: str) -> float:
        return arms[arm_name]["probes"][probe][f"{which}_family_geometry"][
            "cross_family_displacement"
        ]["E_minus_C"]["mean_pairwise_cos"]

    train_aligned = align("FC-FRA", "T2000", "train") > 0.5
    unseen_aligned = align("FC-FRA", "T2000", "unseen") > 0.25
    eval_moving = arms["FC-FRA"]["probes"]["T2000"]["eval_ID"]["cmdr_sma"] > 0.40
    if not train_aligned:
        pattern = "FRA_OBJECTIVE_FAILED"
    elif unseen_aligned or eval_moving:
        pattern = "OBJECTIVE_PRESSURE_SUPPORTED"
    else:
        pattern = "PRESSURE_ABSORBED_BY_MEMORIZATION"

    return diag_header({
        "schema_id": "E0-Q2-DIAG-D4C-FAMILY-RESIDUAL-ALIGNMENT-v0",
        "authority": "issue #3 D4-C (Family-Residual Alignment falsifier); all other factors and v0.6 execution held; does not reopen v0.5 Q2",
        "diagnostic_only": True,
        "non_scientific": True,
        "candidate": "C0", "seed": DIAG_SEED, "updates": updates,
        "wd_scope": "exclude_norm_bias",
        "working_tree_clean_at_start": tree_clean,
        "code_git_commit": subprocess_git_head(REPO_ROOT),
        "parameter_count": arms["FC-FRA"]["parameter_count"],
        "verified_q1_input_digests": ctx["verified_digests"],
        "fra_config": fra_config,
        "d3_train_probe_family_list_sha256": d3_train_digest,
        "unseen_probe_family_list_sha256": unseen_digest,
        "pattern_criteria": {
            "train_aligned": "FC-FRA train-probe E-C mean pairwise cosine at T2000 > 0.5",
            "unseen_aligned": "FC-FRA unseen-probe E-C mean pairwise cosine at T2000 > 0.25 (D3-aligned threshold)",
            "eval_moving": "FC-FRA eval_ID SMA at T2000 > 0.40",
        },
        "arms": arms,
        "observed_pattern": pattern,
    })


# ------------------------------------------------------------------ D4-C1 ----

D4C1_PROBE_UPDATES = (200, 400, 800)
D4C1_PREFLIGHT_OUT = (
    REPO_ROOT / "local_data" / "e0_qualification_bootstrap" / "DIAG" / "d4c1_preflight.json"
)


def run_d4c1_preflight() -> dict:
    """D4-C1 preflight: rehearse the real FRA-ONLY branch (L = L_FRA applied,
    task CE observed but never applied) for one effective update on a fresh C0.
    Output is written OUTSIDE docs/ so the subsequent run records a literally
    clean tree. Non-scientific; fail closed."""
    raw_train = [json.loads(line) for line in
                 (REPO_ROOT / "local_data/e0_qualification_bootstrap/Q1/train_ID.jsonl").open(encoding="utf-8")]
    ctx = scientific_setup("C0", "exclude_norm_bias", DIAG_SEED,
                           REPO_ROOT / "local_data/e0_qualification_bootstrap/Q1", REPO_ROOT)
    device, model, optimizer = ctx["device"], ctx["model"], ctx["optimizer"]
    train_rows = ctx["train_rows"]
    fc_rows = _build_family_rows(raw_train, sorted({r["family_id"] for r in raw_train}))
    sid_to_idx = {r["sample_id"]: i for i, r in enumerate(train_rows)}
    fam_of_sid = {r["sample_id"]: r["family_id"] for r in raw_train}
    stream = FamilyCoherentStream(fc_rows, sid_to_idx, DIAG_SEED)
    hook = ClassifierDecideStateHook(model.classifier)

    micro_reports = []
    checks = {"captured_state_exists_and_attached": True}
    optimizer.zero_grad(set_to_none=True)
    batch_indices = stream.take(128)
    for micro in range(GRAD_ACCUM):
        rows = [train_rows[i] for i in batch_indices[micro * MICROBATCH : (micro + 1) * MICROBATCH]]
        ids, decide, labels = collate(rows, device)
        step = _fc_fra_forward(model, hook, ids, decide, labels, rows, fam_of_sid)
        checks["captured_state_exists_and_attached"] = (
            checks["captured_state_exists_and_attached"] and step["captured_state_attached"]
        )
        # FRA-ONLY: the applied gradient is L_FRA; task CE is observed only.
        (step["fra_loss"] / GRAD_ACCUM).backward()
        micro_reports.append({
            "micro": micro,
            "complete_families": step["n_families"],
            "task_ce_observed": float(step["task_loss"].detach()),
            "fra_loss": float(step["fra_loss"].detach()),
            "fra_finite": bool(torch.isfinite(step["fra_loss"])),
        })
    checks["min_complete_families"] = min(m["complete_families"] for m in micro_reports)
    checks["all_microbatches_ge_3_families"] = checks["min_complete_families"] >= FRA_MIN_FAMILIES
    checks["all_fra_finite"] = all(m["fra_finite"] for m in micro_reports)
    checks["backward_grads_finite"] = all(
        p.grad is not None and bool(torch.isfinite(p.grad).all())
        for p in model.parameters() if p.grad is not None
    )
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    checks["grad_norm_finite"] = bool(torch.isfinite(grad_norm))
    optimizer.step()
    checks["optimizer_step_params_finite"] = all(
        bool(torch.isfinite(p.data).all()) for p in model.parameters()
    )
    ok = all([
        checks["captured_state_exists_and_attached"],
        checks["all_microbatches_ge_3_families"],
        checks["all_fra_finite"],
        checks["backward_grads_finite"],
        checks["grad_norm_finite"],
        checks["optimizer_step_params_finite"],
    ])
    manifest = diag_header({
        "schema_id": "E0-Q2-DIAG-D4C1-PREFLIGHT-v0",
        "authority": "issue #3 D4-C1 (FRA-only self-optimization) preflight; L = L_FRA applied, task CE observed only",
        "diagnostic_only": True,
        "non_scientific": True,
        "candidate": "C0", "seed": DIAG_SEED, "effective_updates": 1,
        "wd_scope": "exclude_norm_bias",
        "working_tree_clean_at_start": subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip() == "",
        "code_git_commit": subprocess_git_head(REPO_ROOT),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "verified_q1_input_digests": ctx["verified_digests"],
        "output_location_note": "written under local_data (gitignored) so the D4-C1 run records a literally clean tree",
        "checks": checks,
        "microbatch_reports": micro_reports,
        "model_discarded": True,
        "status": "PASS" if ok else "FAIL",
    })
    del model, optimizer
    torch.cuda.empty_cache()
    return manifest


def run_d4c1(updates: int = 800) -> dict:
    """D4-C1 FRA-only self-optimization: one C0 arm, L = L_FRA only (exact D4-C
    FRA definition), 800 updates, FC stream, otherwise unchanged production
    recipe. Telemetry aggregates over all eight accumulation microbatches."""
    raw_train = [json.loads(line) for line in
                 (REPO_ROOT / "local_data/e0_qualification_bootstrap/Q1/train_ID.jsonl").open(encoding="utf-8")]
    raw_eval = [json.loads(line) for line in
                (REPO_ROOT / "local_data/e0_qualification_bootstrap/Q1/eval_ID.jsonl").open(encoding="utf-8")]
    unseen_rows, unseen_digest = _d4a_probe_rows(raw_eval)
    d3_train_rows, _d3_hold, d3_meta = d3_family_sets(raw_train)
    d3_train_digest = d3_meta["train_family_list_sha256"]

    ctx = scientific_setup("C0", "exclude_norm_bias", DIAG_SEED,
                           REPO_ROOT / "local_data/e0_qualification_bootstrap/Q1", REPO_ROOT)
    device, model, optimizer = ctx["device"], ctx["model"], ctx["optimizer"]
    train_rows = ctx["train_rows"]
    tree_clean = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip() == ""
    fc_rows = _build_family_rows(raw_train, sorted({r["family_id"] for r in raw_train}))
    sid_to_idx = {r["sample_id"]: i for i, r in enumerate(train_rows)}
    fam_of_sid = {r["sample_id"]: r["family_id"] for r in raw_train}
    stream = FamilyCoherentStream(fc_rows, sid_to_idx, DIAG_SEED)
    hook = ClassifierDecideStateHook(model.classifier)

    telemetry: list[dict] = []
    probes: dict[str, dict] = {}
    fra_family_counts: list[int] = []

    def capture(tag: str, update: int) -> None:
        probes[tag] = {
            "update": update,
            "train_ID": _eval_surface_metrics(model, train_rows, device),
            "eval_ID": _eval_surface_metrics(model, ctx["eval_id_rows"], device),
            "eval_STRUCT": _eval_surface_metrics(model, ctx["eval_struct_rows"], device),
            "train_family_geometry": _d3_probe_geometry(model, d3_train_rows, device),
            "unseen_family_geometry": _d3_probe_geometry(model, unseen_rows, device),
            "fra_loss_train_probe": _fra_loss_probe(model, hook, d3_train_rows, device),
            "fra_loss_unseen_probe": _fra_loss_probe(model, hook, unseen_rows, device),
        }
        disp = probes[tag]["train_family_geometry"]["cross_family_displacement"]
        mean_align = sum(
            disp[name]["mean_pairwise_cos"]
            for name in ("E_minus_C", "E_minus_U", "C_minus_U")
        ) / 3.0
        print(
            f"[D4C1 FRA-only] probe {tag}: trainFRA={probes[tag]['fra_loss_train_probe']:.4f} "
            f"unseenFRA={probes[tag]['fra_loss_unseen_probe']:.4f} "
            f"trainMeanAlign={mean_align:.4f} "
            f"eval_ID={probes[tag]['eval_ID']['cmdr_sma']:.6f}",
            flush=True,
        )

    torch.cuda.reset_peak_memory_stats(device)
    capture("T0", 0)
    started = time.time()
    for update in range(1, updates + 1):
        lr = learning_rate(update)
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch_indices = stream.take(128)
        optimizer.zero_grad(set_to_none=True)
        micro_fra, micro_task_observed = [], []
        for micro in range(GRAD_ACCUM):
            rows = [train_rows[i] for i in batch_indices[micro * MICROBATCH : (micro + 1) * MICROBATCH]]
            ids, decide, labels = collate(rows, device)
            step = _fc_fra_forward(model, hook, ids, decide, labels, rows, fam_of_sid)
            micro_fra.append(float(step["fra_loss"].detach()))
            micro_task_observed.append(float(step["task_loss"].detach()))
            fra_family_counts.append(step["n_families"])
            # FRA-ONLY applied gradient
            (step["fra_loss"] / GRAD_ACCUM).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        if update <= 200 or update % 10 == 0:
            telemetry.append({
                "update": update, "lr": lr,
                "update_mean_fra_loss": sum(micro_fra) / GRAD_ACCUM,
                "update_mean_task_ce_observed": sum(micro_task_observed) / GRAD_ACCUM,
                "grad_norm_pre_clip": float(grad_norm),
            })
        if update in D4C1_PROBE_UPDATES:
            capture(f"T{update}", update)
    wall = time.time() - started

    final = probes[f"T{updates}"]
    disp = final["train_family_geometry"]["cross_family_displacement"]
    mean_align = sum(
        disp[name]["mean_pairwise_cos"] for name in ("E_minus_C", "E_minus_U", "C_minus_U")
    ) / 3.0
    train_fra_loss = final["fra_loss_train_probe"]
    if mean_align > 0.50 and train_fra_loss < 0.80:
        endpoint = "FRA_SELF_OPTIMIZES"
    elif mean_align < 0.25 and train_fra_loss >= 1.00:
        endpoint = "FRA_SELF_OPTIMIZATION_FAILED"
    else:
        endpoint = "INCONCLUSIVE"

    return diag_header({
        "schema_id": "E0-Q2-DIAG-D4C1-FRA-ONLY-v0",
        "authority": "issue #3 D4-C1 (FRA-only self-optimization); all other factors and v0.6 execution held; does not reopen v0.5 Q2",
        "diagnostic_only": True,
        "non_scientific": True,
        "candidate": "C0", "seed": DIAG_SEED, "updates": updates,
        "wd_scope": "exclude_norm_bias",
        "working_tree_clean_at_start": tree_clean,
        "code_git_commit": subprocess_git_head(REPO_ROOT),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "verified_q1_input_digests": ctx["verified_digests"],
        "fra_config": {
            "applied_loss": "L = L_FRA only (task CE observed, never applied)",
            "lambda": None,
            "residual_epsilon": FRA_EPS,
            "prototype": "leave-one-family-out normalized mean of normalized residuals",
            "temperature": None,
            "min_complete_families_per_microbatch": FRA_MIN_FAMILIES,
        },
        "d3_train_probe_family_list_sha256": d3_train_digest,
        "unseen_probe_family_list_sha256": unseen_digest,
        "telemetry_semantics": "update_mean_fra_loss / update_mean_task_ce_observed are means over all 8 accumulation microbatches (corrects D4-C's last-microbatch logging)",
        "endpoint_criteria": {
            "FRA_SELF_OPTIMIZES": "T800 train mean alignment over {E-C,E-U,C-U} > 0.50 AND train FRA probe loss < 0.80",
            "FRA_SELF_OPTIMIZATION_FAILED": "mean alignment < 0.25 AND train FRA probe loss >= 1.00",
            "otherwise": "INCONCLUSIVE",
        },
        "final_train_mean_alignment": mean_align,
        "final_train_fra_probe_loss": train_fra_loss,
        "fra_contributing_family_counts": {
            "min": min(fra_family_counts), "max": max(fra_family_counts),
            "mean": sum(fra_family_counts) / len(fra_family_counts),
            "microbatches": len(fra_family_counts),
        },
        "probes": probes,
        "telemetry": telemetry,
        "observed_endpoint": endpoint,
        "wall_seconds": round(wall, 1),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["d0", "d1", "d2", "d3", "d4a", "d4b", "d4b2", "d4b3", "d4c", "d4c-preflight", "d4c1", "d4c1-preflight", "all"])
    parser.add_argument("--d0-updates", type=int, default=2000)
    parser.add_argument("--d1-updates", type=int, default=2000)
    parser.add_argument("--d3-updates", type=int, default=2000)
    parser.add_argument("--d4a-updates", type=int, default=2000)
    parser.add_argument("--d4b-updates", type=int, default=2000)
    parser.add_argument("--d4b2-updates", type=int, default=2000)
    parser.add_argument("--d4b3-updates", type=int, default=8000)
    parser.add_argument("--d4c-updates", type=int, default=2000)
    parser.add_argument("--d4c1-updates", type=int, default=800)
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
    if args.mode in ("d3", "all"):
        write_diag("d3_representation.json", run_d3(updates=args.d3_updates))
    if args.mode in ("d4a", "all"):
        write_diag("d4a_family_coherent_batching.json", run_d4a(updates=args.d4a_updates))
    if args.mode in ("d4b", "all"):
        write_diag("d4b_label_balanced_disjoint.json", run_d4b(updates=args.d4b_updates))
    if args.mode in ("d4b2", "all"):
        write_diag("d4b2_matched_permutation_control.json", run_d4b2(updates=args.d4b2_updates))
    if args.mode in ("d4b3", "all"):
        write_diag("d4b3_fc_8k.json", run_d4b3(updates=args.d4b3_updates))
    if args.mode in ("d4c-preflight", "all"):
        manifest = run_d4c_preflight()
        write_diag("d4c_preflight.json", manifest)
        raise SystemExit(0 if manifest["status"] == "PASS" else 1)
    if args.mode in ("d4c", "all"):
        write_diag("d4c_family_residual_alignment.json", run_d4c(updates=args.d4c_updates))
    if args.mode in ("d4c1-preflight", "all"):
        manifest = run_d4c1_preflight()
        D4C1_PREFLIGHT_OUT.parent.mkdir(parents=True, exist_ok=True)
        D4C1_PREFLIGHT_OUT.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {D4C1_PREFLIGHT_OUT} (outside docs/ — tree stays literally clean)")
        raise SystemExit(0 if manifest["status"] == "PASS" else 1)
    if args.mode in ("d4c1", "all"):
        write_diag("d4c1_fra_only.json", run_d4c1(updates=args.d4c1_updates))


if __name__ == "__main__":
    main()
