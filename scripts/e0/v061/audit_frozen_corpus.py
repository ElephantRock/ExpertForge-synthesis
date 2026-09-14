"""Read-only frozen-corpus auditor + burn index + identity roots + R1/R4/R16 derivation.

V06-MSEL-POSTMATERIALIZATION-AUDIT-1. This script NEVER modifies the corpus
JSONL files. It:
  1. SHA-256-verifies all 16 JSONL files against MSEL_MANIFEST.json (hard stop on mismatch)
  2. Reruns independent proof/counterfactual verifier over all 134,000 families
  3. Reruns all 6 family-ID and rendered-string split comparisons
  4. Reruns BLISS eval_STRUCT structural isolation
  5. Reruns both shortcut models with the CORRECT macro_cell_accuracy(y, pred, rows) API
  6. Creates a frozen burn index (family_id + StructSig per family)
  7. Freezes semantic identity roots per split/depth
  8. Derives R1/R4/R16 membership by the contract ranking rule
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

import structsig_r3
from msel_corpus import LABELS
from msel_verifier import independent_verify

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CORPUS = REPO_ROOT / "local_data" / "e0_v061_msel"
DOCS = REPO_ROOT / "docs" / "experiments" / "e0" / "v061"
MANIFEST_PATH = DOCS / "MSEL_MANIFEST.json"
OUTPUT_PATH = DOCS / "MSEL_POSTMATERIALIZATION_AUDIT.json"
BURN_INDEX_PATH = REPO_ROOT / "local_data" / "e0_v061_msel" / "MSEL_BURN_INDEX.json"
RUNG_PATH = REPO_ROOT / "local_data" / "e0_v061_msel" / "MSEL_RUNGS.json"
SPLITS = ("train_pool", "dev_ID", "eval_ID", "eval_STRUCT")
DEPTHS = (1, 2, 3, 4)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cjson(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def fam_sig_from_variants(variants: list) -> str:
    reps = sorted(structsig_r3.variant_canonical(v) for v in variants)
    fc = cjson({"v": "CMDR-StructSig-v1-r3-family", "orbit": reps})
    return hashlib.sha256(fc.encode("utf-8")).hexdigest()


def ordered_root(values: list) -> str:
    """SHA-256 over the sorted list of values (deterministic multiset root)."""
    blob = cjson(sorted(values))
    return sha256_bytes(blob.encode("utf-8"))


def main() -> None:
    started = time.time()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    errors = []
    report = {
        "schema_id": "E0-V061-MSEL-POSTMATERIALIZATION-AUDIT-v0",
        "authority": "V06-MSEL-POSTMATERIALIZATION-AUDIT-1 (read-only; corpus JSONL files never modified)",
        "code_git_commit": git_head(),
        "manifest_reference": manifest["code_git_commit"],
    }

    # --- 1. SHA-256 verification against manifest ---
    print("=== SHA-256 verification ===", flush=True)
    digest_checks = {}
    for split in SPLITS:
        for dep in DEPTHS:
            key = f"{split}_d{dep}"
            p = CORPUS / f"{key}.jsonl"
            actual = sha256_file(p)
            expected = manifest["splits"][key]["sha256"]
            match = actual == expected
            digest_checks[key] = {"match": match, "sha256": actual}
            if not match:
                errors.append(f"SHA-256 mismatch for {key}: {actual} != {expected}")
    all_match = all(v["match"] for v in digest_checks.values())
    print(f"  all 16 files match: {all_match}", flush=True)
    if not all_match:
        report["status"] = "FAIL"
        report["errors"] = errors
        OUTPUT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
        raise SystemExit("HARD STOP: corpus digest mismatch — corpus may have been mutated")

    # --- Load corpus into memory ---
    print("=== loading corpus ===", flush=True)
    split_data = {}  # key -> list of variant dicts
    for split in SPLITS:
        for dep in DEPTHS:
            key = f"{split}_d{dep}"
            rows = []
            with open(CORPUS / f"{key}.jsonl", encoding="utf-8") as f:
                for line in f:
                    rows.append(json.loads(line))
            split_data[key] = rows
    total_examples = sum(len(v) for v in split_data.values())
    print(f"  loaded {total_examples} examples", flush=True)

    # --- 2. Independent proof/counterfactual verifier ---
    print("=== independent verifier ===", flush=True)
    verifier_errors = 0
    # Group variants by family for verification
    from collections import defaultdict
    fam_variants = defaultdict(list)
    fam_meta = {}
    for key, rows in split_data.items():
        for row in rows:
            fam_variants[row["family_id"]].append(row)
            fam_meta[row["family_id"]] = {
                "split": row["split"], "surface": row["surface"],
                "depth": row["reasoning_depth_stratum"],
            }
    for fid, variants in fam_variants.items():
        if len(variants) != 3:
            verifier_errors += 1
            errors.append(f"family {fid}: {len(variants)} variants (expected 3)")
            continue
        labels = sorted(v["gold_label"] for v in variants)
        if labels != sorted(LABELS):
            verifier_errors += 1
            errors.append(f"family {fid}: labels {labels}")
            continue
        for v in variants:
            label, proof_depth, q_depth, opp_depth = independent_verify(v)
            if label != v["gold_label"]:
                verifier_errors += 1
                errors.append(f"family {fid}: label mismatch {label} != {v['gold_label']}")
            if label != "UNKNOWN" and proof_depth != v["reasoning_depth_stratum"]:
                verifier_errors += 1
                errors.append(f"family {fid}: depth mismatch {proof_depth} != {v['reasoning_depth_stratum']}")
        # counterfactual invariance: unigram multiset
        from collections import Counter as C
        unigrams = [C(v["rendered"].split()) for v in variants]
        if not (unigrams[0] == unigrams[1] == unigrams[2]):
            verifier_errors += 1
            errors.append(f"family {fid}: unigram multiset mismatch")
    print(f"  verifier errors: {verifier_errors} (over {len(fam_variants)} families)", flush=True)

    # --- 3. Family-ID and rendered-string leakage ---
    print("=== leakage checks ===", flush=True)
    split_fam_ids = {}
    split_renders = {}
    for split in SPLITS:
        fids = set()
        renders = set()
        for dep in DEPTHS:
            for row in split_data[f"{split}_d{dep}"]:
                fids.add(row["family_id"])
                renders.add(row["render_sha256"])
        split_fam_ids[split] = fids
        split_renders[split] = renders
    fam_leak = {}
    render_leak = {}
    for i in range(len(SPLITS)):
        for j in range(i + 1, len(SPLITS)):
            a, b = SPLITS[i], SPLITS[j]
            fo = len(split_fam_ids[a] & split_fam_ids[b])
            ro = len(split_renders[a] & split_renders[b])
            fam_leak[f"{a}|{b}"] = fo
            render_leak[f"{a}|{b}"] = ro
            if fo:
                errors.append(f"family-ID leakage {a}|{b}: {fo}")
            if ro:
                errors.append(f"rendered-string leakage {a}|{b}: {ro}")
    print(f"  family-ID leakage: {fam_leak}", flush=True)
    print(f"  rendered-string leakage: {render_leak}", flush=True)

    # --- 4. BLISS eval_STRUCT isolation ---
    print("=== eval_STRUCT isolation ===", flush=True)
    fam_sigs = {}
    for fid, variants in fam_variants.items():
        fam_sigs[fid] = fam_sig_from_variants(variants)
    split_sigs = {}
    for split in SPLITS:
        split_sigs[split] = {fam_sigs[fid] for fid in split_fam_ids[split]}
    id_union = split_sigs["train_pool"] | split_sigs["dev_ID"] | split_sigs["eval_ID"]
    es_overlap = len(split_sigs["eval_STRUCT"] & id_union)
    es_isolated = es_overlap == 0
    print(f"  eval_STRUCT overlap with ID splits: {es_overlap} ({'PASS' if es_isolated else 'FAIL'})", flush=True)
    if not es_isolated:
        errors.append(f"eval_STRUCT isolation violated: {es_overlap}")

    # --- 5. Shortcut audits (correct API) ---
    print("=== shortcut audits ===", flush=True)
    from q1_verify_and_audit import nuisance_features, macro_cell_accuracy
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import CountVectorizer

    train_flat = split_data["train_pool_d1"] + split_data["train_pool_d2"] + \
                split_data["train_pool_d3"] + split_data["train_pool_d4"]
    eval_id_flat = split_data["eval_ID_d1"] + split_data["eval_ID_d2"] + \
                   split_data["eval_ID_d3"] + split_data["eval_ID_d4"]
    eval_struct_flat = split_data["eval_STRUCT_d1"] + split_data["eval_STRUCT_d2"] + \
                       split_data["eval_STRUCT_d3"] + split_data["eval_STRUCT_d4"]

    shortcut_scores = {}
    x_train, y_train = nuisance_features(train_flat)
    nuisance_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=0))
    nuisance_model.fit(x_train, y_train)
    for name, rows in [("eval_ID", eval_id_flat), ("eval_STRUCT", eval_struct_flat)]:
        x, y = nuisance_features(rows)
        pred = nuisance_model.predict(x)
        score = macro_cell_accuracy(np.asarray(y), pred, rows)
        shortcut_scores[f"nuisance|{name}"] = round(score, 6)
        if score > 0.38:
            errors.append(f"shortcut nuisance|{name}: {score:.4f} > 0.38")

    vectorizer = CountVectorizer(analyzer="word", tokenizer=str.split, token_pattern=None,
                                  lowercase=False, ngram_range=(1, 2), binary=False)
    x_lex = vectorizer.fit_transform([r["rendered"] for r in train_flat])
    lexical = LogisticRegression(max_iter=100, random_state=0, solver="lbfgs", tol=1e-8)
    lexical.fit(x_lex, y_train)
    for name, rows in [("eval_ID", eval_id_flat), ("eval_STRUCT", eval_struct_flat)]:
        x = vectorizer.transform([r["rendered"] for r in rows])
        pred = lexical.predict(x)
        y = np.asarray([r["gold_label"] for r in rows])
        score = macro_cell_accuracy(y, pred, rows)
        shortcut_scores[f"lexical|{name}"] = round(score, 6)
        if score > 0.38:
            errors.append(f"shortcut lexical|{name}: {score:.4f} > 0.38")
    print(f"  scores: {shortcut_scores}", flush=True)
    shortcut_pass = all(v <= 0.38 for v in shortcut_scores.values())

    # --- 6. Frozen burn index ---
    print("=== burn index ===", flush=True)
    burn_entries = sorted(fam_sigs.keys())
    burn_index = {
        "schema_id": "E0-V061-MSEL-BURN-INDEX-v0",
        "namespace": "ExpertForge-E0-v061-msel",
        "family_count": len(burn_entries),
        "entries": {fid: fam_sigs[fid] for fid in burn_entries},
    }
    burn_index_blob = cjson({fid: fam_sigs[fid] for fid in burn_entries})
    burn_index_sha = sha256_bytes(burn_index_blob.encode("utf-8"))
    BURN_INDEX_PATH.write_text(json.dumps(burn_index, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"  {len(burn_entries)} families indexed | content SHA-256: {burn_index_sha[:16]}...", flush=True)

    # --- 7. Semantic identity roots ---
    print("=== semantic identity roots ===", flush=True)
    identity_roots = {}
    for split in SPLITS:
        for dep in DEPTHS:
            key = f"{split}_d{dep}"
            rows = split_data[key]
            fids = sorted({r["family_id"] for r in rows})
            sids = sorted(r["sample_id"] for r in rows)
            renders = sorted(r["render_sha256"] for r in rows)
            sigs = sorted({fam_sigs[r["family_id"]] for r in rows})
            identity_roots[key] = {
                "families": len(fids),
                "examples": len(rows),
                "family_id_root": ordered_root(fids),
                "sample_id_root": ordered_root(sids),
                "render_sha256_root": ordered_root(renders),
                "structsig_root": ordered_root(sigs),
                "unique_structsigs": len(sigs),
            }
        # per-split aggregate
        split_fids = sorted(split_fam_ids[split])
        split_sigs = sorted(split_sigs[split])
        identity_roots[f"{split}|aggregate"] = {
            "family_id_root": ordered_root(split_fids),
            "structsig_root": ordered_root(split_sigs),
            "unique_structsigs": len(split_sigs),
        }
    # exact union count (for corrigendum correction)
    exact_union = len(set().union(*split_sigs.values()))
    identity_roots["exact_union"] = {
        "total_unique_structsigs_across_all_splits": exact_union,
        "note": "exact authoritative union count; replaces the corrigendum's approximate per-split-unique sums",
    }
    print(f"  exact union: {exact_union} unique StructSigs", flush=True)

    # --- 8. R1/R4/R16 membership ---
    print("=== R1/R4/R16 derivation ===", flush=True)
    # Collect train-pool families per depth
    train_families_by_depth = defaultdict(dict)  # depth -> fid -> sig
    for dep in DEPTHS:
        for row in split_data[f"train_pool_d{dep}"]:
            train_families_by_depth[dep][row["family_id"]] = fam_sigs[row["family_id"]]

    rungs = {}
    rung_families = {}
    for dep in DEPTHS:
        depth_fams = train_families_by_depth[dep]
        # rank by SHA256("ExpertForge-E0-v061-msel-rank|" || family_id)
        ranked = sorted(
            depth_fams.keys(),
            key=lambda fid: hashlib.sha256(f"ExpertForge-E0-v061-msel-rank|{fid}".encode()).hexdigest()
        )
        r1 = ranked[:2000]
        r4 = ranked[:8000]
        r16 = ranked[:32000]
        # verify nesting
        assert set(r1) <= set(r4), f"d{dep}: R1 not subset of R4"
        assert set(r4) <= set(r16), f"d{dep}: R4 not subset of R16"
        assert len(r1) == 2000 and len(r4) == 8000 and len(r16) == min(32000, len(ranked)), \
            f"d{dep}: rung sizes {len(r1)}/{len(r4)}/{len(r16)}"
        rungs[f"d{dep}"] = {
            "R1_family_ids_root": ordered_root(r1),
            "R4_family_ids_root": ordered_root(r4),
            "R16_family_ids_root": ordered_root(r16),
            "R1_count": len(r1), "R4_count": len(r4), "R16_count": len(r16),
            "R1_subset_R4": True, "R4_subset_R16": True,
        }
        rung_families[f"d{dep}"] = {"R1": r1, "R4": r4, "R16": r16}
        print(f"  d{dep}: R1={len(r1)} R4={len(r4)} R16={len(r16)} | nesting verified", flush=True)

    # Write rungs
    rung_artifact = {
        "schema_id": "E0-V061-MSEL-RUNGS-v0",
        "ranking_rule": 'SHA256("ExpertForge-E0-v061-msel-rank|" || family_id) ascending within depth',
        "rungs": rungs,
        "family_ids": rung_families,
    }
    RUNG_PATH.write_text(json.dumps(rung_artifact, indent=2) + "\n", encoding="utf-8", newline="\n")
    rung_sha = sha256_file(RUNG_PATH)
    print(f"  rungs artifact SHA-256: {rung_sha[:16]}...", flush=True)

    # --- Final report ---
    report.update({
        "digest_verification": {"all_16_match": all_match, "details": digest_checks},
        "verifier": {"errors": verifier_errors, "families_checked": len(fam_variants)},
        "family_id_leakage": fam_leak,
        "rendered_string_leakage": render_leak,
        "eval_struct_isolation": {
            "overlap_with_id_splits": es_overlap,
            "status": "PASS" if es_isolated else "FAIL",
        },
        "shortcut_audits": {
            "gate": "Q_shortcut <= 0.38",
            "scores": shortcut_scores,
            "api": "macro_cell_accuracy(y, pred, rows) — corrected invocation",
            "status": "PASS" if shortcut_pass else "FAIL",
        },
        "burn_index": {
            "path": str(BURN_INDEX_PATH.relative_to(REPO_ROOT)),
            "family_count": len(burn_entries),
            "content_sha256": burn_index_sha,
        },
        "identity_roots": identity_roots,
        "rungs": {
            "path": str(RUNG_PATH.relative_to(REPO_ROOT)),
            "sha256": rung_sha,
            "per_depth": rungs,
        },
        "total_examples": total_examples,
        "total_families": len(fam_variants),
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
        "wall_seconds": round(time.time() - started, 1),
    })

    OUTPUT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\n=== STATUS: {report['status']} ===", flush=True)
    if errors:
        print(f"ERRORS: {errors[:5]}", flush=True)
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
