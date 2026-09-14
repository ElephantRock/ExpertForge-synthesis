"""Authoritative CMDR-MSEL-v0 corpus materializer — V06-MSEL-CORPUS-FREEZE-PREP-1.

Materializes the authoritative 134,000-family / 402,000-example corpus under
the namespace ExpertForge-E0-v061-msel per the contract-exact topology:
  - 32,000 train_pool / ID per depth (4 depths = 128,000 families)
  - 500 dev_ID / ID per depth
  - 500 eval_ID / ID per depth
  - 500 eval_STRUCT / STRUCT per depth (first-500 ascending-index selection,
    StructSig-disjoint from the three ID splits; NO within-split uniqueness;
    10,000-candidate fail-closed ceiling — frozen in the corrigendum)

Then freezes family/sample/render/StructSig/split digests and runs:
  - independent proof verifier
  - family-ID / rendered-string leakage checks
  - eval_STRUCT structural isolation check
  - shortcut audits (contract SS4.6: Q_shortcut ≤ 0.38)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structsig_r3
from msel_corpus import build_families
from msel_verifier import counterfactual_invariance, verify_family

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
NS = "ExpertForge-E0-v061-msel"
OUT_ROOT = REPO_ROOT / "local_data" / "e0_v061_msel"
DOCS = REPO_ROOT / "docs" / "experiments" / "e0" / "v061"

TRAIN_PER_DEPTH = 32_000
DEV_ID = 500
EVAL_ID = 500
EVAL_STRUCT = 500
EVAL_STRUCT_CEILING = 10_000


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def cjson(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fam_sig(fam) -> str:
    reps = sorted(structsig_r3.variant_canonical(v) for v in fam["variants"])
    fc = cjson({"v": "CMDR-StructSig-v1-r3-family", "orbit": reps})
    return hashlib.sha256(fc.encode("utf-8")).hexdigest()


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def tree_clean() -> bool:
    return subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip() == ""


def write_jsonl(path: Path, rows: list) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = b""
    for row in rows:
        data += cjson(row).encode("utf-8") + b"\n"
    path.write_bytes(data)
    return sha256_bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depths", nargs="+", type=int, default=[1, 2, 3, 4])
    args = parser.parse_args()

    started = time.time()
    all_samples = {}  # split -> flat list of variant dicts
    all_family_ids = {}  # split -> set of family_ids
    all_renders = {}  # split -> set of render_sha256
    all_structsigs = {}  # split -> set of struct sigs
    manifest = {
        "schema_id": "CMDR-MSEL-MANIFEST-v0",
        "namespace": NS,
        "generator": "msel_corpus.py (byte-identical to cfe0840)",
        "canonicalizer": "structsig_r3.py (BLISS via igraph 1.0.0)",
        "code_git_commit": git_head(),
        "working_tree_clean_at_start": tree_clean(),
        "depths": args.depths,
        "splits": {},
        "corpus_invariants": {},
    }
    errors = []

    for depth in args.depths:
        depth_t0 = time.time()
        print(f"=== depth {depth} ===", flush=True)

        # --- train_pool (ID, 32k) ---
        t0 = time.time()
        train = build_families(NS, "train_pool", "ID", depth, range(TRAIN_PER_DEPTH))
        train_variants = [v for fam in train for v in fam["variants"]]
        train_sigs = {fam_sig(f) for f in train}
        print(f"  train_pool: {len(train)} families, {len(train_sigs)} unique sigs ({time.time()-t0:.0f}s)", flush=True)
        manifest["splits"][f"train_pool|d{depth}"] = {
            "surface": "ID", "families": len(train), "examples": len(train_variants),
            "unique_structsigs": len(train_sigs),
        }

        # --- dev_ID + eval_ID ---
        dev = build_families(NS, "dev_ID", "ID", depth, range(DEV_ID))
        eval_id = build_families(NS, "eval_ID", "ID", depth, range(EVAL_ID))
        id_burn = set(train_sigs)
        for fams in (dev, eval_id):
            for f in fams:
                id_burn.add(fam_sig(f))
        dev_variants = [v for f in dev for v in f["variants"]]
        eval_id_variants = [v for f in eval_id for v in f["variants"]]
        print(f"  dev_ID: {len(dev)} | eval_ID: {len(eval_id)} | ID burn: {len(id_burn)}", flush=True)
        manifest["splits"][f"dev_ID|d{depth}"] = {"surface": "ID", "families": len(dev), "examples": len(dev_variants)}
        manifest["splits"][f"eval_ID|d{depth}"] = {"surface": "ID", "families": len(eval_id), "examples": len(eval_id_variants)}

        # --- eval_STRUCT (first-500 ascending, no within-split uniqueness, 10k ceiling) ---
        t0 = time.time()
        es_candidates = build_families(NS, "eval_STRUCT", "STRUCT", depth, range(EVAL_STRUCT_CEILING))
        accepted = []
        ceiling_hit = False
        for fam in es_candidates:  # already ascending by family_index from build_families
            sig = fam_sig(fam)
            if sig in id_burn:
                continue
            accepted.append(fam)
            if len(accepted) >= EVAL_STRUCT:
                break
        if len(accepted) < EVAL_STRUCT:
            ceiling_hit = True
            errors.append(f"eval_STRUCT d{depth}: only {len(accepted)}/{EVAL_STRUCT} filled within 10k ceiling")
        es_variants = [v for f in accepted for v in f["variants"]]
        print(f"  eval_STRUCT: {len(accepted)}/{EVAL_STRUCT} selected from ceiling scan "
              f"({time.time()-t0:.0f}s){' [CEILING HIT]' if ceiling_hit else ''}", flush=True)
        manifest["splits"][f"eval_STRUCT|d{depth}"] = {
            "surface": "STRUCT", "families": len(accepted), "examples": len(es_variants),
            "selection_algorithm": "first-500 ascending index, StructSig-disjoint from ID burn, no within-split uniqueness",
            "ceiling": EVAL_STRUCT_CEILING, "ceiling_hit": ceiling_hit,
        }

        # Store per-depth
        for split, fams in [("train_pool", train), ("dev_ID", dev), ("eval_ID", eval_id), ("eval_STRUCT", accepted)]:
            key = f"{split}_d{depth}"
            variants = [v for f in fams for v in f["variants"]]
            all_samples[key] = variants
            all_family_ids.setdefault(split, set()).update(f["family_id"] for f in fams)
            all_renders.setdefault(split, set()).update(v["render_sha256"] for v in variants)
            all_structsigs.setdefault(split, set()).update(fam_sig(f) for f in fams)

        # Verify all families
        t0 = time.time()
        verr = 0
        for fams in (train, dev, eval_id, accepted):
            for fam in fams:
                verr += len(verify_family(fam)) + len(counterfactual_invariance(fam))
        if verr:
            errors.append(f"depth {depth}: {verr} verifier/invariance errors")
        print(f"  verification: {verr} errors ({time.time()-t0:.0f}s)", flush=True)
        print(f"  depth {depth} completed in {(time.time()-depth_t0)/60:.1f} min", flush=True)

    # --- Write JSONL files ---
    print("=== writing corpus files ===", flush=True)
    split_digests = {}
    for key, variants in all_samples.items():
        path = OUT_ROOT / f"{key}.jsonl"
        digest = write_jsonl(path, variants)
        split_digests[key] = digest
        print(f"  {key}: {len(variants)} examples, sha256={digest[:16]}...", flush=True)
    manifest["split_digests"] = split_digests

    # --- Corpus-level invariants ---
    total_examples = sum(len(v) for v in all_samples.values())
    total_families = sum(len(s) for s in all_family_ids.values())

    # family-ID leakage: pairwise 0 across splits
    fam_leak = {}
    splits = sorted(all_family_ids.keys())
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            overlap = all_family_ids[splits[i]] & all_family_ids[splits[j]]
            fam_leak[f"{splits[i]}|{splits[j]}"] = len(overlap)
            if overlap:
                errors.append(f"family-ID leakage {splits[i]}↔{splits[j]}: {len(overlap)}")

    # rendered-string leakage: pairwise 0 across splits
    render_leak = {}
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            overlap = all_renders[splits[i]] & all_renders[splits[j]]
            render_leak[f"{splits[i]}|{splits[j]}"] = len(overlap)
            if overlap:
                errors.append(f"rendered-string leakage {splits[i]}↔{splits[j]}: {len(overlap)}")

    # eval_STRUCT structural isolation: disjoint from other three splits
    id_sig_union = all_structsigs.get("train_pool", set()) | all_structsigs.get("dev_ID", set()) | all_structsigs.get("eval_ID", set())
    es_overlap = all_structsigs.get("eval_STRUCT", set()) & id_sig_union
    es_isolated = len(es_overlap) == 0
    if not es_isolated:
        errors.append(f"eval_STRUCT structural isolation violated: {len(es_overlap)} overlapping signatures")

    manifest["corpus_invariants"] = {
        "total_families": total_families,
        "total_examples": total_examples,
        "family_id_pairwise_overlap": fam_leak,
        "rendered_string_pairwise_overlap": render_leak,
        "eval_struct_structsig_overlap_with_id_splits": len(es_overlap),
        "eval_struct_isolated": es_isolated,
    }
    print(f"  invariants: {total_families} families, {total_examples} examples", flush=True)
    print(f"  family-ID leakage: {fam_leak}", flush=True)
    print(f"  eval_STRUCT isolation: {'PASS' if es_isolated else 'FAIL'} ({len(es_overlap)} overlaps)", flush=True)

    # --- Shortcut audits (SS4.6) ---
    print("=== shortcut audits ===", flush=True)
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "e0"))
    from q1_verify_and_audit import nuisance_features, macro_cell_accuracy
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train_flat = all_samples.get("train_pool_d1", []) + all_samples.get("train_pool_d2", []) \
        + all_samples.get("train_pool_d3", []) + all_samples.get("train_pool_d4", [])
    eval_id_flat = all_samples.get("eval_ID_d1", []) + all_samples.get("eval_ID_d2", []) \
        + all_samples.get("eval_ID_d3", []) + all_samples.get("eval_ID_d4", [])
    eval_struct_flat = all_samples.get("eval_STRUCT_d1", []) + all_samples.get("eval_STRUCT_d2", []) \
        + all_samples.get("eval_STRUCT_d3", []) + all_samples.get("eval_STRUCT_d4", [])

    shortcut_results = {}
    x_train, y_train = nuisance_features(train_flat)
    nuisance_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=0))
    nuisance_model.fit(x_train, y_train)
    for name, rows in [("eval_ID", eval_id_flat), ("eval_STRUCT", eval_struct_flat)]:
        x, y = nuisance_features(rows)
        pred = nuisance_model.predict(x)
        y_arr = np.asarray(y)
        d_arr = np.asarray([r["reasoning_depth_stratum"] for r in rows])
        score = macro_cell_accuracy(y_arr, pred, d_arr)
        shortcut_results[f"nuisance|{name}"] = round(score, 6)
        if score > 0.38:
            errors.append(f"shortcut audit nuisance|{name}: {score:.4f} > 0.38")

    vectorizer = CountVectorizer(analyzer="word", tokenizer=str.split, token_pattern=None,
                                  lowercase=False, ngram_range=(1, 2), binary=False)
    x_lex = vectorizer.fit_transform([r["rendered"] for r in train_flat])
    lexical = LogisticRegression(max_iter=100, random_state=0, solver="lbfgs", tol=1e-8)
    lexical.fit(x_lex, y_train)
    for name, rows in [("eval_ID", eval_id_flat), ("eval_STRUCT", eval_struct_flat)]:
        x = vectorizer.transform([r["rendered"] for r in rows])
        y = np.asarray([r["gold_label"] for r in rows])
        d = np.asarray([r["reasoning_depth_stratum"] for r in rows])
        pred = lexical.predict(x)
        score = macro_cell_accuracy(y, pred, d)
        shortcut_results[f"lexical|{name}"] = round(score, 6)
        if score > 0.38:
            errors.append(f"shortcut audit lexical|{name}: {score:.4f} > 0.38")

    manifest["shortcut_audits"] = {
        "gate": "Q_shortcut <= 0.38",
        "scores": shortcut_results,
        "status": "PASS" if all(v <= 0.38 for v in shortcut_results.values()) else "FAIL",
    }
    print(f"  shortcut scores: {shortcut_results}", flush=True)

    # --- Final status ---
    manifest["status"] = "PASS" if not errors else "FAIL"
    manifest["errors"] = errors
    manifest["wall_seconds"] = round(time.time() - started, 1)

    # Write manifest
    manifest_path = OUT_ROOT / "CMDR-MSEL-MANIFEST.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nManifest written: {manifest_path}", flush=True)
    print(f"STATUS: {manifest['status']}", flush=True)
    if errors:
        print(f"ERRORS: {errors}", flush=True)

    # Copy to docs for evidence
    docs_manifest = DOCS / "MSEL_MANIFEST.json"
    docs_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")

    raise SystemExit(0 if manifest["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
