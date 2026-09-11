"""Structural-support preflight r1 (V06-STRUCTSIG-REMEDIATION-1).

Replaces run_structural_preflight.py's evidence. Corrections per the ruling:
  - uses structsig_r1 (exact IR canonicalization; multiset incidence;
    full-length internal hashes; family signature over the sorted triple of
    canonical variant representations)
  - fresh burned namespace ExpertForge-E0-v061-msel-preflight-r1
  - exact discrimination tests: nuisance renaming, fact/rule/premise order
    permutation, adversarial multiplicity, orbit-shared-member, and
    canonical-form string-equality verification of every repeated signature
    group
  - support measurement is DESCRIPTIVE ONLY: unique counts/multiplicities at
    frozen prefixes, singleton/doubleton counts, Chao1 and occupancy
    saturation estimates with formulas and assumptions recorded. "Collision
    observed" is NOT a failure condition.
Fail-closed conditions retained: verifier disagreement, regeneration
nondeterminism, renaming/permutation instability, discrimination-test
failures, repeated-group canonical mismatch, IR branch cap exceeded.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structsig_r1
from msel_corpus import apply_renaming, build_families
from msel_verifier import counterfactual_invariance, verify_family

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EVIDENCE_PATH = REPO_ROOT / "docs" / "experiments" / "e0" / "v061" / "STRUCTURAL_PREFLIGHT_R1.json"
FROZEN_PREFIXES = (250, 500, 1000, 2000)


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def tree_clean() -> bool:
    return subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip() == ""


def permutation_stability_test(family: dict) -> bool:
    """Fact order, rule order, and premise-within-rule order are slot-order
    nuisance: permuting them must not change any variant digest."""
    rng = random.Random(777)
    for v in family["variants"]:
        w = copy.deepcopy(v)
        rng.shuffle(w["facts"])
        rng.shuffle(w["rules"])
        for r in w["rules"]:
            rng.shuffle(r["premises"])
        if structsig_r1.variant_digest(w) != structsig_r1.variant_digest(v):
            return False
    return True


def renaming_stability_test(family: dict) -> bool:
    rng = random.Random(12345)
    preds, ents = set(), set()
    for v in family["variants"]:
        for f in v["facts"]:
            preds.add(f["pred"]); ents.add(f["term"])
        for r in v["rules"]:
            for p in r["premises"] + [r["conclusion"]]:
                preds.add(p["pred"])
                if p["term"] != "x":
                    ents.add(p["term"])
    new_preds = [f"ZP{i:03d}" for i in range(len(preds))]
    new_ents = [f"ZE{i:03d}" for i in range(len(ents))]
    rng.shuffle(new_preds); rng.shuffle(new_ents)
    pred_map = dict(zip(sorted(preds), new_preds))
    ent_map = dict(zip(sorted(ents), new_ents))
    renamed = [apply_renaming(v, pred_map, ent_map) for v in family["variants"]]
    return structsig_r1.family_signature(renamed) == structsig_r1.family_signature(family["variants"])


def adversarial_multiplicity_test() -> dict:
    """Two hand-crafted variants with identical node counts but different
    incidence multiplicity of P1 (two facts vs one fact + one rule premise)
    must receive DIFFERENT canonical forms."""
    base = {
        "reasoning_depth_stratum": 1,
        "surface": "ID",
        "facts": None,
        "rules": None,
        "query": {"sign": "+", "pred": "P1", "term": "E1"},
    }
    a = copy.deepcopy(base)
    a["facts"] = [
        {"sign": "+", "pred": "P1", "term": "E1"},
        {"sign": "+", "pred": "P1", "term": "E2"},
    ]
    a["rules"] = [{"premises": [{"sign": "+", "pred": "P3", "term": "x"}],
                   "conclusion": {"sign": "-", "pred": "P4", "term": "x"}}]
    b = copy.deepcopy(base)
    b["facts"] = [
        {"sign": "+", "pred": "P1", "term": "E1"},
        {"sign": "+", "pred": "P3", "term": "E2"},
    ]
    b["rules"] = [{"premises": [{"sign": "+", "pred": "P1", "term": "x"}],
                   "conclusion": {"sign": "-", "pred": "P4", "term": "x"}}]
    ca, cb = structsig_r1.variant_canonical(a), structsig_r1.variant_canonical(b)
    return {"distinct": ca != cb, "a_digest": structsig_r1.variant_digest(a)[:16],
            "b_digest": structsig_r1.variant_digest(b)[:16]}


def orbit_shared_member_test(family: dict) -> dict:
    """Two families sharing one orbit member but differing in the others must
    receive DIFFERENT family signatures (the r1 defect). Build B = [A0, X, Y]
    where X/Y are mutations of A1/A2 (flip one distractor-rule conclusion
    sign)."""
    a_sig = structsig_r1.family_signature(family["variants"])

    def mutate(v):
        w = copy.deepcopy(v)
        for r in w["rules"]:
            if r["conclusion"]["term"] == "x" and r["conclusion"]["sign"] == "+":
                r["conclusion"]["sign"] = "-"
                return w
        return None

    x = mutate(family["variants"][1])
    y = mutate(family["variants"][2])
    if x is None or y is None:
        return {"shared_member_confirmed": False, "orbit_overlap_size": 0,
                "family_signatures_differ": False, "mutation_applied": False}
    b_variants = [copy.deepcopy(family["variants"][0]), x, y]
    shared_rep = (
        structsig_r1.variant_digest(family["variants"][0])
        == structsig_r1.variant_digest(b_variants[0])
    )
    orbits_overlap = set(structsig_r1.variant_digest(v) for v in family["variants"]) & set(
        structsig_r1.variant_digest(v) for v in b_variants
    )
    b_sig = structsig_r1.family_signature(b_variants)
    return {
        "shared_member_confirmed": shared_rep,
        "orbit_overlap_size": len(orbits_overlap),
        "family_signatures_differ": a_sig != b_sig,
        "mutation_applied": True,
    }


def occupancy_fit(n: int, u: int):
    """Solve u = M(1 - e^{-n/M}) for M by bisection (descriptive only).
    Assumes n independent draws with replacement from M equally likely
    signatures — a strong assumption recorded here, not defended."""
    if u >= n:
        return None
    lo, hi = u + 1.0, max(4.0 * n, u * 10.0)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid * (1.0 - math.exp(-n / mid)) < u:
            lo = mid
        else:
            hi = mid
    return round(0.5 * (lo + hi))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families-per-depth", type=int, default=2000)
    parser.add_argument("--surfaces", nargs="+", default=["ID", "STRUCT"])
    parser.add_argument("--pilot-namespace", default="ExpertForge-E0-v061-msel-preflight-r1")
    args = parser.parse_args()

    started = time.time()
    n = args.families_per_depth
    per_cell: dict[str, dict] = {}
    cross_cell: dict[str, list[str]] = {}
    failures: list[str] = []
    determinism_mismatches = 0
    renaming_results, permutation_results = [], []
    orbit_results = []
    group_canonical_mismatches = 0
    orbit_overlap_across_signatures = 0

    multiplicity_test = adversarial_multiplicity_test()
    if not multiplicity_test["distinct"]:
        failures.append("adversarial multiplicity test failed to discriminate")

    for surface in args.surfaces:
        for depth in range(1, 5):
            key = f"{surface}|d{depth}"
            families = build_families(args.pilot_namespace, "pilot", surface, depth, range(n))
            regenerated = build_families(
                args.pilot_namespace, "pilot", surface, depth, range(min(50, n))
            )
            for a, b in zip(families[: len(regenerated)], regenerated):
                if a["family_id"] != b["family_id"] or {
                    v["render_sha256"] for v in a["variants"]
                } != {v["render_sha256"] for v in b["variants"]}:
                    determinism_mismatches += 1

            verifier_errors = []
            sigs = []
            fam_canonicals: dict[str, list[str]] = {}
            variant_digest_seen: set[str] = set()
            try:
                for fam in families:
                    verifier_errors += verify_family(fam)
                    verifier_errors += counterfactual_invariance(fam)
                    fc = structsig_r1.family_canonical(fam["variants"])
                    sig = structsig_r1.family_signature(fam["variants"])
                    sigs.append(sig)
                    fam_canonicals.setdefault(sig, []).append(fc)
                    for v in fam["variants"]:
                        variant_digest_seen.add(structsig_r1.variant_digest(v))
                    cross_cell.setdefault(sig, []).append(key)
            except structsig_r1._BranchCapExceeded as exc:
                failures.append(f"IR branch cap exceeded in {key}: {exc}")

            # repeated-signature groups: every member must have identical FULL
            # canonical serialization (string equality, not digest equality)
            for sig, canons in fam_canonicals.items():
                if len(canons) > 1 and len(set(canons)) != 1:
                    group_canonical_mismatches += 1

            # orbit overlap across DIFFERENT family signatures (informational;
            # demonstrates why the r1 min() defect mattered)
            reps_per_sig = {}
            for fam in families:
                reps_per_sig.setdefault(
                    structsig_r1.family_signature(fam["variants"]), set()
                ).add(structsig_r1.variant_canonical(fam["variants"][0]))
            # informational: families sharing signature already collapsed; skip

            renaming_results += [renaming_stability_test(f) for f in families[:3]]
            permutation_results += [permutation_stability_test(f) for f in families[:3]]
            orbit_results += [orbit_shared_member_test(f) for f in families[:3]]

            sig_counts = Counter(sigs)
            prefix_stats = {}
            for k in FROZEN_PREFIXES:
                if k > n:
                    continue
                sub = Counter(sigs[:k])
                f1 = sum(1 for c in sub.values() if c == 1)
                f2 = sum(1 for c in sub.values() if c == 2)
                chao1 = (f1 * f1 / (2 * f2)) if f2 > 0 else None
                prefix_stats[f"n={k}"] = {
                    "unique": len(sub),
                    "singletons": f1,
                    "doubletons": f2,
                    "chao1_richness_estimate": round(chao1, 1) if chao1 else None,
                    "occupancy_fit_M": occupancy_fit(k, len(sub)),
                }
            per_cell[key] = {
                "attempted": n,
                "accepted_complete": len(families),
                "verifier_errors": len(verifier_errors),
                "unique_family_signatures": len(sig_counts),
                "multiplicity_distribution": dict(Counter(sig_counts.values())),
                "max_multiplicity": max(sig_counts.values()),
                "frozen_prefix_stats": prefix_stats,
                "estimator_notes": {
                    "chao1": "f1^2/(2*f2) on family-signature frequencies at the stated prefix",
                    "occupancy_fit_M": "bisect solve of u = M*(1-exp(-n/M)); assumes iid uniform draws from M equally likely signatures — descriptive only",
                },
            }
            failures += [f"verifier errors in {key}: {len(verifier_errors)}"] if verifier_errors else []

    cross_ambiguity = {s: cells for s, cells in cross_cell.items() if len(set(cells)) > 1}
    if not all(renaming_results):
        failures.append("renaming instability")
    if not all(permutation_results):
        failures.append("order-permutation instability")
    if not all(r["family_signatures_differ"] and r["shared_member_confirmed"] for r in orbit_results):
        failures.append("orbit-shared-member test failed")
    if group_canonical_mismatches:
        failures.append(f"repeated-signature groups with differing canonical forms: {group_canonical_mismatches}")
    if determinism_mismatches:
        failures.append(f"regeneration nondeterminism: {determinism_mismatches}")
    if cross_ambiguity:
        failures.append(f"cross-cell signature ambiguity: {len(cross_ambiguity)}")

    evidence = {
        "schema_id": "E0-V061-STRUCTURAL-PREFLIGHT-R1-v0",
        "authority": "V06-STRUCTSIG-REMEDIATION-1; supersedes 74a80fd evidence (STRUCTURAL_PREFLIGHT_INVALID_FOR_SUPPORT_CONCLUSION)",
        "bootstrap_only": True,
        "model_scoring_performed": False,
        "pilot_namespace": args.pilot_namespace,
        "pilot_namespace_burned": True,
        "authoritative_msel_materialization_performed": False,
        "code_git_commit": git_head(),
        "working_tree_clean_at_start": tree_clean(),
        "structsig_implementation": "structsig_r1: exact IR canonicalization, multiset incidence, full-length hashes, family = hash of sorted triple of canonical variant representations",
        "support_measurement_is_descriptive_only": True,
        "collision_observed_is_not_a_failure": True,
        "discrimination_tests": {
            "adversarial_multiplicity": multiplicity_test,
            "renaming_stability_all": all(renaming_results),
            "order_permutation_stability_all": all(permutation_results),
            "orbit_shared_member_tests": orbit_results,
            "repeated_group_canonical_mismatches": group_canonical_mismatches,
        },
        "per_cell": per_cell,
        "determinism_mismatches": determinism_mismatches,
        "cross_cell_ambiguity_count": len(cross_ambiguity),
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
        "wall_seconds": round(time.time() - started, 1),
    }

    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8", newline="\n")
    summary = {k: evidence[k] for k in (
        "status", "discrimination_tests", "per_cell", "determinism_mismatches",
        "cross_cell_ambiguity_count", "failures",
    )}
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
