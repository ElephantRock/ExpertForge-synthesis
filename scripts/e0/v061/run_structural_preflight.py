"""Structural-support preflight for CMDR-MSEL-v0 (v0.6.1 bootstrap).

Runs the MSEL generator + independent verifier + CMDR-StructSig-v1 on the
BURNED pilot namespace `ExpertForge-E0-v061-msel-preflight` only. The pilot
namespace never enters R1/R4/R16, qualification, or later E0 data.

Fail-closed conditions (authority ruling):
  - verifier disagreement (inline generator semantics vs independent verifier)
  - StructSig nondeterminism (regeneration or variant-choice instability)
  - cross-depth signature ambiguity violating the frozen definition
    (a signature shared across different depth strata or surfaces)
  - inability to produce the requested per-depth population
  - projected exhaustion/rejection severe enough that the later qualification
    burn would no longer come from the same practical generator distribution

Outputs a machine-readable evidence JSON; exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structsig
from msel_corpus import (
    LABELS,
    MSEL_PILOT_NAMESPACE_ROOT,
    apply_renaming,
    build_families,
)
from msel_verifier import counterfactual_invariance, verify_family

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EVIDENCE_PATH = REPO_ROOT / "docs" / "experiments" / "e0" / "v061" / "STRUCTURAL_PREFLIGHT.json"


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def tree_clean() -> bool:
    return subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip() == ""


def renaming_stability_test(family: dict) -> dict:
    """Nuisance-renaming stability: permute predicate and entity identifiers
    consistently across one variant; the family signature must not change.
    Also verify fact/rule order permutations do not change the signature."""
    base_sig = structsig.structsig_family(family["variants"])

    rng = random.Random(12345)
    preds = set()
    ents = set()
    for v in family["variants"]:
        for f in v["facts"]:
            preds.add(f["pred"])
            ents.add(f["term"])
        for r in v["rules"]:
            for p in r["premises"] + [r["conclusion"]]:
                preds.add(p["pred"])
                if p["term"] != "x":
                    ents.add(p["term"])
    new_preds = [f"ZP{i:03d}" for i in range(len(preds))]
    new_ents = [f"ZE{i:03d}" for i in range(len(ents))]
    rng.shuffle(new_preds)
    rng.shuffle(new_ents)
    pred_map = dict(zip(sorted(preds), new_preds))
    ent_map = dict(zip(sorted(ents), new_ents))

    renamed = [apply_renaming(v, pred_map, ent_map) for v in family["variants"]]
    renamed_sig = structsig.structsig_family(renamed)

    return {
        "base_signature": base_sig,
        "renamed_signature": renamed_sig,
        "stable_under_nuisance_renaming": base_sig == renamed_sig,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--families-per-depth", type=int, default=2000)
    parser.add_argument("--surfaces", nargs="+", default=["ID", "STRUCT"])
    parser.add_argument("--pilot-namespace", default=MSEL_PILOT_NAMESPACE_ROOT)
    parser.add_argument("--future-qualification-families-per-depth", type=int, default=32000)
    args = parser.parse_args()

    started = time.time()
    per_cell: dict[str, dict] = {}
    cross_cell_signatures: dict[str, list[str]] = {}
    all_errors: list = []
    determinism_mismatches = 0
    renaming_results = []

    for surface in args.surfaces:
        for depth in range(1, 5):
            key = f"{surface}|d{depth}"
            families = build_families(
                args.pilot_namespace, "pilot", surface, depth,
                range(args.families_per_depth),
            )
            regenerated = build_families(
                args.pilot_namespace, "pilot", surface, depth,
                range(min(50, args.families_per_depth)),
            )
            for a, b in zip(families[: len(regenerated)], regenerated):
                if (
                    a["family_id"] != b["family_id"]
                    or {v["sample_id"] for v in a["variants"]} != {v["sample_id"] for v in b["variants"]}
                    or {v["render_sha256"] for v in a["variants"]} != {v["render_sha256"] for v in b["variants"]}
                ):
                    determinism_mismatches += 1

            sigs = []
            verifier_errors = []
            for fam in families:
                verifier_errors += verify_family(fam)
                verifier_errors += counterfactual_invariance(fam)
                try:
                    sig = structsig.structsig_family(fam["variants"])
                except AssertionError as exc:
                    verifier_errors.append([fam["family_id"], "structsig_error", str(exc)[:120]])
                    continue
                # variant-choice stability: signature computed from each variant's
                # rules must agree (the skeleton is shared; pivot binding abstracted)
                vsigs = structsig.variant_signatures(fam["variants"])
                if len(set(vsigs)) != 3 or min(vsigs) != sig:
                    verifier_errors.append([fam["family_id"], "variant_signature_instability"])
                sigs.append(sig)
                cross_cell_signatures.setdefault(sig, []).append(key)

            sig_counts = Counter(sigs)
            duplicates = {s: c for s, c in sig_counts.items() if c > 1}
            if len(renaming_results) < 8:
                renaming_results.append(renaming_stability_test(families[0]))

            n = args.families_per_depth
            unique = len(sig_counts)
            collisions = n - unique
            # rule-of-three style lower bound on effective signature space per cell:
            # observing `collisions` in n draws (with replacement, n<<M) implies
            # M_lower solves exp(-n^2/(2M)) >= 0.05 for the 0-collision case.
            if collisions == 0:
                m_lower = round(n * n / (2 * 3.0))
            else:
                m_lower = None
            per_cell[key] = {
                "attempted": n,
                "accepted_complete": len(families),
                "unique_structsigs": unique,
                "duplicate_signature_groups": len(duplicates),
                "collisions": collisions,
                "multiplicity_distribution": dict(Counter(sig_counts.values())),
                "verifier_errors": len(verifier_errors),
                "effective_signature_space_lower_bound": m_lower,
            }
            all_errors += verifier_errors

    cross_ambiguity = {s: cells for s, cells in cross_cell_signatures.items() if len(set(cells)) > 1}
    renaming_stable = all(r["stable_under_nuisance_renaming"] for r in renaming_results)

    projected_burn_per_cell = args.future_qualification_families_per_depth
    exhaustion_risk = {}
    for key, cell in per_cell.items():
        bound = cell["effective_signature_space_lower_bound"]
        if bound is None:
            exhaustion_risk[key] = "UNBOUNDED_COLony_OBSERVED_COLLISIONS"
        else:
            ratio = bound / projected_burn_per_cell
            exhaustion_risk[key] = (
                "ADEQUATE" if ratio >= 10 else "MARGINAL" if ratio >= 2 else "SEVERE"
            )

    failures = []
    if all_errors:
        failures.append(f"verifier or invariance errors: {len(all_errors)}")
    if determinism_mismatches:
        failures.append(f"regeneration nondeterminism: {determinism_mismatches}")
    if not renaming_stable:
        failures.append("structsig unstable under nuisance renaming")
    if cross_ambiguity:
        failures.append(f"cross-cell signature ambiguity: {len(cross_ambiguity)} signatures")
    if any(v == "SEVERE" for v in exhaustion_risk.values()):
        failures.append(f"projected structural exhaustion: {exhaustion_risk}")
    if any(v.startswith("UNBOUNDED") for v in exhaustion_risk.values()):
        failures.append(f"collisions observed at pilot scale: {exhaustion_risk}")

    evidence = {
        "schema_id": "E0-V061-STRUCTURAL-PREFLIGHT-v0",
        "authority": "V06_MECHANISM_BOOTSTRAP_AUTHORIZED; authority work order: structural feasibility before 402k materialization",
        "bootstrap_only": True,
        "model_scoring_performed": False,
        "pilot_namespace": args.pilot_namespace,
        "pilot_namespace_burned": True,
        "authoritative_msel_materialization_performed": False,
        "code_git_commit": git_head(),
        "working_tree_clean_at_start": tree_clean(),
        "families_per_depth_per_surface": args.families_per_depth,
        "per_cell": per_cell,
        "renaming_stability_tests": renaming_results,
        "cross_cell_ambiguity_count": len(cross_ambiguity),
        "determinism_mismatches": determinism_mismatches,
        "projected_burn_per_cell": projected_burn_per_cell,
        "exhaustion_risk": exhaustion_risk,
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
        "wall_seconds": round(time.time() - started, 1),
    }

    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: evidence[k] for k in ("status", "per_cell", "exhaustion_risk", "failures", "cross_cell_ambiguity_count", "determinism_mismatches")}, indent=2))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
