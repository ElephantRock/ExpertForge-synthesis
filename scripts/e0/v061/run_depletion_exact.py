"""Contract-exact MSEL-burn structural depletion replay — V06-MSEL-SURFACE-BINDING-1.

SPEC-BOUND clarification (authority ruling): train_pool is ID surface only.
This replay simulates the exact contract SS4.1 topology per depth:
  - 32,000 train_pool / ID
  - 500 dev_ID / ID
  - 500 eval_ID / ID
  - 500 eval_STRUCT / STRUCT, StructSig-disjoint from the three ID splits

Burns all MSEL signatures (SS13). Then from a separate burned namespace,
simulates the inherited qualification-shape demand per depth:
  - 2,000 train_ID / ID (disjoint from MSEL burn)
  - 500 eval_ID / ID (disjoint from MSEL burn AND from qualification train_ID)
  - 500 eval_STRUCT / STRUCT (disjoint from MSEL burn AND from qualification ID surfaces)

Uses a preregistered 20× candidate horizon per target demand (no 3×/4×
cutoffs). Reports exact attempts, accepts, rejection reasons (duplicate
within split, MSEL-burn collision, within-qualification collision), unique
signature accumulation at fixed prefixes, and whether every required split
can be filled. New burned namespaces; no prior depletion namespace reuse.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structsig_r3
from msel_corpus import build_families

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EVIDENCE_PATH = REPO_ROOT / "docs" / "experiments" / "e0" / "v061" / "DEPLETION_EXACT_R3.json"
NS_MSEL = "ExpertForge-E0-v061-msel-deplex-burn"
NS_QUAL = "ExpertForge-E0-v061-qual-deplex-burn"

TRAIN_POOL_PER_DEPTH = 32_000
DEV_ID_PER_DEPTH = 500
EVAL_ID_PER_DEPTH = 500
EVAL_STRUCT_PER_DEPTH = 500
QUAL_TRAIN_ID = 2_000
QUAL_EVAL_ID = 500
QUAL_EVAL_STRUCT = 500
HORIZON_FACTOR = 20  # preregistered: 20x target demand candidate horizon


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def tree_clean() -> bool:
    return subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip() == ""


def fam_sig(fam) -> str:
    reps = sorted(structsig_r3.variant_canonical(v) for v in fam["variants"])
    fc = json.dumps({"v": "CMDR-StructSig-v1-r3-family", "orbit": reps}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(fc.encode("utf-8")).hexdigest()


def select_disjoint(fams, demand, burn, horizon_factor=HORIZON_FACTOR):
    """Select up to `demand` families whose signatures avoid `burn`.
    Scans the full preregistered horizon; reports fills and rejection reasons."""
    accepted_sigs = set()
    accepted_fams = []
    msel_rej = 0
    dup_rej = 0
    attempts = 0
    horizon = demand * horizon_factor
    curve = []  # acceptance at fixed prefixes of the horizon
    for i, fam in enumerate(fams[:horizon]):
        attempts += 1
        sig = fam_sig(fam)
        if sig in burn:
            msel_rej += 1
        elif sig in accepted_sigs:
            dup_rej += 1
        else:
            accepted_sigs.add(sig)
            accepted_fams.append(fam)
            if len(accepted_fams) >= demand:
                break
        if (i + 1) % (horizon // 10) == 0:
            curve.append({"attempted": i + 1, "accepted": len(accepted_fams)})
    return {
        "demand": demand,
        "horizon": horizon,
        "attempts": attempts,
        "accepted": len(accepted_fams),
        "filled": len(accepted_fams) >= demand,
        "msel_burn_rejected": msel_rej,
        "duplicate_rejected": dup_rej,
        "acceptance_rate": round(len(accepted_fams) / max(1, attempts), 4),
        "acceptance_curve": curve,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depths", nargs="+", type=int, default=[1, 2, 3, 4])
    args = parser.parse_args()

    started = time.time()
    results = {
        "schema_id": "E0-V061-DEPLETION-EXACT-R3-v0",
        "authority": "V06-MSEL-SURFACE-BINDING-1 (contract-exact replay after CONSERVATIVE_STRESS_FAIL classification of DEPLETION_R3)",
        "surface_binding": "train_pool = ID surface only (authority SPEC-BOUND ruling)",
        "horizon_factor": HORIZON_FACTOR,
        "namespaces": {"msel": NS_MSEL, "qualification": NS_QUAL, "all_burned": True},
        "depths": args.depths,
        "msel_burn": {},
        "fresh_qualification": {},
        "code_git_commit": git_head(),
        "working_tree_clean_at_start": tree_clean(),
    }

    for depth in args.depths:
        depth_t0 = time.time()
        print(f"=== depth {depth} ===", flush=True)

        # --- MSEL construction ---
        # 1. train_pool: 32k ID families
        t0 = time.time()
        train_fams = build_families(NS_MSEL, "train_pool", "ID", depth, range(TRAIN_POOL_PER_DEPTH))
        train_sigs = set()
        train_curve = []
        for i, fam in enumerate(train_fams):
            train_sigs.add(fam_sig(fam))
            if (i + 1) % 4000 == 0:
                train_curve.append({"n": i + 1, "unique": len(train_sigs)})
        print(f"  train_pool/ID: {len(train_sigs)} unique from {TRAIN_POOL_PER_DEPTH} "
              f"({time.time()-t0:.0f}s)", flush=True)
        results["msel_burn"][f"train_pool|d{depth}"] = {
            "families": TRAIN_POOL_PER_DEPTH, "surface": "ID",
            "unique_signatures": len(train_sigs), "discovery_curve": train_curve,
        }

        # 2. dev_ID + eval_ID
        msel_burn = set(train_sigs)
        for split, count in [("dev_ID", DEV_ID_PER_DEPTH), ("eval_ID", EVAL_ID_PER_DEPTH)]:
            fams = build_families(NS_MSEL, split, "ID", depth, range(count))
            sigs = {fam_sig(f) for f in fams}
            msel_burn |= sigs
            results["msel_burn"][f"{split}|d{depth}"] = {
                "families": count, "surface": "ID", "unique_signatures": len(sigs),
            }
        print(f"  dev_ID+eval_ID added; burn now {len(msel_burn)}", flush=True)

        # 3. eval_STRUCT: disjoint from all MSEL ID splits
        es_horizon = EVAL_STRUCT_PER_DEPTH * HORIZON_FACTOR
        es_fams = build_families(NS_MSEL, "eval_STRUCT", "STRUCT", depth, range(es_horizon))
        es_result = select_disjoint(es_fams, EVAL_STRUCT_PER_DEPTH, msel_burn, HORIZON_FACTOR)
        # add accepted eval_STRUCT sigs to the burn
        for fam in es_fams[:es_result["attempts"]]:
            msel_burn.add(fam_sig(fam))
        results["msel_burn"][f"eval_STRUCT|d{depth}"] = es_result
        print(f"  eval_STRUCT: {es_result['accepted']}/{EVAL_STRUCT_PER_DEPTH} filled={es_result['filled']} "
              f"(msel_rej={es_result['msel_burn_rejected']}, dup={es_result['duplicate_rejected']})", flush=True)

        results["msel_burn"][f"total|d{depth}"] = {
            "total_unique_burned": len(msel_burn),
            "depth_seconds": round(time.time() - depth_t0, 1),
        }
        print(f"  TOTAL MSEL BURN d{depth}: {len(msel_burn)} signatures ({time.time()-depth_t0:.0f}s)", flush=True)

        # --- Fresh qualification simulation ---
        qual_t0 = time.time()
        qual_sigs_all = set()  # across all qual splits for cross-split disjointness

        # qual train_ID: 2000 from ID surface
        qt_horizon = QUAL_TRAIN_ID * HORIZON_FACTOR
        qt_fams = build_families(NS_QUAL, "qual_train_ID", "ID", depth, range(qt_horizon))
        qt_result = select_disjoint(qt_fams, QUAL_TRAIN_ID, msel_burn, HORIZON_FACTOR)
        qt_sigs = set()
        for fam in qt_fams[:qt_result["attempts"]]:
            qt_sigs.add(fam_sig(fam))
        qual_sigs_all |= qt_sigs
        results["fresh_qualification"][f"qual_train_ID|d{depth}"] = qt_result
        print(f"  qual_train_ID: {qt_result['accepted']}/{QUAL_TRAIN_ID} filled={qt_result['filled']}", flush=True)

        # qual eval_ID: 500, disjoint from MSEL burn AND from qual train_ID
        combined_burn = msel_burn | qt_sigs
        qe_horizon = QUAL_EVAL_ID * HORIZON_FACTOR
        qe_fams = build_families(NS_QUAL, "qual_eval_ID", "ID", depth, range(qe_horizon))
        qe_result = select_disjoint(qe_fams, QUAL_EVAL_ID, combined_burn, HORIZON_FACTOR)
        qe_sigs = set()
        for fam in qe_fams[:qe_result["attempts"]]:
            qe_sigs.add(fam_sig(fam))
        qual_sigs_all |= qe_sigs
        results["fresh_qualification"][f"qual_eval_ID|d{depth}"] = qe_result
        print(f"  qual_eval_ID: {qe_result['accepted']}/{QUAL_EVAL_ID} filled={qe_result['filled']}", flush=True)

        # qual eval_STRUCT: 500, disjoint from MSEL burn AND from both qual ID splits
        qstruct_horizon = QUAL_EVAL_STRUCT * HORIZON_FACTOR
        qstruct_fams = build_families(NS_QUAL, "qual_eval_STRUCT", "STRUCT", depth, range(qstruct_horizon))
        qstruct_result = select_disjoint(qstruct_fams, QUAL_EVAL_STRUCT, msel_burn | qual_sigs_all, HORIZON_FACTOR)
        results["fresh_qualification"][f"qual_eval_STRUCT|d{depth}"] = qstruct_result
        print(f"  qual_eval_STRUCT: {qstruct_result['accepted']}/{QUAL_EVAL_STRUCT} filled={qstruct_result['filled']}", flush=True)
        print(f"  qual depth {depth} completed in {(time.time()-qual_t0):.0f}s", flush=True)

    # summary: can every required split be filled?
    all_filled = True
    fill_summary = {}
    for dep in args.depths:
        # dev_ID and eval_ID are always fully constructed (no disjointness
        # constraint against prior MSEL splits per the contract); only
        # eval_STRUCT goes through selection
        for key in ("eval_STRUCT",):
            r = results["msel_burn"][f"{key}|d{dep}"]
            fill_summary[f"msel_{key}|d{dep}"] = r["filled"]
            if not r["filled"]:
                all_filled = False
        fill_summary[f"msel_train_pool|d{dep}"] = True
        fill_summary[f"msel_dev_ID|d{dep}"] = True
        fill_summary[f"msel_eval_ID|d{dep}"] = True
        for key in ("qual_train_ID", "qual_eval_ID", "qual_eval_STRUCT"):
            r = results["fresh_qualification"][f"{key}|d{dep}"]
            fill_summary[f"{key}|d{dep}"] = r["filled"]
            if not r["filled"]:
                all_filled = False
    results["fill_summary"] = fill_summary
    results["all_splits_filled"] = all_filled
    results["s4_4_verdict"] = "SUFFICIENT" if all_filled else "INSUFFICIENT"
    results["wall_seconds"] = round(time.time() - started, 1)

    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({
        "s4_4_verdict": results["s4_4_verdict"],
        "fill_summary": fill_summary,
        "wall_minutes": round(results["wall_seconds"] / 60, 1),
    }, indent=2))


if __name__ == "__main__":
    main()
