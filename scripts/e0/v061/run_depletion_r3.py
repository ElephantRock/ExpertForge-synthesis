"""Full MSEL-burn structural depletion probe — V06-STRUCTSIG-REMEDIATION-3.

Simulates the complete MSEL structural burn per depth:
  - 32,000 train_pool families (both surfaces as in the MSEL contract)
  - 500 dev_ID families
  - 500 eval_ID families
  - 500 eval_STRUCT families (StructSig-disjoint from all other MSEL splits)

Burns every MSEL signature. Then, from a completely separate burned
namespace, simulates the inherited qualification-scale demand and measures
whether fresh families can be admitted while avoiding the entire MSEL
signature set. Reports attempts, accepts, rejection reasons, discovery
curves, per-split consumed signatures, eval_STRUCT selection rejection rate,
and fresh-qualification rejection rate by depth.
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
EVIDENCE_PATH = REPO_ROOT / "docs" / "experiments" / "e0" / "v061" / "DEPLETION_R3.json"
NS_MSEL = "ExpertForge-E0-v061-msel-depletion-sim"
NS_FRESH = "ExpertForge-E0-v061-msel-depletion-fresh"

TRAIN_POOL_PER_DEPTH = 32000
DEV_ID_PER_DEPTH = 500
EVAL_ID_PER_DEPTH = 500
EVAL_STRUCT_PER_DEPTH = 500
# inherited qualification-scale demand per depth (SS12 of the contract:
# the v0.5 qualification used 2000 families/depth for its ID train surface)
QUAL_DEMAND_PER_DEPTH = 2000
# how many extra families to attempt for eval_STRUCT disjoint selection
EVAL_STRUCT_EXTRA_FACTOR = 4


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def fam_sig(fam) -> str:
    reps = sorted(structsig_r3.variant_canonical(v) for v in fam["variants"])
    fc = json.dumps({"v": "CMDR-StructSig-v1-r3-family", "orbit": reps}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(fc.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depths", nargs="+", type=int, default=[1, 2, 3, 4])
    args = parser.parse_args()

    started = time.time()
    burn = {}  # (depth, surface_for_split) -> set of signatures
    results = {"depths": args.depths, "msel_burn": {}, "fresh_qualification": {}, "assumptions": [
        "train_pool simulated at 32,000 families/depth on both ID and STRUCT surfaces (SS4.1 does not split the pool by surface; both are generated to bound the burn)",
        "dev_ID and eval_ID at 500 families/depth each (ID surface)",
        "eval_STRUCT at 500 families/depth selected StructSig-disjoint from all other MSEL splits",
        "fresh-namespace demand set at 2,000 families/depth (v0.5 qualification-scale) on both surfaces",
        "all MSEL signature hashes are burned from any later corpus (SS13), not merely eval_STRUCT",
    ]}

    for depth in args.depths:
        depth_t0 = time.time()
        all_msel_sigs = set()

        # 1. train_pool: 32k families per surface per depth
        for surface in ("ID", "STRUCT"):
            fams = build_families(NS_MSEL, "train_pool", surface, depth, range(TRAIN_POOL_PER_DEPTH))
            sigs = set()
            curve = []
            for i, fam in enumerate(fams):
                sigs.add(fam_sig(fam))
                if (i + 1) % 4000 == 0:
                    curve.append({"n": i + 1, "unique": len(sigs)})
            all_msel_sigs |= sigs
            results["msel_burn"].setdefault(f"train_{surface}|d{depth}", {
                "families": TRAIN_POOL_PER_DEPTH,
                "unique_signatures": len(sigs),
                "discovery_curve": curve,
            })
            print(f"  train_{surface} d{depth}: {len(sigs)} unique from {TRAIN_POOL_PER_DEPTH}", flush=True)

        # 2. dev_ID + eval_ID
        for split in ("dev_ID", "eval_ID"):
            fams = build_families(NS_MSEL, split, "ID", depth, range(DEV_ID_PER_DEPTH))
            sigs = {fam_sig(f) for f in fams}
            all_msel_sigs |= sigs
            results["msel_burn"][f"{split}|d{depth}"] = {
                "families": DEV_ID_PER_DEPTH if split == "dev_ID" else EVAL_ID_PER_DEPTH,
                "unique_signatures": len(sigs),
            }
            print(f"  {split} d{depth}: {len(sigs)} unique", flush=True)

        # 3. eval_STRUCT: StructSig-disjoint from all other MSEL splits
        struct_fams = build_families(
            NS_MSEL, "eval_STRUCT", "STRUCT", depth,
            range(EVAL_STRUCT_PER_DEPTH * EVAL_STRUCT_EXTRA_FACTOR)
        )
        accepted = []
        rejected_burned = 0
        for fam in struct_fams:
            sig = fam_sig(fam)
            if sig in all_msel_sigs:
                rejected_burned += 1
            else:
                accepted.append(fam)
                all_msel_sigs.add(sig)
                if len(accepted) >= EVAL_STRUCT_PER_DEPTH:
                    break
        results["msel_burn"][f"eval_STRUCT_selection|d{depth}"] = {
            "demand": EVAL_STRUCT_PER_DEPTH,
            "accepted_disjoint": len(accepted),
            "rejected_burned_signature": rejected_burned,
            "attempts": len(accepted) + rejected_burned,
            "rejection_rate": round(rejected_burned / max(1, len(accepted) + rejected_burned), 4),
            "exhausted_pool": len(accepted) < EVAL_STRUCT_PER_DEPTH,
        }
        print(f"  eval_STRUCT d{depth}: accepted {len(accepted)}/{EVAL_STRUCT_PER_DEPTH} "
              f"(rejected {rejected_burned})", flush=True)

        burn[depth] = all_msel_sigs
        results["msel_burn"][f"total_burn|d{depth}"] = {
            "total_unique_signatures": len(all_msel_sigs),
            "depth_minutes": round((time.time() - depth_t0) / 60, 1),
        }
        print(f"  TOTAL BURN d{depth}: {len(all_msel_sigs)} signatures | "
              f"{(time.time()-depth_t0)/60:.1f} min", flush=True)

    # 4. fresh-namespace qualification-demand acceptance
    for depth in args.depths:
        burn_sigs = burn[depth]
        for surface in ("ID", "STRUCT"):
            accepted = rejected = 0
            fams = build_families(
                NS_FRESH, "qualification_sim", surface, depth,
                range(QUAL_DEMAND_PER_DEPTH * 3)  # 3x pool
            )
            for fam in fams:
                if accepted >= QUAL_DEMAND_PER_DEPTH:
                    break
                sig = fam_sig(fam)
                if sig in burn_sigs:
                    rejected += 1
                else:
                    accepted += 1
            total = accepted + rejected
            results["fresh_qualification"][f"{surface}|d{depth}"] = {
                "demand": QUAL_DEMAND_PER_DEPTH,
                "accepted_disjoint": accepted,
                "rejected_burned": rejected,
                "acceptance_rate": round(accepted / max(1, total), 4),
                "exhausted_pool": accepted < QUAL_DEMAND_PER_DEPTH,
            }
            print(f"  fresh {surface} d{depth}: accepted {accepted}/{QUAL_DEMAND_PER_DEPTH} "
                  f"(rejected {rejected}, rate {accepted/max(1,total):.3f})", flush=True)

    results["status"] = "COMPLETED"
    results["wall_seconds"] = round(time.time() - started, 1)
    results["code_git_commit"] = git_head()

    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({
        "total_burn_per_depth": {f"d{d}": results["msel_burn"][f"total_burn|d{d}"]["total_unique_signatures"] for d in args.depths},
        "eval_struct_rejection": {f"d{d}": results["msel_burn"][f"eval_STRUCT_selection|d{d}"]["rejection_rate"] for d in args.depths},
        "fresh_qual": {k: v["acceptance_rate"] for k, v in results["fresh_qualification"].items()},
        "wall_minutes": round(results["wall_seconds"] / 60, 1),
    }, indent=2))


if __name__ == "__main__":
    main()
