"""Generate burned smoke + replay corpora for V06-GPU-BOOTSTRAP.

Creates fresh namespaces disjoint from the frozen MSEL burn, validates
against the MSEL burn index (family IDs and r3 StructSigs), and writes
compact JSONL corpora for the 400-update resource smokes and deterministic
model-replay fixtures.
"""

from __future__ import annotations

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
from msel_verifier import verify_family, counterfactual_invariance

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CORPUS = REPO_ROOT / "local_data" / "e0_v061_msel"
SMOKE_OUT = REPO_ROOT / "local_data" / "e0_v061_smoke"
REPLAY_OUT = REPO_ROOT / "local_data" / "e0_v061_replay"
DOCS = REPO_ROOT / "docs" / "experiments" / "e0" / "v061"
OUTPUT = DOCS / "SMOKE_REPLAY_CORPORA_EVIDENCE.json"

NS_SMOKE = "ExpertForge-E0-v061-smoke-burn"
NS_REPLAY = "ExpertForge-E0-v061-replay-burn"

# Smoke: enough families for 400 updates × 128 = 51,200 presentations
# (stream cycles, so we just need a representative disjoint set).
# 2 surfaces × 4 depths × 1,000 families = 8,000 families / 24,000 examples.
SMOKE_FAMILIES_PER_DEPTH = 1000

# Replay: much smaller (short trajectory with ≥2 checkpoints).
# 2 surfaces × 4 depths × 100 families = 800 families / 2,400 examples.
REPLAY_FAMILIES_PER_DEPTH = 100


def cjson(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fam_sig(fam) -> str:
    reps = sorted(structsig_r3.variant_canonical(v) for v in fam["variants"])
    fc = cjson({"v": "CMDR-StructSig-v1-r3-family", "orbit": reps})
    return hashlib.sha256(fc.encode("utf-8")).hexdigest()


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def generate_and_validate(ns: str, out_dir: Path, families_per_depth: int, burn_fids: set, burn_sigs: set, label: str) -> dict:
    """Generate families, validate disjointness from MSEL burn, write JSONL."""
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    all_variants = []
    total_fams = 0
    verr = 0
    fid_overlaps = 0
    sig_overlaps = 0

    for surface in ("ID", "STRUCT"):
        for depth in range(1, 5):
            fams = build_families(ns, label, surface, depth, range(families_per_depth))
            for fam in fams:
                verr += len(verify_family(fam)) + len(counterfactual_invariance(fam))
                fid = fam["family_id"]
                sig = fam_sig(fam)
                if fid in burn_fids:
                    fid_overlaps += 1
                if sig in burn_sigs:
                    sig_overlaps += 1
                total_fams += 1
                for v in fam["variants"]:
                    all_variants.append(v)

    # Write combined train file (smoke trains on all depths together)
    train_path = out_dir / "train.jsonl"
    data = b""
    for row in all_variants:
        data += cjson(row).encode("utf-8") + b"\n"
    train_path.write_bytes(data)
    train_sha = hashlib.sha256(data).hexdigest()

    # Write a small dev file (subset of the same namespace, different family indices)
    dev_variants = []
    for surface in ("ID",):
        for depth in range(1, 5):
            fams = build_families(ns, f"{label}_dev", surface, depth, range(50))
            for fam in fams:
                for v in fam["variants"]:
                    dev_variants.append(v)
    dev_path = out_dir / "dev.jsonl"
    dev_data = b""
    for row in dev_variants:
        dev_data += cjson(row).encode("utf-8") + b"\n"
    dev_path.write_bytes(dev_data)
    dev_sha = hashlib.sha256(dev_data).hexdigest()

    return {
        "namespace": ns,
        "families": total_fams,
        "train_examples": len(all_variants),
        "dev_examples": len(dev_variants),
        "train_sha256": train_sha,
        "dev_sha256": dev_sha,
        "verifier_errors": verr,
        "family_id_overlaps_with_msel": fid_overlaps,
        "structsig_overlaps_with_msel": sig_overlaps,
        "disjoint": fid_overlaps == 0 and sig_overlaps == 0,
        "seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    started = time.time()

    # Load MSEL burn index
    burn_index = json.loads((CORPUS / "MSEL_BURN_INDEX.json").read_text(encoding="utf-8"))
    burn_fids = set(burn_index["entries"].keys())
    burn_sigs = set(burn_index["entries"].values())
    print(f"MSEL burn: {len(burn_fids)} families, {len(burn_sigs)} unique StructSigs", flush=True)

    # Generate smoke corpus
    print("=== Generating smoke corpus ===", flush=True)
    smoke = generate_and_validate(NS_SMOKE, SMOKE_OUT, SMOKE_FAMILIES_PER_DEPTH, burn_fids, burn_sigs, "smoke")
    print(f"  {smoke['families']} families / {smoke['train_examples']} examples | "
          f"disjoint={smoke['disjoint']} | verr={smoke['verifier_errors']}", flush=True)

    # Generate replay corpus
    print("=== Generating replay corpus ===", flush=True)
    replay = generate_and_validate(NS_REPLAY, REPLAY_OUT, REPLAY_FAMILIES_PER_DEPTH, burn_fids, burn_sigs, "replay")
    print(f"  {replay['families']} families / {replay['train_examples']} examples | "
          f"disjoint={replay['disjoint']} | verr={replay['verifier_errors']}", flush=True)

    all_ok = (
        smoke["disjoint"] and smoke["verifier_errors"] == 0
        and replay["disjoint"] and replay["verifier_errors"] == 0
    )

    evidence = {
        "schema_id": "E0-V061-SMOKE-REPLAY-CORPORA-v0",
        "authority": "V06-GPU-BOOTSTRAP-EXECUTION-AUTHORIZED",
        "code_git_commit": git_head(),
        "smoke_corpus": smoke,
        "replay_corpus": replay,
        "namespaces_burned": [NS_SMOKE, NS_REPLAY],
        "status": "PASS" if all_ok else "FAIL",
        "wall_seconds": round(time.time() - started, 1),
    }

    OUTPUT.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nSTATUS: {evidence['status']}", flush=True)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
