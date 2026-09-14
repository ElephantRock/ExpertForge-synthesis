"""V06-GPU-BOOTSTRAP-REMEDIATION-1 (3/5): structurally fresh smoke/replay fixtures.

Rejection-sampling generator producing r2 fixture corpora with ZERO overlap
against the cumulative burn in BOTH identity classes (family IDs AND r3
StructSig hashes):

  cumulative burn = frozen MSEL burn index (134,000 families / 28,138 sigs)
                    ∪ r0 fixture identities+signatures (smoke + replay,
                      train + dev — burned by the r0 incident; their ~3,280
                      structurally fresh signatures included)

Firewall semantics (per authority ruling recorded in
incidents/STRUCTSIG_FIREWALL_GATE_WEAKENED_POST_OBSERVATION.md):
  - "duplicate StructSigs are allowed inside MSEL" ≠ "a StructSig already
    burned by MSEL may be reused in a later bootstrap fixture". The latter is
    forbidden. Within-fixture (and train↔dev within one fixture) structural
    multiplicity remains permitted, mirroring the production corpus.
  - Cross-fixture: replay_r2 additionally avoids smoke_r2 signatures
    (conservative cumulative-burn reading; once a fixture is generated and
    burned, its signatures are used).
  - The criterion is NOT relaxed anywhere; rejection rates are recorded as
    evidence.

Deterministic: sequential family-index scan with fixed chunk size.
Second pass independently re-verifies the WRITTEN files against the extended
burn set (both identity classes = 0) and re-runs the verifier.

New namespaces burned by this run:
  ExpertForge-E0-v061-smoke-burn-r2 (+ its _dev split)
  ExpertForge-E0-v061-replay-burn-r2 (+ its _dev split)
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import structsig_r3
from msel_corpus import build_families
from msel_verifier import verify_family, counterfactual_invariance

REPO_ROOT = HERE.parent.parent.parent
CORPUS = REPO_ROOT / "local_data" / "e0_v061_msel"
R0_SMOKE = REPO_ROOT / "local_data" / "e0_v061_smoke"
R0_REPLAY = REPO_ROOT / "local_data" / "e0_v061_replay"
SMOKE_OUT = REPO_ROOT / "local_data" / "e0_v061_smoke_r2"
REPLAY_OUT = REPO_ROOT / "local_data" / "e0_v061_replay_r2"
DOCS = REPO_ROOT / "docs" / "experiments" / "e0" / "v061"
OUTPUT = DOCS / "SMOKE_REPLAY_CORPORA_R2_EVIDENCE.json"

NS_SMOKE = "ExpertForge-E0-v061-smoke-burn-r2"
NS_REPLAY = "ExpertForge-E0-v061-replay-burn-r2"

SMOKE_TRAIN_PER_CELL = 1000   # 2 surfaces × 4 depths → 8,000 families / 24,000 ex
SMOKE_DEV_PER_CELL = 50       # ID surface × 4 depths → 200 families / 600 ex
REPLAY_TRAIN_PER_CELL = 100   # 2 surfaces × 4 depths → 800 families / 2,400 ex
REPLAY_DEV_PER_CELL = 50      # ID surface × 4 depths → 200 families / 600 ex
CHUNK = 256


def cjson(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def fam_sig(fam) -> str:
    reps = sorted(structsig_r3.variant_canonical(v) for v in fam["variants"])
    fc = cjson({"v": "CMDR-StructSig-v1-r3-family", "orbit": reps})
    return hashlib.sha256(fc.encode("utf-8")).hexdigest()


def load_r0_fixture_identities() -> tuple[set, set, dict]:
    """Recompute r0 fixture family IDs + signatures from the written JSONLs
    (train + dev of both r0 fixtures). These identities are burned."""
    fids: set = set()
    sigs: set = set()
    counts = {}
    by_fam: dict = {}
    for base, label in ((R0_SMOKE, "r0_smoke"), (R0_REPLAY, "r0_replay")):
        for part in ("train", "dev"):
            path = base / f"{part}.jsonl"
            n_rows = 0
            with open(path, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    by_fam.setdefault(row["family_id"], []).append(row)
                    n_rows += 1
            counts[f"{label}_{part}"] = {"examples": n_rows}
    t0 = time.time()
    for fid, rows in by_fam.items():
        fids.add(fid)
        sigs.add(fam_sig({"variants": rows}))
    counts["r0_family_total"] = len(by_fam)
    counts["r0_unique_sig_total"] = len(sigs)
    counts["r0_sig_recompute_seconds"] = round(time.time() - t0, 1)
    return fids, sigs, counts


def generate_fresh(ns: str, label: str, train_per_cell: int, dev_per_cell: int,
                   out_dir: Path, burn_fids: set, burn_sigs: set) -> dict:
    """Deterministic sequential rejection sampling with both-class firewall."""
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    stats = {
        "candidates_generated": 0,
        "accepted_families": 0,
        "rejected_family_id_collision": 0,
        "rejected_structsig_burn": 0,
        "verifier_errors": 0,
    }

    def fresh_stream(split: str, surface: str, depth: int, quota: int):
        """Yield accepted families for one (surface, depth) cell."""
        produced = 0
        index = 0
        while produced < quota:
            fams = build_families(ns, split, surface, depth, range(index, index + CHUNK))
            index += CHUNK
            for fam in fams:
                if produced >= quota:
                    break
                stats["candidates_generated"] += 1
                stats["verifier_errors"] += len(verify_family(fam)) + len(counterfactual_invariance(fam))
                fid = fam["family_id"]
                sig = fam_sig(fam)
                if fid in burn_fids:
                    stats["rejected_family_id_collision"] += 1
                    continue
                if sig in burn_sigs:
                    stats["rejected_structsig_burn"] += 1
                    continue
                produced += 1
                stats["accepted_families"] += 1
                yield fam

    train_fams = []
    for surface in ("ID", "STRUCT"):
        for depth in range(1, 5):
            train_fams.extend(fresh_stream(label, surface, depth, train_per_cell))
    train_rows = [v for fam in train_fams for v in fam["variants"]]
    data = b"".join(cjson(row).encode("utf-8") + b"\n" for row in train_rows)
    (out_dir / "train.jsonl").write_bytes(data)
    train_sha = hashlib.sha256(data).hexdigest()

    dev_fams = []
    for surface in ("ID",):
        for depth in range(1, 5):
            dev_fams.extend(fresh_stream(f"{label}_dev", surface, depth, dev_per_cell))
    dev_rows = [v for fam in dev_fams for v in fam["variants"]]
    dev_data = b"".join(cjson(row).encode("utf-8") + b"\n" for row in dev_rows)
    (out_dir / "dev.jsonl").write_bytes(dev_data)
    dev_sha = hashlib.sha256(dev_data).hexdigest()

    fixture_sigs = [fam_sig(f) for fam in train_fams] + [fam_sig(f) for fam in dev_fams]
    stats.update({
        "namespace": ns,
        "train_families": len(train_fams),
        "train_examples": len(train_rows),
        "dev_families": len(dev_fams),
        "dev_examples": len(dev_rows),
        "train_sha256": train_sha,
        "dev_sha256": dev_sha,
        "unique_structsigs_in_fixture": len(set(fixture_sigs)),
        "fixture_structsig_multiplicity": round(len(fixture_sigs) / len(set(fixture_sigs)), 3),
        "acceptance_rate": round(stats["accepted_families"] / max(1, stats["candidates_generated"]), 4),
        "seconds": round(time.time() - t0, 1),
    })
    return stats, set(fixture_sigs)


def reverify_written(out_dir: Path, burn_fids: set, burn_sigs: set) -> dict:
    """Independent second pass over the WRITTEN files."""
    fid_hits = 0
    sig_hits = 0
    by_fam: dict = {}
    verr = 0
    examples = 0
    for part in ("train", "dev"):
        with open(out_dir / f"{part}.jsonl", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                by_fam.setdefault(row["family_id"], []).append(row)
                examples += 1
    for fid, rows in by_fam.items():
        if fid in burn_fids:
            fid_hits += 1
        sig = fam_sig({"variants": rows})
        if sig in burn_sigs:
            sig_hits += 1
        fam = {"variants": rows}
        verr += len(verify_family(fam)) + len(counterfactual_invariance(fam))
    return {
        "examples": examples,
        "families": len(by_fam),
        "family_id_hits": fid_hits,
        "structsig_hits": sig_hits,
        "verifier_errors": verr,
    }


def main() -> None:
    started = time.time()

    burn_index = json.loads((CORPUS / "MSEL_BURN_INDEX.json").read_text(encoding="utf-8"))
    msel_fids = set(burn_index["entries"].keys())
    msel_sigs = set(burn_index["entries"].values())
    print(f"MSEL burn: {len(msel_fids)} families, {len(msel_sigs)} unique StructSigs", flush=True)

    r0_fids, r0_sigs, r0_counts = load_r0_fixture_identities()
    print(f"r0 fixtures burned: {r0_counts['r0_family_total']} families, "
          f"{r0_counts['r0_unique_sig_total']} unique sigs "
          f"({r0_counts['r0_sig_recompute_seconds']}s)", flush=True)

    burn_fids = msel_fids | r0_fids
    burn_sigs = msel_sigs | r0_sigs

    print("=== Generating smoke_r2 (rejection sampling, both-class firewall) ===", flush=True)
    smoke, smoke_sigs = generate_fresh(
        NS_SMOKE, "smoke", SMOKE_TRAIN_PER_CELL, SMOKE_DEV_PER_CELL,
        SMOKE_OUT, burn_fids, burn_sigs)
    print(f"  accepted {smoke['accepted_families']} / {smoke['candidates_generated']} candidates "
          f"(rate {smoke['acceptance_rate']}); sig-rejections {smoke['rejected_structsig_burn']}; "
          f"fid-rejections {smoke['rejected_family_id_collision']}; verr {smoke['verifier_errors']}; "
          f"{smoke['seconds']}s", flush=True)

    # Cross-fixture cumulative burn: replay avoids smoke_r2 signatures too.
    replay_burn_sigs = burn_sigs | smoke_sigs
    print("=== Generating replay_r2 (cumulative burn incl. smoke_r2) ===", flush=True)
    replay, replay_sigs = generate_fresh(
        NS_REPLAY, "replay", REPLAY_TRAIN_PER_CELL, REPLAY_DEV_PER_CELL,
        REPLAY_OUT, burn_fids, replay_burn_sigs)
    print(f"  accepted {replay['accepted_families']} / {replay['candidates_generated']} candidates "
          f"(rate {replay['acceptance_rate']}); sig-rejections {replay['rejected_structsig_burn']}; "
          f"fid-rejections {replay['rejected_family_id_collision']}; verr {replay['verifier_errors']}; "
          f"{replay['seconds']}s", flush=True)

    print("=== Independent second pass over written files ===", flush=True)
    rv_smoke = reverify_written(SMOKE_OUT, burn_fids, burn_sigs)
    rv_replay = reverify_written(REPLAY_OUT, burn_fids, replay_burn_sigs)
    print(f"  smoke_r2: {rv_smoke}", flush=True)
    print(f"  replay_r2: {rv_replay}", flush=True)

    smoke_gate = (rv_smoke["family_id_hits"] == 0 and rv_smoke["structsig_hits"] == 0
                  and rv_smoke["verifier_errors"] == 0 and smoke["verifier_errors"] == 0)
    replay_gate = (rv_replay["family_id_hits"] == 0 and rv_replay["structsig_hits"] == 0
                   and rv_replay["verifier_errors"] == 0 and replay["verifier_errors"] == 0)

    evidence = {
        "schema_id": "E0-V061-SMOKE-REPLAY-CORPORA-R2-v0",
        "authority": "V06-GPU-BOOTSTRAP-REMEDIATION-1",
        "supersedes": "incidents/gpu_bootstrap_r0/SMOKE_REPLAY_CORPORA_EVIDENCE.json (diagnostic)",
        "code_git_commit": git_head(),
        "firewall_semantics": {
            "identity_classes": ["family_id", "r3_structsig_sha256"],
            "cumulative_burn": {
                "msel_families": len(msel_fids),
                "msel_unique_structsigs": len(msel_sigs),
                "r0_fixture_families": r0_counts["r0_family_total"],
                "r0_fixture_unique_structsigs": r0_counts["r0_unique_sig_total"],
                "total_burn_families": len(burn_fids),
                "total_burn_structsigs": len(burn_sigs),
            },
            "within_fixture_multiplicity": "permitted (mirrors production corpus)",
            "cross_fixture": "replay_r2 additionally avoids smoke_r2 signatures (cumulative burn)",
            "relaxations": "NONE",
        },
        "r0_fixture_identity_recompute": r0_counts,
        "smoke_corpus_r2": smoke,
        "replay_corpus_r2": replay,
        "second_pass_reverification": {"smoke_r2": rv_smoke, "replay_r2": rv_replay},
        "namespaces_burned": [NS_SMOKE, NS_REPLAY],
        "gates": {"smoke_r2_both_class_zero": smoke_gate,
                  "replay_r2_both_class_zero": replay_gate},
        "status": "PASS" if (smoke_gate and replay_gate) else "FAIL",
        "wall_seconds": round(time.time() - started, 1),
    }
    OUTPUT.write_text(json.dumps(evidence, indent=2, default=str) + "\n",
                      encoding="utf-8", newline="\n")
    print(f"\nSTATUS: {evidence['status']}  evidence: {OUTPUT}", flush=True)
    sys.exit(0 if evidence["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
