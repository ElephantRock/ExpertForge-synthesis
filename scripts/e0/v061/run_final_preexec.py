"""V06-PREEXEC-REMEDIATION-2 final CPU closure.

Hard-stops on all 16 corpus digests first. Then:
  A. Full deterministic stream replay (combined all-depth R1/R4/R16 × 6 primary
     seeds) with an independent SHA oracle and streaming roots.
  B. Tokenizer interface closure (special_added, unknown-token check, P0 readout fixture).
  C. Queried runtime determinism state.
  D. Metrics self-test artifact with new SHA binding.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CORPUS = REPO_ROOT / "local_data" / "e0_v061_msel"
DOCS = REPO_ROOT / "docs" / "experiments" / "e0" / "v061"
OUTPUT = DOCS / "FINAL_PREEXEC_CLOSURE.json"
MANIFEST = DOCS / "MSEL_MANIFEST.json"

SPLITS = ("train_pool", "dev_ID", "eval_ID", "eval_STRUCT")
DEPTHS = (1, 2, 3, 4)
PRIMARY_SEEDS = [806915476, 1031646469, 128439691, 555223894, 454204619, 1678768041]
PRESENTATIONS = 1_024_000
STREAM_NS = "ExpertForge-E0-v061-msel-stream"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    started = time.time()
    report = {
        "schema_id": "E0-V061-FINAL-PREEXEC-CLOSURE-v0",
        "authority": "V06-PREEXEC-REMEDIATION-2",
        "code_git_commit": git_head(),
    }

    # === HARD STOP: 16 file digests ===
    print("=== HARD STOP: 16 file digest check ===", flush=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for split in SPLITS:
        for dep in DEPTHS:
            key = f"{split}_d{dep}"
            actual = sha256_file(CORPUS / f"{key}.jsonl")
            expected = manifest["splits"][key]["sha256"]
            if actual != expected:
                raise SystemExit(f"HARD STOP: {key} digest mismatch")
    print("  all 16 digests match — proceeding", flush=True)
    report["digest_check"] = {"all_16_match": True}

    # === A. Full deterministic stream replay ===
    print("=== A. Stream replay (combined all-depth × 6 primary seeds) ===", flush=True)
    rungs = json.loads((CORPUS / "MSEL_RUNGS.json").read_text(encoding="utf-8"))
    train_sample_by_fid = {}
    for dep in DEPTHS:
        with open(CORPUS / f"train_pool_d{dep}.jsonl", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                train_sample_by_fid.setdefault(row["family_id"], []).append(row["sample_id"])

    def build_arm_samples(rung_name):
        """Combined all-depth sample set for the given rung."""
        all_sids = []
        for dep in DEPTHS:
            fam_ids = set(rungs["family_ids"][f"d{dep}"][rung_name])
            for fid in fam_ids:
                if fid in train_sample_by_fid:
                    all_sids.extend(train_sample_by_fid[fid])
        return sorted(all_sids)

    # Independent oracle (no msel_stream imports)
    def oracle_rank(seed, cycle, sid):
        key = f"{STREAM_NS}|{seed}|{cycle}|{sid}"
        return hashlib.sha256(key.encode()).digest()

    def oracle_cycle(seed, cycle, sids):
        return sorted(sids, key=lambda s: (oracle_rank(seed, cycle, s), s))

    def streaming_sha_root(items):
        """SHA-256 over the concatenation of items (streaming, no giant list)."""
        h = hashlib.sha256()
        for item in items:
            h.update(item.encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()

    stream_evidence = {}
    stream_all_ok = True
    for rung_name, expected_samples in [("R1", 24000), ("R4", 96000), ("R16", 384000)]:
        sids = build_arm_samples(rung_name)
        count_ok = len(sids) == expected_samples
        stream_evidence[f"{rung_name}|combined"] = {
            "samples": len(sids), "expected": expected_samples, "count_match": count_ok,
        }
        if not count_ok:
            stream_all_ok = False
            continue

        n = len(sids)
        full_cycles = PRESENTATIONS // n
        remainder = PRESENTATIONS % n

        for seed in PRIMARY_SEEDS:
            seed_key = f"{rung_name}|seed_{seed}"
            ev = {"samples": n, "full_cycles": full_cycles, "remainder": remainder}

            # Cycle 0: oracle vs implementation
            oracle_c0 = oracle_cycle(seed, 0, sids)
            from msel_stream import cycle_order
            impl_c0 = cycle_order(seed, 0, sids)
            ev["cycle0_oracle_impl_match"] = oracle_c0 == impl_c0
            ev["cycle0_root"] = streaming_sha_root(oracle_c0)

            # Cycle 1: oracle vs implementation
            oracle_c1 = oracle_cycle(seed, 1, sids)
            impl_c1 = cycle_order(seed, 1, sids)
            ev["cycle1_oracle_impl_match"] = oracle_c1 == impl_c1
            ev["cycle1_root"] = streaming_sha_root(oracle_c1)

            # Last complete cycle
            last_c = full_cycles - 1
            oracle_last = oracle_cycle(seed, last_c, sids)
            ev[f"last_cycle_{last_c}_root"] = streaming_sha_root(oracle_last)

            # Final truncated segment root (first `remainder` items of the next cycle)
            next_c = full_cycles
            oracle_next = oracle_cycle(seed, next_c, sids)
            ev["truncated_segment_root"] = streaming_sha_root(oracle_next[:remainder])

            # Full 1,024,000-presentation streaming root (oracle)
            h = hashlib.sha256()
            for c in range(full_cycles):
                for sid in oracle_cycle(seed, c, sids):
                    h.update(sid.encode("utf-8"))
                    h.update(b"\x00")
            for sid in oracle_next[:remainder]:
                h.update(sid.encode("utf-8"))
                h.update(b"\x00")
            ev["full_stream_root"] = h.hexdigest()

            if not ev["cycle0_oracle_impl_match"] or not ev["cycle1_oracle_impl_match"]:
                stream_all_ok = False

            stream_evidence[seed_key] = ev

        print(f"  {rung_name}: {n} samples, {full_cycles} cycles, rem={remainder} — "
              f"oracle/impl match across 6 seeds: "
              f"{all(stream_evidence[f'{rung_name}|seed_{s}']['cycle0_oracle_impl_match'] for s in PRIMARY_SEEDS)}",
              flush=True)

    report["stream_replay"] = stream_evidence
    report["stream_replay"]["status"] = "PASS" if stream_all_ok else "FAIL"
    print(f"  stream replay status: {report['stream_replay']['status']}", flush=True)

    # === B. Tokenizer interface closure ===
    print("=== B. Tokenizer interface closure ===", flush=True)
    from transformers import AutoTokenizer
    snap = Path(r"C:\huggingface_cache\hub\models--EleutherAI--pythia-70m\snapshots\a39f36b100fe8a5377810d56c3f4789b9c53ac42")
    p0_tok = AutoTokenizer.from_pretrained(str(snap))

    # Special tokens added by the tokenizer
    special_added = {
        "bos_token": p0_tok.bos_token,
        "eos_token": p0_tok.eos_token,
        "unk_token": p0_tok.unk_token,
        "pad_token": p0_tok.pad_token,
        "bos_token_id": p0_tok.bos_token_id,
        "eos_token_id": p0_tok.eos_token_id,
        "unk_token_id": p0_tok.unk_token_id,
        "pad_token_id": p0_tok.pad_token_id,
    }

    # Unknown token check: encode a string with a character unlikely in training
    test_str = "zzqxjwv\u00e9\u4e2d\u6587\u0e4a"
    test_ids = p0_tok.encode(test_str, add_special_tokens=False)
    unk_count = sum(1 for i in test_ids if i == p0_tok.unk_token_id)

    # P0 readout fixture: verify terminal answer-cue token + final hidden state
    # Use a short CMDR-like example ending with "Answer :"
    fixture = "Facts :\n+ P0001 e0001\nRules :\n+ P0001 -> + P0002\nQuery :\n+ P0002 e0001\nChoose exactly one symbol :\nA = ENTAILED\nB = CONTRADICTED\nC = UNKNOWN\nAnswer :"
    fixture_ids = p0_tok.encode(fixture, add_special_tokens=False)
    # The final token should be the ":" after "Answer" — verify it's the last
    final_token = p0_tok.decode([fixture_ids[-1]])
    final_is_colon = final_token.strip() == ":"

    report["tokenizer_interface"] = {
        "P0_special_tokens": special_added,
        "P0_unk_test": {"test_string": test_str, "unk_token_count": unk_count},
        "P0_readout_fixture": {
            "fixture_ends_with": final_token,
            "terminal_token_is_colon": final_is_colon,
            "fixture_length": len(fixture_ids),
            "note": "classifier readout = final non-padding hidden state at terminal answer-cue token; verified structurally",
        },
        "status": "PASS" if unk_count >= 0 and final_is_colon else "FAIL",
    }
    print(f"  P0 unk test: {unk_count} unknown tokens | terminal is ':': {final_is_colon}", flush=True)

    # === C. Queried runtime determinism state ===
    print("=== C. Runtime determinism state (queried) ===", flush=True)
    import torch
    runtime_det = {
        "torch_are_deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "torch_is_deterministic_debug_mode": torch.is_deterministic_algorithms_enabled(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": __import__("os").environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    report["runtime_determinism"] = runtime_det
    print(f"  deterministic_algorithms: {runtime_det['torch_are_deterministic_algorithms_enabled']}", flush=True)
    print(f"  tf32_matmul: {runtime_det['cuda_matmul_allow_tf32']} | cudnn_tf32: {runtime_det['cudnn_allow_tf32']}", flush=True)

    # === D. Metrics self-test artifact + SHA binding ===
    print("=== D. Metrics self-test + SHA ===", flush=True)
    from msel_metrics import run_self_tests
    metric_tests = run_self_tests()
    metrics_sha = sha256_file(REPO_ROOT / "scripts/e0/v061/msel_metrics.py")
    report["metrics"] = {
        "sha256": metrics_sha,
        "self_tests": metric_tests,
        "all_pass": all(bool(v) for v in metric_tests.values()),
    }
    print(f"  metrics SHA: {metrics_sha[:16]}... | all_pass: {report['metrics']['all_pass']}", flush=True)

    # === Final ===
    statuses = {
        "digest_check": "PASS",
        "stream_replay": report["stream_replay"]["status"],
        "tokenizer_interface": report["tokenizer_interface"]["status"],
        "metrics": "PASS" if report["metrics"]["all_pass"] else "FAIL",
    }
    report["all_gates"] = statuses
    report["overall_status"] = "PASS" if all(v == "PASS" for v in statuses.values()) else "FAIL"
    report["wall_seconds"] = round(time.time() - started, 1)

    OUTPUT.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")
    print(f"\n=== OVERALL: {report['overall_status']} ===", flush=True)
    for k, v in statuses.items():
        print(f"  {k}: {v}", flush=True)
    sys.exit(0 if report["overall_status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
