"""V06-PREEXEC-IMPLEMENTATION-1 comprehensive evidence bundle.

Runs all CPU-bound pre-execution gates in one pass:
  1. Full verifier + counterfactual_invariance over all 134k frozen families
  2. ExampleStream validation (cycle properties, boundary math)
  3. Metric self-tests
  4. Decision program r1 digest + oracle + report schema digest
  5. Seed/substream recomputation from normative namespaces
  6. Tokenizer audit (P0 native + CMDR-Lex-v1)
  7. Runtime snapshot
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
DOCS = REPO_ROOT / "docs" / "experiments" / "e0" / "v061"
OUTPUT = DOCS / "PREEXEC_BUNDLE.json"
CORPUS = REPO_ROOT / "local_data" / "e0_v061_msel"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    started = time.time()
    report = {
        "schema_id": "E0-V061-PREEXEC-BUNDLE-v0",
        "authority": "V06-PREEXEC-IMPLEMENTATION-1",
        "code_git_commit": git_head(),
        "modules": {},
    }

    # --- Module digests ---
    for mod in ("msel_corpus.py", "msel_verifier.py", "structsig_r3.py",
                "msel_stream.py", "msel_metrics.py"):
        p = REPO_ROOT / "scripts" / "e0" / "v061" / mod
        if p.exists():
            report["modules"][mod] = sha256_file(p)

    # --- Item 1: Full verifier + counterfactual_invariance ---
    print("=== 1. Full verifier + counterfactual ===", flush=True)
    from msel_verifier import independent_verify
    from msel_corpus import LABELS
    t0 = time.time()

    fam_variants = defaultdict(list)
    for split in ("train_pool", "dev_ID", "eval_ID", "eval_STRUCT"):
        for dep in (1, 2, 3, 4):
            with open(CORPUS / f"{split}_d{dep}.jsonl", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    fam_variants[row["family_id"]].append(row)

    verifier_errors = 0
    cf_invariance_errors = 0
    for fid, variants in fam_variants.items():
        if sorted(v["gold_label"] for v in variants) != sorted(LABELS):
            verifier_errors += 1
            continue
        for v in variants:
            label, proof_depth, _, _ = independent_verify(v)
            if label != v["gold_label"]:
                verifier_errors += 1
            if label != "UNKNOWN" and proof_depth != v["reasoning_depth_stratum"]:
                verifier_errors += 1
        # full counterfactual invariance (unigram + bigram)
        unigrams = [Counter(v["rendered"].split()) for v in variants]
        if not (unigrams[0] == unigrams[1] == unigrams[2]):
            cf_invariance_errors += 1
        toks = [v["rendered"].split() for v in variants]
        bigrams = [Counter(zip(t, t[1:])) for t in toks]
        if not (bigrams[0] == bigrams[1] == bigrams[2]):
            cf_invariance_errors += 1

    report["item_1_full_verifier"] = {
        "families": len(fam_variants),
        "verifier_errors": verifier_errors,
        "cf_invariance_errors": cf_invariance_errors,
        "status": "PASS" if verifier_errors == 0 and cf_invariance_errors == 0 else "FAIL",
        "seconds": round(time.time() - t0, 1),
    }
    print(f"  {len(fam_variants)} families | verifier_errors={verifier_errors} | cf_errors={cf_invariance_errors}", flush=True)

    # --- Item 2: ExampleStream validation ---
    print("=== 2. ExampleStream ===", flush=True)
    from msel_stream import MSELExampleStream, cycle_order, validate_stream, PRESENTATIONS

    # Get R1 sample IDs
    rungs = json.loads((CORPUS / "MSEL_RUNGS.json").read_text(encoding="utf-8"))
    r1_d1_fams = set(rungs["family_ids"]["d1"]["R1"])
    r1_sample_ids = sorted(
        v["sample_id"] for fid, variants in fam_variants.items()
        if fid in r1_d1_fams for v in variants
    )
    stream_val = validate_stream(1647674144, r1_sample_ids)
    report["item_2_stream"] = {
        "r1_sample_count": len(r1_sample_ids),
        "validation": stream_val,
        "status": "PASS" if stream_val["valid"] else "FAIL",
    }
    # quick stream prefix check
    s = MSELExampleStream(r1_sample_ids, 1647674144)
    prefix = s.take(128)
    assert len(prefix) == 128 and len(set(prefix)) <= len(r1_sample_ids)
    # verify prefix is a prefix of the first cycle order
    first_cycle = cycle_order(1647674144, 0, r1_sample_ids)
    if len(r1_sample_ids) >= 128:
        assert prefix == first_cycle[:128]
        report["item_2_stream"]["prefix_matches_cycle0"] = True
    print(f"  R1 samples: {len(r1_sample_ids)} | stream valid: {stream_val['valid']}", flush=True)

    # --- Item 3: Metric self-tests ---
    print("=== 3. Metrics ===", flush=True)
    from msel_metrics import run_self_tests
    metric_tests = run_self_tests()
    report["item_3_metrics"] = {
        "self_tests": metric_tests,
        "status": "PASS" if all(metric_tests.values()) else "FAIL",
    }
    print(f"  all pass: {all(metric_tests.values())}", flush=True)

    # --- Item 4: Decision program + schema ---
    print("=== 4. Decision program + schema ===", flush=True)
    r1_path = DOCS / "e0_v061_mechanism_decision_r1.py"
    r1_sha = sha256_file(r1_path)
    expected_r1 = "ab2fb6f69948f50aa44a1ae0b53a9f6949084ca9f30f4be7fc3885a8d1efb54"
    schema_path = DOCS / "E0_v0.6.1_MSEL_REPORT_SCHEMA.json"
    schema_sha = sha256_file(schema_path)
    expected_schema = "933731003536c4fb91318ad44abb830e917eb3e1ed00b2f71ef83bac67ce8100"
    # run oracle
    oracle = subprocess.run([sys.executable, str(r1_path)], capture_output=True, text=True, timeout=60)
    report["item_4_decision"] = {
        "r1_sha256_recomputed": r1_sha,
        "r1_sha256_matches": r1_sha == expected_r1,
        "oracle_exit": oracle.returncode,
        "oracle_output": json.loads(oracle.stdout.strip()) if oracle.returncode == 0 else None,
        "schema_sha256_recomputed": schema_sha,
        "schema_sha256_matches": schema_sha == expected_schema,
        "status": "PASS" if r1_sha == expected_r1 and oracle.returncode == 0 and schema_sha == expected_schema else "FAIL",
    }
    print(f"  r1 match: {r1_sha == expected_r1} | oracle: {oracle.returncode} | schema match: {schema_sha == expected_schema}", flush=True)

    # --- Item 5: Seeds ---
    print("=== 5. Seeds ===", flush=True)
    def uint31(s):
        return int.from_bytes(hashlib.sha256(s.encode()).digest()[:4], "big") & 0x7fffffff

    contract_primary = [806915476, 1031646469, 128439691, 555223894, 454204619, 1678768041]
    contract_comp = [1228139313, 1536284461, 293488859, 1941939586, 1046059599, 920620107]
    seeds_match = all(uint31(f"ExpertForge-E0-v06-msel-seed|{i}") == v for i, v in enumerate(contract_primary))
    seeds_match &= all(uint31(f"ExpertForge-E0-v06-msel-seed|{i}") == v for i, v in enumerate(contract_comp, start=6))
    seeds_match &= uint31("ExpertForge-E0-v06-msel-bootstrap|0") == 1611111118
    substreams = {}
    for seed in contract_primary:
        for sub in ("backbone_init", "classifier_init", "data_order", "dataloader_workers"):
            substreams[f"{seed}|{sub}"] = uint31(f"ExpertForge-E0-v061-rng|{seed}|{sub}")
    report["item_5_seeds"] = {
        "all_seeds_match": seeds_match,
        "substreams_derived": len(substreams),
        "substream_sample": dict(list(substreams.items())[:4]),
        "status": "PASS" if seeds_match else "FAIL",
    }
    print(f"  all match: {seeds_match} | substreams: {len(substreams)}", flush=True)

    # --- Item 6: Tokenizer audit ---
    print("=== 6. Tokenizer audit ===", flush=True)
    tok_results = {}

    # P0 native tokenizer
    from transformers import AutoTokenizer
    snap = Path(r"C:\huggingface_cache\hub\models--EleutherAI--pythia-70m\snapshots\a39f36b100fe8a5377810d56c3f4789b9c53ac42")
    p0_tok = AutoTokenizer.from_pretrained(str(snap))
    p0_lengths = []
    p0_max_id = 0
    p0_truncations = 0
    p0_invalid = 0
    # sample from eval_ID (representative, not all 402k)
    sample_rows = []
    for dep in (1, 2, 3, 4):
        with open(CORPUS / f"eval_ID_d{dep}.jsonl", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < 250:
                    sample_rows.append(json.loads(line))
                else:
                    break
    for row in sample_rows:
        ids = p0_tok.encode(row["rendered"], add_special_tokens=False)
        p0_lengths.append(len(ids))
        p0_max_id = max(p0_max_id, max(ids))
        if len(ids) > 384:
            p0_truncations += 1
        if any(i >= 50304 for i in ids):
            p0_invalid += 1
    tok_results["P0"] = {
        "tokenizer_len": len(p0_tok),
        "vocab_size_declared": p0_tok.vocab_size,
        "sample_count": len(sample_rows),
        "token_length_min": min(p0_lengths),
        "token_length_median": sorted(p0_lengths)[len(p0_lengths)//2],
        "token_length_p95": sorted(p0_lengths)[int(len(p0_lengths)*0.95)],
        "token_length_max": max(p0_lengths),
        "max_token_id": p0_max_id,
        "all_ids_below_50304": p0_invalid == 0,
        "truncations_at_384": p0_truncations,
    }

    # CMDR-Lex-v1
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "e0"))
    from q1_cmdr_bootstrap import LEX
    lex_lengths = []
    for row in sample_rows:
        ids = LEX.encode(row["rendered"], append_decide=True)
        lex_lengths.append(len(ids))
    tok_results["CMDR_Lex_v1"] = {
        "vocab_size": len(LEX.tokens),
        "sample_count": len(sample_rows),
        "token_length_min": min(lex_lengths),
        "token_length_median": sorted(lex_lengths)[len(lex_lengths)//2],
        "token_length_max": max(lex_lengths),
        "all_below_384": max(lex_lengths) <= 384,
    }
    tok_pass = tok_results["P0"]["all_ids_below_50304"] and tok_results["P0"]["truncations_at_384"] == 0 \
                and tok_results["CMDR_Lex_v1"]["all_below_384"]
    report["item_6_tokenizers"] = {
        "results": tok_results,
        "status": "PASS" if tok_pass else "FAIL",
    }
    print(f"  P0 max_len: {tok_results['P0']['token_length_max']} | max_id: {p0_max_id} | "
          f"truncations: {p0_truncations} | Lex max: {tok_results['CMDR_Lex_v1']['token_length_max']}", flush=True)

    # --- Item 7: Runtime snapshot ---
    print("=== 7. Runtime snapshot ===", flush=True)
    import igraph
    runtime = {
        "os": platform.platform(),
        "python_version": sys.version.split()[0],
        "python_implementation": sys.implementation.name,
        "igraph_version": igraph.__version__,
    }
    try:
        import torch
        runtime["torch_version"] = torch.__version__
        runtime["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            runtime["gpu_name"] = torch.cuda.get_device_name(0)
            runtime["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory
            runtime["bf16_supported"] = torch.cuda.is_bf16_supported()
    except ImportError:
        runtime["torch"] = "not installed"
    try:
        import transformers
        runtime["transformers_version"] = transformers.__version__
    except ImportError:
        pass
    try:
        import numpy
        runtime["numpy_version"] = numpy.__version__
    except ImportError:
        pass
    try:
        import scipy
        runtime["scipy_version"] = scipy.__version__
    except ImportError:
        pass
    try:
        import sklearn
        runtime["sklearn_version"] = sklearn.__version__
    except ImportError:
        pass
    # pip freeze digest
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    runtime["pip_freeze_sha256"] = hashlib.sha256(freeze.stdout.encode()).hexdigest()
    report["item_7_runtime"] = runtime
    print(f"  torch: {runtime.get('torch_version', 'N/A')} | igraph: {runtime['igraph_version']}", flush=True)

    # --- Final ---
    all_items = [report[f"item_{i}_{k}"]["status"]
                 for i in range(1, 8) for k in []]  # won't work, do manually
    statuses = [
        report["item_1_full_verifier"]["status"],
        report["item_2_stream"]["status"],
        report["item_3_metrics"]["status"],
        report["item_4_decision"]["status"],
        report["item_5_seeds"]["status"],
        report["item_6_tokenizers"]["status"],
    ]
    report["all_cpu_gates"] = {f"item_{i+1}": s for i, s in enumerate(statuses)}
    report["overall_status"] = "PASS" if all(s == "PASS" for s in statuses) else "FAIL"
    report["wall_seconds"] = round(time.time() - started, 1)

    OUTPUT.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")
    print(f"\n=== OVERALL: {report['overall_status']} ===", flush=True)
    for k, v in report["all_cpu_gates"].items():
        print(f"  {k}: {v}", flush=True)
    sys.exit(0 if report["overall_status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
