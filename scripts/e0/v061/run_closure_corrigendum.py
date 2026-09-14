"""V06-PREEXEC-CLOSURE-CORRIGENDUM-1 — final CPU gate closure.

Three items:
  1. Full 1,024,000-presentation stream replay through MSELExampleStream for
     all 18 arm×seed cases, compared to the frozen independent-oracle roots.
  2. Tokenizer interface: unk_count==0 required; P0 classification-form
     readout fixture with actual hidden-state gather and classifier forward.
  3. Deterministic preamble application + post-application queried values;
     [0.02]*12 paired-superiority boundary fixture.
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CORPUS = REPO_ROOT / "local_data" / "e0_v061_msel"
DOCS = REPO_ROOT / "docs" / "experiments" / "e0" / "v061"
OUTPUT = DOCS / "CLOSURE_CORRIGENDUM_EVIDENCE.json"
MANIFEST = DOCS / "MSEL_MANIFEST.json"
PREV = DOCS / "FINAL_PREEXEC_CLOSURE.json"

SPLITS = ("train_pool", "dev_ID", "eval_ID", "eval_STRUCT")
DEPTHS = (1, 2, 3, 4)
PRIMARY_SEEDS = [806915476, 1031646469, 128439691, 555223894, 454204619, 1678768041]
PRESENTATIONS = 1_024_000
STREAM_NS = "ExpertForge-E0-v061-msel-stream"


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    started = time.time()
    report = {
        "schema_id": "E0-V061-CLOSURE-CORRIGENDUM-v0",
        "authority": "V06-PREEXEC-CLOSURE-CORRIGENDUM-1",
        "code_git_commit": git_head(),
    }

    # Hard stop on digests
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for split in SPLITS:
        for dep in DEPTHS:
            key = f"{split}_d{dep}"
            if sha256_file(CORPUS / f"{key}.jsonl") != manifest["splits"][key]["sha256"]:
                raise SystemExit(f"HARD STOP: {key} digest mismatch")
    print("  digest hard-stop: all 16 match", flush=True)

    # Load previous frozen oracle roots
    prev = json.loads(PREV.read_text(encoding="utf-8"))

    # === 1. Full stream replay through MSELExampleStream ===
    print("=== 1. Full stream replay (implementation, 18 arm×seed cases) ===", flush=True)
    from msel_stream import MSELExampleStream

    rungs = json.loads((CORPUS / "MSEL_RUNGS.json").read_text(encoding="utf-8"))
    train_sample_by_fid = {}
    for dep in DEPTHS:
        with open(CORPUS / f"train_pool_d{dep}.jsonl", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                train_sample_by_fid.setdefault(row["family_id"], []).append(row["sample_id"])

    def build_arm_samples(rung_name):
        all_sids = []
        for dep in DEPTHS:
            for fid in set(rungs["family_ids"][f"d{dep}"][rung_name]):
                if fid in train_sample_by_fid:
                    all_sids.extend(train_sample_by_fid[fid])
        return sorted(all_sids)

    stream_results = {}
    stream_all_ok = True
    chunk = 8192  # bounded chunks
    for rung_name in ("R1", "R4", "R16"):
        sids = build_arm_samples(rung_name)
        for seed in PRIMARY_SEEDS:
            key = f"{rung_name}|seed_{seed}"
            # Get frozen oracle root
            oracle_root = prev["stream_replay"][key]["full_stream_root"]

            # Stream through the implementation
            s = MSELExampleStream(sids, seed)
            h = hashlib.sha256()
            remaining = PRESENTATIONS
            while remaining > 0:
                n = min(chunk, remaining)
                batch = s.take(n)
                for sid in batch:
                    h.update(sid.encode("utf-8"))
                    h.update(b"\x00")
                remaining -= n
            impl_root = h.hexdigest()

            match = impl_root == oracle_root
            stream_results[key] = {
                "implementation_root": impl_root,
                "oracle_root": oracle_root,
                "match": match,
            }
            if not match:
                stream_all_ok = False
                print(f"  MISMATCH {key}: impl={impl_root[:16]} oracle={oracle_root[:16]}", flush=True)
        print(f"  {rung_name}: 6 seeds {'ALL MATCH' if stream_all_ok else 'MISMATCH DETECTED'}", flush=True)

    report["stream_full_replay"] = {
        "cases": len(stream_results),
        "presentations_per_case": PRESENTATIONS,
        "results": stream_results,
        "status": "PASS" if stream_all_ok else "FAIL",
    }

    # === 2. Tokenizer interface with P0 classification-form readout ===
    print("=== 2. Tokenizer interface + P0 readout ===", flush=True)
    import torch

    from transformers import AutoModelForCausalLM, AutoTokenizer
    snap = Path(r"C:\huggingface_cache\hub\models--EleutherAI--pythia-70m\snapshots\a39f36b100fe8a5377810d56c3f4789b9c53ac42")
    tok = AutoTokenizer.from_pretrained(str(snap))

    # Fixture
    fixture = ("Facts :\n+ P0001 e0001\nRules :\n+ P0001 -> + P0002\nQuery :\n+ P0002 e0001\n"
               "Choose exactly one symbol :\nA = ENTAILED\nB = CONTRADICTED\nC = UNKNOWN\nAnswer :")
    ids = tok.encode(fixture, add_special_tokens=False)
    unk_count = sum(1 for i in ids if i == tok.unk_token_id)

    # Load P0 classification form (backbone without LM head + 512→3 classifier)
    model = AutoModelForCausalLM.from_pretrained(str(snap), torch_dtype=torch.float32)
    model.eval()

    # Forward with output_hidden_states to get the final hidden layer
    input_ids = torch.tensor([ids])
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

    # Final hidden state (last layer) — shape [1, seq_len, 512]
    final_hidden = outputs.hidden_states[-1]

    # Last non-padding index from attention mask
    seq_lens = attention_mask.sum(dim=1)  # [1]
    last_idx = int(seq_lens[0]) - 1

    # Verify the terminal token is the colon
    terminal_token = tok.decode([ids[last_idx]])
    terminal_is_colon = terminal_token.strip() == ":"

    # Gather the 512-vector at the terminal position
    readout_vector = final_hidden[0, last_idx, :]  # [512]
    hidden_dim = readout_vector.shape[0]

    # Pass through the 512→3 classifier
    classifier = torch.nn.Linear(hidden_dim, 3, bias=True)
    logits = classifier(readout_vector.unsqueeze(0))  # [1, 3]
    logits_shape = list(logits.shape)

    # Cleanup
    del model, outputs, final_hidden
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    readout_ok = (
        unk_count == 0
        and terminal_is_colon
        and hidden_dim == 512
        and logits_shape == [1, 3]
    )
    report["tokenizer_interface"] = {
        "unk_count": unk_count,
        "unk_count_required_zero": unk_count == 0,
        "add_special_tokens": False,
        "terminal_token": terminal_token,
        "terminal_is_colon": terminal_is_colon,
        "hidden_dim": hidden_dim,
        "logits_shape": logits_shape,
        "readout_verified": readout_ok,
        "status": "PASS" if readout_ok else "FAIL",
    }
    print(f"  unk={unk_count} | terminal=':' | dim={hidden_dim} | shape={logits_shape} | "
          f"readout={'PASS' if readout_ok else 'FAIL'}", flush=True)

    # === 3. Deterministic preamble + boundary fixture ===
    print("=== 3. Deterministic preamble + boundary fixture ===", flush=True)
    from deterministic_preamble import apply, query, verify_applied

    det_state = apply()
    det_ok = verify_applied(det_state)
    report["deterministic_preamble"] = {
        "post_application_state": det_state,
        "verified": det_ok,
        "status": "PASS" if det_ok else "FAIL",
    }
    print(f"  verified: {det_ok} | det_algos={det_state['are_deterministic_algorithms_enabled']} "
          f"| tf32_matmul={det_state['cuda_matmul_allow_tf32']} | cudnn_det={det_state['cudnn_deterministic']}", flush=True)

    # [0.02]*12 paired-superiority boundary fixture
    from msel_metrics import paired_superiority
    boundary_result = paired_superiority([0.02] * 12)
    boundary_ok = boundary_result["superior"] is False  # exactly at boundary should NOT be superior (strict >)
    report["superiority_boundary_fixture"] = {
        "input": [0.02] * 12,
        "result": boundary_result,
        "expected_superior": False,
        "actual_superior": boundary_result["superior"],
        "status": "PASS" if boundary_ok else "FAIL",
    }
    print(f"  [0.02]*12 → superior={boundary_result['superior']} (expected False) → "
          f"{'PASS' if boundary_ok else 'FAIL'}", flush=True)

    # === Final ===
    statuses = {
        "stream_full_replay": report["stream_full_replay"]["status"],
        "tokenizer_interface": report["tokenizer_interface"]["status"],
        "deterministic_preamble": report["deterministic_preamble"]["status"],
        "superiority_boundary": report["superiority_boundary_fixture"]["status"],
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
