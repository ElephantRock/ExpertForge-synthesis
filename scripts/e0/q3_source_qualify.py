"""E0 Q3 — S0 source qualification harness (frozen registry implementation).

Usage:
  python scripts/e0/q3_source_qualify.py --smoke            # tokenizers (all) + S0C0 model interface
  python scripts/e0/q3_source_qualify.py --smoke --candidate S0C1   # add full model load check
  python scripts/e0/q3_source_qualify.py --tokenizer-check  # tokenizer A/B/C checks only, all candidates
  (capability/headroom and cost-benchmark full runs are gated on Q2 results; see --help)

Frozen contract: docs/experiments/e0/Q2_Q3_EXECUTION_RELEASE.md
Frozen registry: docs/experiments/e0/S0_CANDIDATE_REGISTRY_FROZEN.yaml

Interpretation notes — RESOLVED by project authority (Q2_FULL_RUN_RELEASE.md):
  - R3 (answer context): source content ends with "Answer:" and no trailing
    newline; the ONLY authorized continuations are the exact strings
    " A", " B", " C" — no fallback to bare letters.
  - R4 (replay subset): within each (label, depth) cell of eval_ID, rank by
    ascending sha256("ExpertForge-E0-Q3|replay|<sample_id>") and take the
    first 20 (240 total).
  - Source prompt = Q1-canonical Facts/Rules/Query lines followed by the exact
    frozen decision suffix from Q2_Q3_EXECUTION_RELEASE.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = (Path(__file__).resolve().parent.parent.parent).resolve()
REGISTRY_PATH = REPO_ROOT / "docs" / "experiments" / "e0" / "S0_CANDIDATE_REGISTRY_FROZEN.yaml"

LABELS = ["ENTAILED", "CONTRADICTED", "UNKNOWN"]
DECISION_SUFFIX_LINES = [
    "Choose exactly one symbol:",
    "A = ENTAILED",
    "B = CONTRADICTED",
    "C = UNKNOWN",
    "Answer:",
]
MAX_GPU_BYTES = 11 * 1024**3          # frozen ceiling on measured GPU allocation
MAX_CPU_BYTES = 24 * 1024**3          # frozen ceiling on measured host allocation
ACCELERATE_GPU_PLACEMENT_BYTES = int(10.5 * 1024**3)  # placement budget; leaves activation headroom so the measured peak stays under the frozen ceiling
REPLAY_NAMESPACE = "ExpertForge-E0-Q3|replay"
TOL = 1e-5
BENCHMARK_REPS = 3


def cjson(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def render_lit(value: dict) -> str:
    return f"{value['sign']} {value['pred']} {value['term']}"


def render_rule(rule: dict) -> str:
    lhs = " & ".join(render_lit(x) for x in rule["premises"])
    return f"{lhs} -> {render_lit(rule['conclusion'])}"


def build_source_content(example: dict) -> str:
    """Q1-canonical logical content + the exact frozen decision suffix."""
    lines = ["Facts :"]
    lines += [render_lit(x) for x in example["facts"]]
    lines += ["Rules :"]
    lines += [render_rule(x) for x in example["rules"]]
    lines += ["Query :", render_lit(example["query"])]
    lines += DECISION_SUFFIX_LINES
    return "\n".join(lines)


def load_registry() -> dict:
    registry = yaml.safe_load(REGISTRY_PATH.read_text(encoding="utf-8"))
    if registry.get("status") != "FROZEN_FOR_QUALIFICATION_BOOTSTRAP":
        raise AssertionError("registry is not frozen")
    return registry


def load_eval_rows(data_root: Path, name: str = "eval_ID") -> list[dict]:
    with (data_root / f"{name}.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def replay_subset(rows: list[dict]) -> list[dict]:
    subset = []
    for depth in range(1, 5):
        for label in LABELS:
            cell = [r for r in rows if r["reasoning_depth_stratum"] == depth and r["gold_label"] == label]
            ranked = sorted(
                cell,
                key=lambda r: hashlib.sha256(f"{REPLAY_NAMESPACE}|{r['sample_id']}".encode()).hexdigest(),
            )
            if len(ranked) < 20:
                raise AssertionError(f"cell {label}|d{depth} has {len(ranked)} < 20 examples")
            subset.extend(ranked[:20])
    if len(subset) != 240:
        raise AssertionError(f"replay subset size {len(subset)} != 240")
    return subset


class SourceHarness:
    """Model + tokenizer + TPDS tap + semantic logits for one registry candidate."""

    def __init__(self, cand: dict, data_root: Path, load_model: bool = True) -> None:
        self.cand = cand
        self.repo = cand["repository_id"]
        self.rev = cand["immutable_revision"]
        self.registry_id = cand["registry_id"]
        self.data_root = data_root
        self.offload = cand["resource_precheck"]["host_offload_expected"] is True
        self.tokenizer = AutoTokenizer.from_pretrained(self.repo, revision=self.rev)
        self.model = None
        self.tpds_module_path: str | None = None
        self.captured: dict[str, torch.Tensor] = {}
        if load_model:
            self._load_model()

    def _load_model(self) -> None:
        if self.offload:
            # CPU side of the placement budget stays at the frozen 24 GiB ceiling;
            # weights are pinned in RAM (no disk paging is permitted).
            max_memory = {0: ACCELERATE_GPU_PLACEMENT_BYTES, "cpu": MAX_CPU_BYTES}
            self.model = AutoModelForCausalLM.from_pretrained(
                self.repo,
                revision=self.rev,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                max_memory=max_memory,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.repo,
                revision=self.rev,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
            )
        self.model.eval()
        self._check_no_disk_offload()
        self._attach_tpds_hook()

    def _check_no_disk_offload(self) -> None:
        device_map = getattr(self.model, "hf_device_map", None)
        if device_map and any(str(v) == "disk" for v in device_map.values()):
            raise RuntimeError(
                f"disk-backed weight paging detected in resolved device map: {device_map}"
            )

    def _attach_tpds_hook(self) -> None:
        out_proj = self.model.get_output_embeddings()
        if out_proj is None:
            raise RuntimeError("model has no output embeddings module")
        for name, module in self.model.named_modules():
            if module is out_proj:
                self.tpds_module_path = name
                break

        def pre_hook(module, args, kwargs):
            tensor = kwargs.get("input", args[0] if args else None)
            if tensor is not None:
                self.captured["tpds"] = tensor.detach()
            return None

        out_proj.register_forward_pre_hook(pre_hook, with_kwargs=True)
        self.out_proj = out_proj

    # -- prompting ---------------------------------------------------------

    def templated_text(self, example: dict) -> str:
        content = build_source_content(example)
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=False,
        )

    def templated_ids(self, example: dict) -> list[int]:
        return self.tokenizer.encode(self.templated_text(example), add_special_tokens=False)

    def abc_contextual_tokens(self, probe_example: dict) -> dict:
        """R3 (Q2_FULL_RUN_RELEASE.md): the only authorized continuations are
        the exact UTF-8 strings " A", " B", " C" at the chat-templated answer
        context — no fallback forms. Tokenize(base + continuation) must append
        exactly one token over tokenize(base), without resegmenting the prompt.
        """
        base_text = self.templated_text(probe_example)
        base_ids = self.tokenizer.encode(base_text, add_special_tokens=False)
        ids_by_letter: dict[str, int] = {}
        for letter in "ABC":
            full_ids = self.tokenizer.encode(base_text + f" {letter}", add_special_tokens=False)
            appended = full_ids[len(base_ids) :]
            if len(appended) != 1 or full_ids[: len(base_ids)] != base_ids:
                return {}
            ids_by_letter[letter] = appended[0]
        if len(set(ids_by_letter.values())) != 3:
            return {}
        return {
            "answer_form": " <L>",
            "token_ids": dict(ids_by_letter),
            "token_pieces": {
                l: self.tokenizer.convert_ids_to_tokens(i) for l, i in ids_by_letter.items()
            },
            "token_decoded": {
                l: self.tokenizer.decode([i], skip_special_tokens=False)
                for l, i in ids_by_letter.items()
            },
        }

    # -- forward / TPDS ----------------------------------------------------

    @torch.no_grad()
    def forward_terminal(self, example: dict) -> dict:
        ids = torch.tensor([self.templated_ids(example)], device=self.model.device)
        self.captured.clear()
        logits = self.model(ids).logits[0, -1, :].detach()
        tpds = self.captured.get("tpds")
        if tpds is None:
            raise RuntimeError("TPDS pre-hook did not capture the projection input")
        tpds_terminal = tpds[0, -1, :].detach()
        return {
            "prompt_tokens": int(ids.shape[1]),
            "logits": logits,
            "tpds_terminal": tpds_terminal,
            "tpds_hidden_width": int(tpds_terminal.shape[-1]),
        }

    @torch.no_grad()
    def tpds_replay_check(self, example: dict) -> dict:
        out = self.forward_terminal(example)
        # Replay the FULL captured projection input with the same (1, T, H) GEMM
        # shape the model forward used; comparing a (1, 1, H) product can differ
        # by one BF16 ULP on split-device candidates (different kernel tiling).
        full_tpds = self.captured["tpds"]
        replayed = self.out_proj(full_tpds)[0, -1, :]
        max_abs = float((replayed - out["logits"]).abs().max())
        return {
            "max_abs_diff": max_abs,
            "atol": TOL,
            "rtol": TOL,
            "pass": bool(torch.allclose(replayed, out["logits"], atol=TOL, rtol=TOL)),
            "tpds_hidden_width": out["tpds_hidden_width"],
            "tpds_sequence_shape": list(full_tpds.shape),
        }

    def semantic_logits(self, out: dict, abc_ids: dict) -> torch.Tensor:
        return torch.stack([out["logits"][abc_ids[l]] for l in "ABC"])

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def static_config_check(model, cand: dict) -> dict:
    cfg = model.config
    checks = {
        "hidden_size": (cfg.hidden_size, cand["hidden_size"]),
        "num_hidden_layers": (cfg.num_hidden_layers, cand["num_hidden_layers"]),
        "num_attention_heads": (cfg.num_attention_heads, cand["num_attention_heads"]),
        "num_key_value_heads": (
            getattr(cfg, "num_key_value_heads", cand["num_key_value_heads"]),
            cand["num_key_value_heads"],
        ),
    }
    return {
        "fields": {k: {"actual": v[0], "registry": v[1]} for k, v in checks.items()},
        "ok": all(v[0] == v[1] for v in checks.values()),
        "param_count": sum(p.numel() for p in model.parameters()),
    }


def deterministic_replay_check(harness: SourceHarness, rows: list[dict], abc_ids: dict) -> dict:
    mismatches = []
    max_tpds_diff = 0.0
    max_logit_diff = 0.0
    for row in rows:
        a = harness.forward_terminal(row)
        b = harness.forward_terminal(row)
        tpds_diff = float((a["tpds_terminal"].float() - b["tpds_terminal"].float()).abs().max())
        sem_a = harness.semantic_logits(a, abc_ids).float()
        sem_b = harness.semantic_logits(b, abc_ids).float()
        logit_diff = float((sem_a - sem_b).abs().max())
        max_tpds_diff = max(max_tpds_diff, tpds_diff)
        max_logit_diff = max(max_logit_diff, logit_diff)
        if tpds_diff > TOL or logit_diff > TOL or sem_a.argmax().item() != sem_b.argmax().item():
            mismatches.append(row["sample_id"])
    return {
        "examples": len(rows),
        "max_tpds_abs_diff": max_tpds_diff,
        "max_semantic_logit_abs_diff": max_logit_diff,
        "argmax_mismatches": len(mismatches),
        "pass": len(mismatches) == 0 and max_tpds_diff <= TOL and max_logit_diff <= TOL,
    }


def run_tokenizer_check(cand: dict, data_root: Path) -> dict:
    try:
        harness = SourceHarness(cand, data_root, load_model=False)
        probe = load_eval_rows(data_root)[0]
        abc = harness.abc_contextual_tokens(probe)
        ok = bool(abc) and len(set(abc["token_ids"].values())) == 3
        return {
            "registry_id": cand["registry_id"],
            "ok": ok,
            "error": None,
            "abc": abc,
            "chat_template_available": harness.tokenizer.chat_template is not None,
        }
    except Exception as exc:  # noqa: BLE001 — record access/runtime failures honestly
        return {
            "registry_id": cand["registry_id"],
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "abc": None,
            "chat_template_available": None,
        }


def run_model_interface_check(cand: dict, data_root: Path) -> dict:
    try:
        harness = SourceHarness(cand, data_root, load_model=True)
        rows = load_eval_rows(data_root)
        probe_rows = [rows[0], rows[len(rows) // 2], rows[-1]]
        torch.cuda.reset_peak_memory_stats(0)
        static = static_config_check(harness.model, cand)
        abc = harness.abc_contextual_tokens(probe_rows[0])
        replay = harness.tpds_replay_check(probe_rows[0])
        determinism = deterministic_replay_check(harness, probe_rows, abc["token_ids"])
        cuda_peak = int(torch.cuda.max_memory_allocated(0))
        host_peak = None
        try:
            import psutil

            host_peak = psutil.Process().memory_info().peak_wset
        except Exception:
            host_peak = None
        resolved_device_map = {
            k: str(v) for k, v in (getattr(harness.model, "hf_device_map", {"": "cuda:0"}) or {}).items()
        }
        harness.close()
        resource_feasible = True
        resource_notes = []
        if cand["resource_precheck"]["host_offload_expected"] is True:
            if cuda_peak > MAX_GPU_BYTES:
                resource_feasible = False
                resource_notes.append(
                    f"measured CUDA peak {cuda_peak} exceeds frozen 11 GiB ceiling"
                )
            if host_peak is not None and host_peak > MAX_CPU_BYTES:
                resource_feasible = False
                resource_notes.append(
                    f"measured host peak {host_peak} exceeds frozen 24 GiB ceiling"
                )
        ok = (
            static["ok"]
            and bool(abc)
            and replay["pass"]
            and determinism["pass"]
            and replay["tpds_hidden_width"] >= 256
            and resource_feasible
        )
        return {
            "registry_id": cand["registry_id"],
            "ok": ok,
            "error": None,
            "static_config": static,
            "tpds_module_path": harness.tpds_module_path,
            "resolved_device_map": resolved_device_map,
            "resource_feasible": resource_feasible,
            "resource_ceiling_notes": resource_notes,
            "host_peak_memory_bytes": host_peak,
            "abc": abc,
            "tpds_replay": replay,
            "two_pass_determinism": determinism,
            "cuda_peak_allocated_bytes": cuda_peak,
            "execution_dtype": "bfloat16",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "registry_id": cand["registry_id"],
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def git_head() -> str:
    import subprocess

    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--tokenizer-check", action="store_true")
    parser.add_argument("--candidate", choices=["S0C0", "S0C1", "S0C2"], action="append")
    parser.add_argument(
        "--data-root", type=Path, default=Path("local_data/e0_qualification_bootstrap/Q1")
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if not (args.smoke or args.tokenizer_check):
        parser.error("capability/headroom and benchmark full runs are gated on Q2 results; use --smoke or --tokenizer-check")

    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    registry = load_registry()
    candidates = {c["registry_id"]: c for c in registry["candidates"]}
    data_root = args.data_root.resolve()

    report: dict = {
        "schema_id": "E0-Q3-SMOKE-v0" if args.smoke else "E0-Q3-TOKENIZER-CHECK-v0",
        "git_commit": git_head(),
        "non_scientific": True,
        "tokenizer_checks": {},
        "model_interface_checks": {},
    }

    if args.smoke or args.tokenizer_check:
        for rid, cand in candidates.items():
            report["tokenizer_checks"][rid] = run_tokenizer_check(cand, data_root)

    if args.smoke:
        wanted = args.candidate or ["S0C0"]
        for rid in wanted:
            report["model_interface_checks"][rid] = run_model_interface_check(candidates[rid], data_root)

    tok_ok = all(v["ok"] for v in report["tokenizer_checks"].values())
    model_checks = report.get("model_interface_checks", {})
    model_ok = all(v["ok"] for v in model_checks.values()) if model_checks else None
    report["status"] = "PASS" if (tok_ok and (model_ok is not False)) else "FAIL"
    report["status_detail"] = {
        "tokenizers_all_ok": tok_ok,
        "model_checks_ok": model_ok,
    }

    out_path = args.out or (
        REPO_ROOT
        / "docs"
        / "experiments"
        / "e0"
        / ("q3_smoke_manifest.json" if args.smoke else "q3_tokenizer_check.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
