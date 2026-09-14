"""Six-seed cell summary + hierarchical bootstrap (§9.8) + frozen state gates (§10.2/10.3).

Authorized by V06-MECHANISM-PRIMARY-BATCH-EXECUTION-AUTHORIZED: after all six
primary seeds of a cell complete, compute the frozen aggregation. This tool
consumes the committed per-seed evidence reports plus the frozen corpus; it
recomputes Q from each report's stored prediction vectors and asserts exact
agreement with the reported metrics before aggregating (fail closed).

This file is NOT part of the training CODE_MANIFEST_FILES set — the canonical
training code_sha256 bd73dce3… is unaffected.

State determination (frozen):
  TRANSFER_PASS: median Q >= 0.55 AND median Q_2:4 >= 0.50 AND
                 median Q_STRUCT >= 0.50 AND median min_label_recall >= 0.45
                 AND every seed Q >= 0.50
  NO_TRANSFER:   median Q <= 0.40 AND median Q_STRUCT <= 0.40 AND
                 bootstrap 95% UCB (ci_97_5) < 0.45
  else:          INCONCLUSIVE_FOR_QUALIFICATION

DIVERGED_SCIENTIFIC seeds enter the point-estimate aggregation with their
§10.1 zero metrics; if any seed diverged, the family-level bootstrap cannot
resample its (absent) predictions and the bootstrap is withheld pending
authority guidance (recorded explicitly — fail closed, never fabricated).

Usage:
  .venv/Scripts/python.exe scripts/e0/v061/aggregate_cell_summary.py --cell R1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from deterministic_preamble import apply as apply_det  # side-effect-free on CPU

apply_det()

import numpy as np
import msel_metrics
from mechanism_evidence_harness import DEPTHS, LABEL_TO_ID, load_split

REPO_ROOT = HERE.parent.parent.parent
CORPUS = REPO_ROOT / "local_data" / "e0_v061_msel"
EVIDENCE_DIR = REPO_ROOT / "docs" / "experiments" / "e0" / "v061" / "evidence"
PRIMARY_SEEDS = (806915476, 1031646469, 128439691, 555223894, 454204619, 1678768041)


def cjson(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                          capture_output=True, text=True, check=True).stdout.strip()


def eval_rows(split: str) -> list[dict]:
    rows = []
    for dep in DEPTHS:
        with (CORPUS / f"{split}_d{dep}.jsonl").open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                rows.append({"label_id": LABEL_TO_ID[r["gold_label"]],
                             "depth": r["reasoning_depth_stratum"],
                             "family_id": r["family_id"]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", required=True,
                        choices=["R1", "R4", "R16", "P-FROZEN", "P-FT", "P-RANDOM"])
    args = parser.parse_args()
    cell = args.cell

    id_rows = eval_rows("eval_ID")
    struct_rows = eval_rows("eval_STRUCT")

    reports = []
    for seed in PRIMARY_SEEDS:
        path = EVIDENCE_DIR / cell / f"{cell}_seed{seed}.json"
        if not path.is_file():
            raise SystemExit(f"fail-closed: missing report {path}")
        r = json.loads(path.read_text(encoding="utf-8"))
        if r["regime_id"] != cell or r["master_seed"] != seed:
            raise SystemExit(f"fail-closed: {path} identity mismatch")
        if r["run_status"] not in ("VALID", "DIVERGED_SCIENTIFIC"):
            raise SystemExit(f"fail-closed: {path} inadmissible status {r['run_status']}")
        reports.append(r)

    per_seed = []
    per_seed_data = []
    diverged = []
    code_manifests = set()
    for r in reports:
        m = r["metrics"]
        preds = m.get("predictions")
        if r["run_status"] == "VALID":
            pred_id = preds["eval_ID_argmax"]
            pred_struct = preds["eval_STRUCT_argmax"]
            if len(pred_id) != len(id_rows) or len(pred_struct) != len(struct_rows):
                raise SystemExit(f"fail-closed: seed {r['master_seed']} prediction-vector length mismatch")
            y_i = [x["label_id"] for x in id_rows]
            d_i = [x["depth"] for x in id_rows]
            f_i = [x["family_id"] for x in id_rows]
            y_s = [x["label_id"] for x in struct_rows]
            d_s = [x["depth"] for x in struct_rows]
            f_s = [x["family_id"] for x in struct_rows]
            recomputed = msel_metrics.q_metrics(y_i, pred_id, d_i, y_s, pred_struct, d_s)
            if recomputed["Q"] != m["Q"] or recomputed["Q_ID"] != m["Q_ID"] \
                    or recomputed["Q_STRUCT"] != m["Q_STRUCT"]:
                raise SystemExit(
                    f"fail-closed: seed {r['master_seed']} stored predictions do not reproduce "
                    f"reported Q ({recomputed} vs Q={m['Q']})")
            per_seed_data.append({
                "y_true_id": y_i, "y_pred_id": pred_id, "depths_id": d_i, "family_ids_id": f_i,
                "y_true_struct": y_s, "y_pred_struct": pred_struct,
                "depths_struct": d_s, "family_ids_struct": f_s,
            })
        else:
            diverged.append(r["master_seed"])
        code_manifests.add(r["code_sha256"])
        per_seed.append({
            "master_seed": r["master_seed"],
            "run_status": r["run_status"],
            "selected_checkpoint_update": r.get("selected_checkpoint_update"),
            "Q": m["Q"], "Q_ID": m["Q_ID"], "Q_STRUCT": m["Q_STRUCT"],
            "Q_2_4": m["Q_2_4"], "minimum_label_recall": m["minimum_label_recall"],
            "family_exact_consistency": m["family_exact_consistency"],
            "wall_seconds": r["resource"]["wall_seconds"],
            "code_sha256": r["code_sha256"],
            "repository_commit": r["repository_commit"],
        })

    if len(code_manifests) != 1:
        raise SystemExit(f"fail-closed: cell {cell} reports span multiple code manifests: {code_manifests}")

    q_vals = [s["Q"] for s in per_seed]
    q24_vals = [s["Q_2_4"] for s in per_seed]
    qs_vals = [s["Q_STRUCT"] for s in per_seed]
    mlr_vals = [s["minimum_label_recall"] for s in per_seed]
    med = lambda v: float(np.median(v))
    summary = {
        "median_Q": med(q_vals),
        "median_Q_2_4": med(q24_vals),
        "median_Q_STRUCT": med(qs_vals),
        "median_minimum_label_recall": med(mlr_vals),
        "min_seed_Q": min(q_vals),
        "max_seed_Q": max(q_vals),
    }

    gates = {
        "TRANSFER_PASS": (
            summary["median_Q"] >= 0.55
            and summary["median_Q_2_4"] >= 0.50
            and summary["median_Q_STRUCT"] >= 0.50
            and summary["median_minimum_label_recall"] >= 0.45
            and summary["min_seed_Q"] >= 0.50),
        "NO_TRANSFER_gate_values": {
            "median_Q_le_0.40": summary["median_Q"] <= 0.40,
            "median_Q_STRUCT_le_0.40": summary["median_Q_STRUCT"] <= 0.40,
        },
    }

    bootstrap = None
    if diverged:
        bootstrap = {
            "status": "WITHHELD",
            "reason": f"diverged seeds {diverged} carry §10.1 zero metrics and no prediction "
                      "vectors; family-level bootstrap cannot resample them — held for "
                      "authority guidance rather than fabricated",
        }
        no_transfer_ucb = None
    else:
        boot = msel_metrics.hierarchical_bootstrap(list(PRIMARY_SEEDS), per_seed_data)
        bootstrap = {"status": "COMPUTED", **boot}
        no_transfer_ucb = boot["ci_97_5"]
        gates["NO_TRANSFER_gate_values"]["bootstrap_ucb_97_5_lt_0.45"] = bool(no_transfer_ucb < 0.45)

    if gates["TRANSFER_PASS"]:
        state = "TRANSFER_PASS"
    elif (not diverged
          and gates["NO_TRANSFER_gate_values"]["median_Q_le_0.40"]
          and gates["NO_TRANSFER_gate_values"]["median_Q_STRUCT_le_0.40"]
          and gates["NO_TRANSFER_gate_values"]["bootstrap_ucb_97_5_lt_0.45"]):
        state = "NO_TRANSFER"
    else:
        state = "INCONCLUSIVE_FOR_QUALIFICATION"

    evidence = {
        "schema_id": "E0-V061-MSEL-CELL-SIX-SEED-SUMMARY-v0",
        "authority": "V06-MECHANISM-PRIMARY-BATCH-EXECUTION-AUTHORIZED",
        "regime_id": cell,
        "code_git_commit": git_head(),
        "training_code_sha256": next(iter(code_manifests)),
        "per_seed": per_seed,
        "summary": summary,
        "gates": gates,
        "hierarchical_bootstrap": bootstrap,
        "transfer_state": state,
        "notes": [
            "frozen §10.2/§10.3 gates only — no interpretation beyond the frozen thresholds",
            "Q values recomputed from stored prediction vectors and asserted equal to "
            "reported metrics before aggregation",
            "aggregator itself is outside the training CODE_MANIFEST_FILES set",
        ],
    }
    out = EVIDENCE_DIR / cell / f"{cell}_SIX_SEED_SUMMARY.json"
    out.write_text(json.dumps(evidence, indent=2, default=str) + "\n",
                   encoding="utf-8", newline="\n")
    print(f"{cell}: median_Q={summary['median_Q']:.6f} "
          f"median_Q_STRUCT={summary['median_Q_STRUCT']:.6f} "
          f"median_min_recall={summary['median_minimum_label_recall']:.6f}")
    if no_transfer_ucb is not None:
        print(f"bootstrap ci_97_5 (UCB) = {no_transfer_ucb:.6f}")
    print(f"TRANSFER_STATE: {state}")
    print(f"summary: {out}")


if __name__ == "__main__":
    main()
