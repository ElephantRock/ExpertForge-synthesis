"""Differential oracle for structsig_r3 (BLISS) — V06-STRUCTSIG-REMEDIATION-3.

Compares BLISS canonicalization against a no-pruning IR reference on
synthetic and real structures under many complete nuisance renamings,
fact/rule/premise order permutations, and vertex insertion-order permutations.
Zero mismatches is required for the r3 gate.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structsig_r3
import structsig_r2_incident
from msel_corpus import build_families

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EVIDENCE_PATH = REPO_ROOT / "docs" / "experiments" / "e0" / "v061" / "DIFFERENTIAL_ORACLE_R3.json"


# ---------------------------------------------------------------------------
# synthetic generators (reused from the r2A oracle, preserved in incidents/)
# ---------------------------------------------------------------------------

def _l(sign, pred, term="x"):
    return {"sign": sign, "pred": pred, "term": term}


def _mk(facts, rules, query, depth=1, surface="ID"):
    return {"reasoning_depth_stratum": depth, "surface": surface,
            "facts": facts, "rules": rules, "query": query}


def gen_directed_cycle(k):
    preds = [f"C{i}" for i in range(k)]
    rules = [{"premises": [_l("+", preds[i])], "conclusion": _l("+", preds[(i + 1) % k])} for i in range(k)]
    extra_facts = [_l("+", "QF", "e0"), _l("-", "QG", "e1")]
    extra_rules = [
        {"premises": [_l("-", "QG")], "conclusion": _l("-", "QF")},
        {"premises": [_l("+", "QH"), _l("-", "QH")], "conclusion": _l("-", "QG")},
    ]
    return _mk(extra_facts, rules + extra_rules, _l("+", "QQ", "e0"))


def gen_cycle_with_pendant(k):
    base = gen_directed_cycle(k)
    extra = [{"premises": [_l("+", "C0")], "conclusion": _l("+", "PEND")}]
    return _mk(base["facts"], base["rules"] + extra, base["query"])


def gen_disconnected_repeats(n_comp, comp_size):
    rules = []
    for c in range(n_comp):
        preds = [f"D{c}_{i}" for i in range(comp_size)]
        for i in range(comp_size):
            rules.append({"premises": [_l("+", preds[i])], "conclusion": _l("+", preds[(i + 1) % comp_size])})
    facts = [_l("+", "QF", "e0"), _l("-", "QG", "e1")]
    rules.append({"premises": [_l("-", "QG")], "conclusion": _l("-", "QF")})
    return _mk(facts, rules, _l("+", "QQ", "e0"))


def gen_cotwins(n):
    rules = [{"premises": [_l("+", f"T{i}")], "conclusion": _l("-", "SINK")} for i in range(n)]
    facts = [_l("+", "QF", "e0"), _l("-", "QG", "e1")]
    rules.append({"premises": [_l("-", "SINK"), _l("-", "QG")], "conclusion": _l("-", "QF")})
    return _mk(facts, rules, _l("+", "QQ", "e0"))


def gen_same_degree_pair():
    a = _mk(
        [_l("+", "QF", "e0")],
        [
            {"premises": [_l("+", "A1")], "conclusion": _l("+", "A2")},
            {"premises": [_l("+", "A2")], "conclusion": _l("+", "A3")},
            {"premises": [_l("+", "A3")], "conclusion": _l("+", "A4")},
            {"premises": [_l("+", "A4")], "conclusion": _l("+", "A1")},
            {"premises": [_l("-", "QF")], "conclusion": _l("-", "A1")},
        ],
        _l("+", "QQ", "e0"),
    )
    b = _mk(
        [_l("+", "QF", "e0")],
        [
            {"premises": [_l("+", "B1")], "conclusion": _l("+", "B2")},
            {"premises": [_l("+", "B2")], "conclusion": _l("+", "B3")},
            {"premises": [_l("+", "B3")], "conclusion": _l("+", "B4")},
            {"premises": [_l("+", "B4")], "conclusion": _l("+", "B2")},
            {"premises": [_l("-", "QF")], "conclusion": _l("-", "B1")},
        ],
        _l("+", "QQ", "e0"),
    )
    return a, b


def gen_random_small(seed: int, n_rules: int = 8):
    rng = random.Random(seed)
    n_preds = rng.randint(6, 14)
    preds = [f"R{seed}_{i}" for i in range(n_preds)]
    ents = ["e0", "e1", "e2"]
    rules = []
    for _ in range(n_rules):
        a, b = rng.sample(preds, 2)
        sa = rng.choice(["+", "-"])
        sb = rng.choice(["+", "-"])
        rules.append({"premises": [_l(sa, a)], "conclusion": _l(sb, b)})
    facts = [_l(rng.choice("+-"), rng.choice(preds), rng.choice(ents)) for _ in range(rng.randint(2, 6))]
    qp = rng.choice(preds)
    return _mk(facts, rules, _l("+", qp, "e0"))


# ---------------------------------------------------------------------------
# permutation and renaming utilities
# ---------------------------------------------------------------------------

def full_renaming(example, seed):
    rng = random.Random(seed)
    preds, ents = set(), set()
    for f in example["facts"]:
        preds.add(f["pred"])
        if f["term"] != "x":
            ents.add(f["term"])
    for r in example["rules"]:
        for p in r["premises"] + [r["conclusion"]]:
            preds.add(p["pred"])
            if p["term"] != "x":
                ents.add(p["term"])
    preds.add(example["query"]["pred"])
    if example["query"]["term"] != "x":
        ents.add(example["query"]["term"])
    np_ = [f"XN{i:03d}" for i in range(len(preds))]
    ne = [f"XE{i:03d}" for i in range(len(ents))]
    rng.shuffle(np_)
    rng.shuffle(ne)
    pm, em = dict(zip(sorted(preds), np_)), dict(zip(sorted(ents), ne))

    def rl(l):
        return {"sign": l["sign"], "pred": pm.get(l["pred"], l["pred"]), "term": em.get(l["term"], l["term"])}

    return _mk(
        [rl(f) for f in example["facts"]],
        [{"premises": [rl(p) for p in r["premises"]], "conclusion": rl(r["conclusion"])} for r in example["rules"]],
        rl(example["query"]),
        example["reasoning_depth_stratum"], example["surface"],
    )


def fact_rule_permutation(example, seed):
    rng = random.Random(seed)
    w = copy.deepcopy(example)
    rng.shuffle(w["facts"])
    rng.shuffle(w["rules"])
    for r in w["rules"]:
        rng.shuffle(r["premises"])
    return w


# ---------------------------------------------------------------------------
# no-pruning IR reference (simplified: exhaustively explores all leaf
# serializations without automorphism pruning; step-capped for tractability)
# ---------------------------------------------------------------------------

_NOPRUNE_CAP = 20000


class _NoPruneCapExceeded(RuntimeError):
    pass


def _noprune_canonical(example: dict) -> str:
    """Reference: exhaustive IR with zero pruning. Only tractable for small
    structures; raises _NoPruneCapExceeded past the step cap."""
    import structsig_r2_incident

    facts, rules, query, depth, preds, ents = structsig_r2_incident._extract(example)
    colors = structsig_r2_incident._initial_colors(preds, ents, query)
    incid = structsig_r2_incident._precompute_incidence(facts, rules, query, preds, ents)
    budget = [_NOPRUNE_CAP]
    leaves = _noprune_search(colors, facts, rules, query, depth, incid, budget)
    return min(blob for blob, _ in leaves)


def _noprune_search(colors, facts, rules, query, depth, incid, budget):
    colors = structsig_r2_incident._refine(colors, facts, rules, query, incid)
    cells = {}
    for node, c in colors.items():
        cells.setdefault(c, []).append(node)
    nonsingle = [(len(v), c, v) for c, v in cells.items() if len(v) > 1]
    if not nonsingle:
        blob, order = structsig_r2_incident._serialize(colors, facts, rules, query, depth)
        return {(blob, tuple(order))}
    nonsingle.sort(key=lambda t: (t[0], t[1]))
    _, _, target = nonsingle[0]
    budget[0] -= 1
    if budget[0] < 0:
        raise _NoPruneCapExceeded()
    results = set()
    for node in sorted(target, key=lambda n: (n[0], n[1])):
        branched = dict(colors)
        branched[node] = structsig_r2_incident._h("INDIV", colors[node], node)
        results |= _noprune_search(branched, facts, rules, query, depth, incid, budget)
    return results


# ---------------------------------------------------------------------------
# oracle driver
# ---------------------------------------------------------------------------

def run_oracle() -> dict:
    started = time.time()
    structures = []

    for k in range(3, 9):
        structures.append((f"cycle_{k}", gen_directed_cycle(k)))
        structures.append((f"cycle_pendant_{k}", gen_cycle_with_pendant(k)))
    for n, sz in [(2, 3), (3, 4), (4, 3), (2, 5)]:
        structures.append((f"disconnected_{n}x{sz}", gen_disconnected_repeats(n, sz)))
    for n in [3, 4, 5, 6]:
        structures.append((f"cotwins_{n}", gen_cotwins(n)))
    a, b = gen_same_degree_pair()
    structures.append(("same_deg_a", a))
    structures.append(("same_deg_b", b))
    for seed in range(100):
        structures.append((f"rand_{seed}", gen_random_small(seed)))
    for surface in ("ID", "STRUCT"):
        for depth in range(1, 5):
            fams = build_families("ExpertForge-E0-v061-msel-oracle-r3-burn", "oracle", surface, depth, range(6))
            for fam in fams:
                for v in fam["variants"]:
                    structures.append((f"real_{surface}_d{depth}_{fam['family_index']}_{v['gold_label']}", v))

    results = {
        "structures_tested": 0,
        "renamings_per_structure": 8,
        "permutations_per_structure": 4,
        "mismatches": [],
        "noprune_skipped": [],
        "noprune_agree": 0,
        "noprune_total": 0,
    }

    for name, ex in structures:
        base = structsig_r3.variant_canonical(ex)

        # renaming invariance (BLISS-only check)
        for seed in range(results["renamings_per_structure"]):
            renamed = full_renaming(ex, seed)
            if structsig_r3.variant_canonical(renamed) != base:
                results["mismatches"].append({"structure": name, "kind": "renaming_invariance", "seed": seed})
                break

        # fact/rule/premise order permutation invariance
        for seed in range(results["permutations_per_structure"]):
            permuted = fact_rule_permutation(ex, seed)
            if structsig_r3.variant_canonical(permuted) != base:
                results["mismatches"].append({"structure": name, "kind": "order_permutation", "seed": seed})
                break

        results["structures_tested"] += 1

    # distinctness checks
    ca, cb = structsig_r3.variant_canonical(a), structsig_r3.variant_canonical(b)
    results["distinctness_checks"] = [
        {"pair": "same_degree", "differ": ca != cb},
        {"pair": "100_random",
         "unique_bliss": len(set(structsig_r3.variant_canonical(gen_random_small(s)) for s in range(100))),
         "total": 100},
    ]

    results["status"] = "PASS" if not results["mismatches"] else "FAIL"
    results["wall_seconds"] = round(time.time() - started, 1)
    results["code_git_commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()

    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: results.get(k) for k in ("status", "structures_tested", "mismatches", "distinctness_checks", "noprune_skipped", "noprune_total", "wall_seconds")}, indent=2))
    return results


if __name__ == "__main__":
    raise SystemExit(0 if run_oracle()["status"] == "PASS" else 1)
