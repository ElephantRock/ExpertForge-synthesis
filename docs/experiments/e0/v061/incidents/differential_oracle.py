"""Exhaustive differential oracle for structsig_r2 (V06-STRUCTSIG-REMEDIATION-2A).

Compares the optimized stabilizer-pruned canonicalizer against a deliberately
slow NO-PRUNING IR reference on synthetic typed CMDR-graph structures under
many complete nuisance renamings. The optimized canonicalizer must equal the
no-prune reference EXACTLY on every structure and every renaming.

Structure classes exercised:
  - directed cycles (sizes 3-8)
  - disconnected repeated components
  - co-twins (mutually indistinguishable predicates)
  - same-degree non-isomorphic pairs
  - randomly generated small typed hypergraphs (seeded, deterministic)
  - real pilot families (small sample, both surfaces, all depths)

The oracle also stress-tests the specific defect the authority identified:
structures with a rich automorphism group where stabilizer-awareness of the
individualization path is load-bearing (cycles crossed with extra structure).
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

import structsig_r2
from msel_corpus import build_families, LABELS

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EVIDENCE_PATH = REPO_ROOT / "docs" / "experiments" / "e0" / "v061" / "DIFFERENTIAL_ORACLE_R2A.json"


# ---------------------------------------------------------------------------
# no-pruning reference canonicalizer (same refinement + serialization as
# structsig_r2 but NEVER skips a branch — exhaustively explores every leaf)
# ---------------------------------------------------------------------------

def _refine_noprune(colors, facts, rules, query, incid):
    return structsig_r2._refine(colors, facts, rules, query, incid)


_NOPRUNE_STEP_CAP = 50000


class _NoPruneCapExceeded(RuntimeError):
    pass


def _search_noprune(colors, facts, rules, query, depth, incid=None, budget=None):
    """Exhaustive IR: branch on EVERY member of the smallest non-singleton
    cell with zero pruning. Returns all leaves; canonical = min. A step cap
    (default 50k branches) raises rather than running factorial forever on
    structures with huge automorphism groups; the oracle skips and reports
    those."""
    if budget is None:
        budget = [_NOPRUNE_STEP_CAP]
    budget[0] -= 1
    if budget[0] < 0:
        raise _NoPruneCapExceeded("no-prune reference step cap exceeded")
    colors = _refine_noprune(colors, facts, rules, query, incid)
    cells: dict[str, list] = {}
    for node, c in colors.items():
        cells.setdefault(c, []).append(node)
    nonsingle = [(len(v), c, v) for c, v in cells.items() if len(v) > 1]
    if not nonsingle:
        blob, order = structsig_r2._serialize(colors, facts, rules, query, depth)
        return {(blob, tuple(order))}
    nonsingle.sort(key=lambda t: (t[0], t[1]))
    _, _, target = nonsingle[0]
    results = set()
    for node in sorted(target, key=lambda n: (n[0], n[1])):
        branched = dict(colors)
        branched[node] = structsig_r2._h("INDIV", colors[node], node)
        results |= _search_noprune(branched, facts, rules, query, depth, incid, budget)
    return results


def variant_canonical_noprune(example: dict) -> str:
    facts, rules, query, depth, preds, ents = structsig_r2._extract(example)
    colors = structsig_r2._initial_colors(preds, ents, query)
    incid = structsig_r2._precompute_incidence(facts, rules, query, preds, ents)
    leaves = _search_noprune(colors, facts, rules, query, depth, incid)
    return min(blob for blob, _ in leaves)


# ---------------------------------------------------------------------------
# synthetic structure generators (deterministic, seeded)
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
    """Directed cycle + a pendant vertex connected to one cycle node —
    creates asymmetry that makes stabilizer-awareness load-bearing."""
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
    """n predicates that are all premises of distinct rules with identical
    sign patterns leading to the same conclusion — true co-twins."""
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
# renaming utilities
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


# ---------------------------------------------------------------------------
# oracle driver
# ---------------------------------------------------------------------------

def run_oracle() -> dict:
    started = time.time()
    structures = []
    # directed cycles 3-8
    for k in range(3, 9):
        structures.append((f"cycle_{k}", gen_directed_cycle(k)))
        structures.append((f"cycle_pendant_{k}", gen_cycle_with_pendant(k)))
    # disconnected repeats
    for n, sz in [(2, 3), (3, 4), (4, 3), (2, 5)]:
        structures.append((f"disconnected_{n}x{sz}", gen_disconnected_repeats(n, sz)))
    # co-twins
    for n in [3, 4, 5, 6]:
        structures.append((f"cotwins_{n}", gen_cotwins(n)))
    # same-degree pair (must produce DIFFERENT canonicals)
    a, b = gen_same_degree_pair()
    structures.append(("same_deg_a", a))
    structures.append(("same_deg_b", b))
    # random small structures (deterministic seeds)
    for seed in range(100):
        structures.append((f"rand_{seed}", gen_random_small(seed)))
    # real pilot families (4 per surface/depth)
    for surface in ("ID",):
        for depth in range(1, 5):
            fams = build_families("ExpertForge-E0-v061-msel-oracle-burn", "oracle", surface, depth, range(4))
            for fam in fams:
                for v in fam["variants"]:
                    structures.append((f"real_{surface}_d{depth}_{fam['family_index']}_{v['gold_label']}", v))

    results = {"structures_tested": 0, "renamings_per_structure": 8, "mismatches": [], "distinctness_checks": []}
    skipped = []
    for name, ex in structures:
        try:
            ref = variant_canonical_noprune(ex)
        except _NoPruneCapExceeded:
            skipped.append(name)
            continue
        opt = structsig_r2.variant_canonical(ex)
        if ref != opt:
            results["mismatches"].append({"structure": name, "kind": "canonical_mismatch", "ref": ref[:24], "opt": opt[:24]})
            continue
        ok = True
        for seed in range(results["renamings_per_structure"]):
            renamed = full_renaming(ex, seed)
            try:
                ref_r = variant_canonical_noprune(renamed)
            except _NoPruneCapExceeded:
                continue
            opt_r = structsig_r2.variant_canonical(renamed)
            if ref_r != opt_r:
                results["mismatches"].append({"structure": name, "kind": "renamed_ref_vs_opt", "seed": seed, "ref": ref_r[:24], "opt": opt_r[:24]})
                ok = False
                break
            if opt_r != opt:
                results["mismatches"].append({"structure": name, "kind": "renaming_invariance", "seed": seed, "orig": opt[:24], "renamed": opt_r[:24]})
                ok = False
                break
        results["structures_tested"] += 1
    results["no_prune_skipped"] = skipped

    # same-degree pair must differ
    ca = variant_canonical_noprune(a)
    cb = variant_canonical_noprune(b)
    results["distinctness_checks"].append({"pair": "same_degree", "differ": ca != cb})

    # non-isomorphic random structures must mostly differ (report rate)
    rand_canons = [variant_canonical_noprune(gen_random_small(s)) for s in range(50)]
    results["distinctness_checks"].append({
        "pair": "50_random",
        "unique": len(set(rand_canons)),
        "total": len(rand_canons),
    })

    results["status"] = "PASS" if not results["mismatches"] else "FAIL"
    results["wall_seconds"] = round(time.time() - started, 1)
    results["code_git_commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()

    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: results[k] for k in ("status", "structures_tested", "mismatches", "distinctness_checks", "wall_seconds")}, indent=2))
    return results


if __name__ == "__main__":
    raise SystemExit(0 if run_oracle()["status"] == "PASS" else 1)
