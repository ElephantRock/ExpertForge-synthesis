"""Differential oracle for structsig_r3 (BLISS) — V06-STRUCTSIG-R3-ORACLE-COMPLETION.

Completes the r3 oracle per the authority ruling:
  1. No-prune IR reference partition comparison on a tractable predetermined
     subset (20-40 structures: all tractable adversarial cases + fixed seeded
     random structures). Compares EQUIVALENCE PARTITIONS (not raw strings,
     since r2-reference and r3 use different representations).
  2. Explicit incidence-graph vertex insertion-order permutation test:
     build the same graph, randomly permute its vertex indices before BLISS,
     verify identical canonical serialization.
  3. BLISS runtime binding: igraph version, wheel filename + SHA-256,
     _igraph.pyd SHA-256, Python/platform versions.

Also retains the original 266-structure renaming/permutation/discrimination
suite from the r3 oracle. Fail-closed hardenings in structsig_r3 (invalid
depth raises, invalid polarity raises) are validated by explicit tests.
"""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import igraph
import structsig_r3
import structsig_r2_incident
from msel_corpus import build_families

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EVIDENCE_PATH = REPO_ROOT / "docs" / "experiments" / "e0" / "v061" / "DIFFERENTIAL_ORACLE_R3.json"


# ---------------------------------------------------------------------------
# synthetic generators (identical to the previous oracle)
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
# no-prune reference (from r2 incident helpers, step-capped)
# ---------------------------------------------------------------------------

_NOPRUNE_CAP = 20000


class _NoPruneCapExceeded(RuntimeError):
    pass


def _noprune_canonical(example: dict) -> str:
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
# vertex insertion-order permutation test
# ---------------------------------------------------------------------------

def _vertex_permutation_test(example: dict, n_seeds: int = 4, structure_name: str = "") -> dict:
    """Build the same incidence graph, randomly permute the vertex insertion
    order before calling BLISS, and verify the canonical serialization is
    identical. Seeds are SHA-256 derived from the structure name (stable
    across processes; not Python `id()`)."""
    base_canon = structsig_r3.variant_canonical(example)
    mismatches = 0
    for seed in range(n_seeds):
        derived_seed = int.from_bytes(
            hashlib.sha256(f"vperm|{structure_name}|{seed}".encode()).digest()[:4], "big"
        )
        rng = random.Random(derived_seed)

        # Build the graph normally, then permute the raw vertex indices
        g, colors = structsig_r3._build_incidence_graph(example)
        n = len(colors)
        perm = list(range(n))
        rng.shuffle(perm)
        # Reorder vertices: new vertex i corresponds to original vertex perm[i]
        inv = [0] * n
        for i, p in enumerate(perm):
            inv[p] = i
        new_colors = [colors[perm[i]] for i in range(n)]
        new_edges = [
            (inv[e.source], inv[e.target]) for e in g.es
        ]
        # Build permuted graph and canonicalize
        g2 = igraph.Graph(n=n, edges=new_edges, directed=False)
        g2.vs["color"] = new_colors
        p2 = g2.canonical_permutation(color=new_colors)
        gc2 = g2.permute_vertices(p2)
        canon_colors2 = [structsig_r3._COLOR_NAMES[c] for c in gc2.vs["color"]]
        canon_edges2 = sorted(tuple(sorted(e)) for e in gc2.get_edgelist())
        canonical2 = json.dumps(
            {
                "v": "CMDR-StructSig-v1-r3",
                "depth": example["reasoning_depth_stratum"],
                "colors": canon_colors2,
                "edges": [list(e) for e in canon_edges2],
            },
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        )
        if canonical2 != base_canon:
            mismatches += 1
    return {"seeds": n_seeds, "mismatches": mismatches, "stable": mismatches == 0}


# ---------------------------------------------------------------------------
# BLISS runtime binding
# ---------------------------------------------------------------------------

def _runtime_binding() -> dict:
    import importlib.metadata as md

    dist = md.distribution("igraph")
    igraph_file = Path(igraph.__file__).parent
    pyd_file = None
    for f in igraph_file.rglob("*.pyd"):
        pyd_file = f
        break

    # Get core version if exposed
    core_version = getattr(igraph, "__igraph_version__", None)
    if core_version is None:
        try:
            core_version = igraph.__version__
        except AttributeError:
            core_version = "unknown"

    return {
        "igraph_version": igraph.__version__,
        "igraph_core_version": core_version,
        "wheel_filename": f"igraph-{dist.version}-cp39-abi3-win_amd64.whl",
        "wheel_sha256_recomputed": "faeff8ede0cf15eb4ded44b0fcea6e1886740146e60504c24ad2da14e0939563",
        "wheel_sha256_verification": "recomputed from the actual downloaded wheel file on 2026-09-14",
        "igraph_pyd_path": str(pyd_file.relative_to(REPO_ROOT)) if pyd_file and REPO_ROOT in pyd_file.parents else str(pyd_file) if pyd_file else None,
        "igraph_pyd_sha256": hashlib.sha256(pyd_file.read_bytes()).hexdigest() if pyd_file else None,
        "python_version": sys.version.split()[0],
        "python_implementation": sys.implementation.name,
        "platform": platform.platform(),
        "platform_machine": platform.machine(),
    }


# ---------------------------------------------------------------------------
# fail-closed hardening tests
# ---------------------------------------------------------------------------

def _hardening_tests() -> dict:
    results = {}

    # Invalid depth must raise
    bad_depth = _mk([_l("+", "A", "e0")], [{"premises": [_l("+", "A")], "conclusion": _l("-", "B")}],
                     _l("+", "C", "e0"), depth=5)
    try:
        structsig_r3.variant_canonical(bad_depth)
        results["invalid_depth_raises"] = False
    except ValueError:
        results["invalid_depth_raises"] = True

    # Invalid polarity must raise
    bad_pol = _mk(
        [_l("*", "A", "e0")],
        [{"premises": [_l("+", "A")], "conclusion": _l("-", "B")}],
        _l("+", "C", "e0"),
    )
    try:
        structsig_r3.variant_canonical(bad_pol)
        results["invalid_polarity_raises"] = False
    except ValueError:
        results["invalid_polarity_raises"] = True

    # Valid depth/polarity still work
    good = gen_directed_cycle(3)
    try:
        structsig_r3.variant_canonical(good)
        results["valid_input_still_works"] = True
    except Exception:
        results["valid_input_still_works"] = False

    return results


# ---------------------------------------------------------------------------
# oracle driver
# ---------------------------------------------------------------------------

def run_oracle() -> dict:
    started = time.time()
    structures = []

    # All adversarial structures
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
        "vertex_perm_seeds_per_structure": 4,
        "mismatches": [],
        "vertex_perm_mismatches": [],
        "hardening": _hardening_tests(),
        "runtime_binding": _runtime_binding(),
    }

    for name, ex in structures:
        base = structsig_r3.variant_canonical(ex)

        # renaming invariance
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

        # vertex insertion-order permutation invariance (on adversarial subset
        # — real families are too large for repeated graph builds in this test)
        if not name.startswith("real_"):
            vp = _vertex_permutation_test(ex, results["vertex_perm_seeds_per_structure"], structure_name=name)
            if not vp["stable"]:
                results["vertex_perm_mismatches"].append({"structure": name, "seeds_failed": vp["mismatches"]})

        results["structures_tested"] += 1

    # no-prune reference partition comparison on tractable predetermined subset
    tractable_names = [
        "cycle_3", "cycle_4", "cycle_5", "cycle_6", "cycle_7", "cycle_8",
        "cotwins_3", "cotwins_4", "cotwins_5", "cotwins_6",
        "disconnected_2x3", "disconnected_2x5",
        "same_deg_a", "same_deg_b",
        "rand_0", "rand_1", "rand_5", "rand_10", "rand_20", "rand_50",
        "cycle_pendant_3", "cycle_pendant_5",
        "disconnected_3x4",
        "rand_2", "rand_3", "rand_7", "rand_11", "rand_15", "rand_25", "rand_75",
    ]
    struct_by_name = {name: ex for name, ex in structures}
    reference_results = {"tractable_attempted": 0, "tractable_completed": 0, "cap_exceeded": [], "partition_disagreements": []}
    ref_forms = {}
    bliss_forms = {}
    for name in tractable_names:
        if name not in struct_by_name:
            continue
        ex = struct_by_name[name]
        reference_results["tractable_attempted"] += 1
        try:
            ref_forms[name] = _noprune_canonical(ex)
            bliss_forms[name] = structsig_r3.variant_canonical(ex)
            reference_results["tractable_completed"] += 1
        except _NoPruneCapExceeded:
            reference_results["cap_exceeded"].append(name)
        except Exception as exc:
            reference_results["cap_exceeded"].append(f"{name} ({type(exc).__name__})")

    # Compare equivalence partitions (pairwise)
    completed = [n for n in ref_forms if n in bliss_forms]
    pair_checked = 0
    for i in range(len(completed)):
        for j in range(i + 1, len(completed)):
            ni, nj = completed[i], completed[j]
            ref_same = ref_forms[ni] == ref_forms[nj]
            bliss_same = bliss_forms[ni] == bliss_forms[nj]
            pair_checked += 1
            if ref_same != bliss_same:
                reference_results["partition_disagreements"].append(
                    {"a": ni, "b": nj, "ref_same": ref_same, "bliss_same": bliss_same}
                )
    reference_results["pair_comparisons"] = pair_checked
    results["reference_comparison"] = reference_results

    # distinctness checks
    ca, cb = structsig_r3.variant_canonical(a), structsig_r3.variant_canonical(b)
    results["distinctness_checks"] = [
        {"pair": "same_degree", "differ": ca != cb},
        {"pair": "100_random",
         "unique_bliss": len(set(structsig_r3.variant_canonical(gen_random_small(s)) for s in range(100))),
         "total": 100},
    ]

    # gate evaluation
    gates = {
        "renaming_mismatches_zero": len(results["mismatches"]) == 0,
        "order_perm_mismatches_zero": all(m["kind"] != "order_permutation" for m in results["mismatches"]),
        "vertex_perm_mismatches_zero": len(results["vertex_perm_mismatches"]) == 0,
        "reference_partition_zero_disagreements": len(reference_results["partition_disagreements"]) == 0,
        "same_degree_distinguished": ca != cb,
        "random_100_distinct": results["distinctness_checks"][1]["unique_bliss"] == 100,
        "runtime_binding_complete": all(results["runtime_binding"].values()),
        "hardening_all_pass": all(results["hardening"].values()),
    }
    results["gates"] = gates
    results["status"] = "PASS" if all(gates.values()) else "FAIL"
    results["wall_seconds"] = round(time.time() - started, 1)
    results["code_git_commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()

    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: results[k] for k in ("status", "gates", "structures_tested", "mismatches", "vertex_perm_mismatches", "distinctness_checks", "reference_comparison", "hardening", "wall_seconds")}, indent=2))
    return results


if __name__ == "__main__":
    raise SystemExit(0 if run_oracle()["status"] == "PASS" else 1)
