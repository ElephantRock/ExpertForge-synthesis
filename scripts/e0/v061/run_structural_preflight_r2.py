"""Structural preflight r2 + depletion probe (V06-STRUCTSIG-REMEDIATION-2).

Modes:
  --mode correctness : adversarial suite + ALL-family renaming/order
      permutation invariance + independent typed graph-isomorphism check on
      every repeated-signature group (2,000 families/depth/surface default).
  --mode depletion   : discovery curve through 32,000 ID families/depth
      (train-pool scale), signature burn, then from a SEPARATE burned
      namespace the acceptance rate when requesting families disjoint from
      the burn at qualification-scale demand.

Namespace(s) burned on use. msel_corpus.py / msel_verifier.py byte-identical
to cfe0840. Two-commit discipline: this file is implementation; evidence is
committed separately.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structsig_r2
from msel_corpus import apply_renaming, build_families
from msel_verifier import counterfactual_invariance, verify_family

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EVIDENCE = REPO_ROOT / "docs" / "experiments" / "e0" / "v061"
NS_CORRECTNESS = "ExpertForge-E0-v061-msel-preflight-r2"
NS_DEPLETION_DISCOVERY = "ExpertForge-E0-v061-msel-preflight-r2-disc"
NS_DEPLETION_FRESH = "ExpertForge-E0-v061-msel-preflight-r2-fresh"


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def tree_clean() -> bool:
    return subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip() == ""


# --------------------------------------------------------------------------
# adversarial structure builders (hand-constructed CMDR-schema examples)
# --------------------------------------------------------------------------

def _mk_example(facts, rules, query, depth=1, surface="ID"):
    return {"reasoning_depth_stratum": depth, "surface": surface, "facts": facts, "rules": rules, "query": query}


def _l(sign, pred, term="x"):
    return {"sign": sign, "pred": pred, "term": term}


def _cycle_family(k: int):
    """Directed cycle of k distractor predicates P1->P2->...->Pk->P1, query
    in a separate component (the authority's counterexample class)."""
    preds = [f"C{i}" for i in range(k)]
    rules = [
        {"premises": [_l("+", preds[i])], "conclusion": _l("+", preds[(i + 1) % k])}
        for i in range(k)
    ]
    facts = [_l("+", "QF", "e0"), _l("-", "QG", "e1")]
    rules += [
        {"premises": [_l("-", "QG")], "conclusion": _l("+", "QH")},
        {"premises": [_l("+", "QH"), _l("-", "QH")], "conclusion": _l("-", "QF")},
    ]
    return _mk_example(facts, rules, _l("+", "QQ", "e0"))


def _renamed(example, mapping):
    def rl(l):
        return {"sign": l["sign"], "pred": mapping.get(l["pred"], l["pred"]), "term": mapping.get(l["term"], l["term"])}

    return _mk_example(
        [rl(f) for f in example["facts"]],
        [{"premises": [rl(p) for p in r["premises"]], "conclusion": rl(r["conclusion"])} for r in example["rules"]],
        rl(example["query"]),
        example["reasoning_depth_stratum"], example["surface"],
    )


def _random_renaming(example, seed):
    rng = random.Random(seed)
    names = set()
    for f in example["facts"]:
        names.add(f["pred"]); names.add(f["term"])
    for r in example["rules"]:
        for p in r["premises"] + [r["conclusion"]]:
            names.add(p["pred"])
            if p["term"] != "x":
                names.add(p["term"])
    names.discard("x")
    pool = [f"RN{i:03d}" for i in range(len(names))]
    rng.shuffle(pool)
    return dict(zip(sorted(names), pool))


def _same_degree_nonisomorphic_pair():
    """Two 4-predicate graphs, every predicate appearing exactly once as a
    premise and once as a conclusion, but non-isomorphic: a 4-cycle vs a
    4-chain plus a 2-cycle."""
    cycle = [
        {"premises": [_l("+", "A1")], "conclusion": _l("+", "A2")},
        {"premises": [_l("+", "A2")], "conclusion": _l("+", "A3")},
        {"premises": [_l("+", "A3")], "conclusion": _l("+", "A4")},
        {"premises": [_l("+", "A4")], "conclusion": _l("+", "A1")},
    ]
    chain = [
        {"premises": [_l("+", "B1")], "conclusion": _l("+", "B2")},
        {"premises": [_l("+", "B2")], "conclusion": _l("+", "B3")},
        {"premises": [_l("+", "B3")], "conclusion": _l("+", "B4")},
        {"premises": [_l("+", "B4")], "conclusion": _l("+", "B2")},
    ]
    facts = [_l("+", "ZF", "e0"), _l("-", "ZG", "e1")]
    extra = [{"premises": [_l("-", "ZG")], "conclusion": _l("-", "ZF")}]
    return (
        _mk_example(facts, cycle + extra, _l("+", "ZQ", "e0")),
        _mk_example(facts, chain + extra, _l("+", "ZQ", "e0")),
    )


def _adversarial_suite() -> dict:
    results = {}

    # directed cycles: renaming invariance under random bijections
    for k in (3, 4, 5, 6):
        ex = _cycle_family(k)
        base = structsig_r2.variant_canonical(ex)
        invar = True
        for seed in range(5):
            m = _random_renaming(ex, seed)
            if structsig_r2.variant_canonical(_renamed(ex, m)) != base:
                invar = False
                break
        results[f"directed_cycle_{k}_renaming_invariant"] = invar

    # same-degree non-isomorphic pair must differ
    a, b = _same_degree_nonisomorphic_pair()
    results["same_degree_nonisomorphic_differ"] = (
        structsig_r2.variant_canonical(a) != structsig_r2.variant_canonical(b)
    )
    # ...and each is individually renaming-invariant
    ra = _random_renaming(a, 11)
    rb = _random_renaming(b, 12)
    results["pair_a_renaming_invariant"] = (
        structsig_r2.variant_canonical(_renamed(a, ra)) == structsig_r2.variant_canonical(a)
    )
    results["pair_b_renaming_invariant"] = (
        structsig_r2.variant_canonical(_renamed(b, rb)) == structsig_r2.variant_canonical(b)
    )

    # multiplicity: one predicate twice in facts vs once in facts + once in premises
    fa = [_l("+", "P1", "E1"), _l("+", "P1", "E2")]
    ra_ = [{"premises": [_l("+", "P3")], "conclusion": _l("-", "P4")}]
    fb = [_l("+", "P1", "E1"), _l("+", "P3", "E2")]
    rb_ = [{"premises": [_l("+", "P1")], "conclusion": _l("-", "P4")}]
    ea = _mk_example(fa, ra_, _l("+", "P1", "E1"))
    eb = _mk_example(fb, rb_, _l("+", "P1", "E1"))
    results["multiplicity_distinguished"] = (
        structsig_r2.variant_canonical(ea) != structsig_r2.variant_canonical(eb)
    )

    # disconnected repeated components: renaming invariance
    d1 = _mk_example(
        [_l("+", "U1", "e0"), _l("+", "U2", "e1")],
        [
            {"premises": [_l("+", "U1")], "conclusion": _l("-", "V1")},
            {"premises": [_l("+", "U2")], "conclusion": _l("-", "V1")},
            {"premises": [_l("-", "V1")], "conclusion": _l("+", "W1")},
            {"premises": [_l("+", "W1"), _l("-", "W1")], "conclusion": _l("-", "X1")},
        ],
        _l("+", "U1", "e0"),
    )
    base = structsig_r2.variant_canonical(d1)
    ok = True
    for seed in range(5):
        if structsig_r2.variant_canonical(_renamed(d1, _random_renaming(d1, seed))) != base:
            ok = False
            break
    results["disconnected_components_renaming_invariant"] = ok
    return results


# --------------------------------------------------------------------------
# independent typed graph-isomorphism matcher (backtracking + refinement)
# --------------------------------------------------------------------------

def _iso_refine(colors, facts, rules, query):
    return structsig_r2._refine(colors, facts, rules, query)


_GI_STEP_BUDGET = 200000


class _GIExceeded(RuntimeError):
    pass


def _variant_to_nx(example: dict):
    """Encode one variant as an attributed networkx DiGraph for VF2:
    predicate/entity nodes (kind attribute) plus one attribute-rich node per
    literal slot (fact/premise/conclusion/query role, polarity, grounding);
    meta node carries the depth stratum. Independent of structsig_r2
    canonical labeling."""
    import networkx as nx

    facts, rules, query, depth, preds, ents = structsig_r2._extract(example)
    g = nx.DiGraph()
    g.add_node(("meta",), kind="meta", depth=depth)
    for p in preds:
        g.add_node(("P", p), kind="P")
    for e in ents:
        g.add_node(("E", e), kind="E")

    def add_lit(l, role, eid):
        pid = ("slot",) + eid + (role, l[0], id(l) % 1)
        g.add_node(pid, kind=role, sign=l[0], grounded=l[2] != "x")
        g.add_edge(("P", l[1]), pid, etype="holds")
        if l[2] != "x":
            g.add_edge(("E", l[2]), pid, etype="holds")
        return pid

    for i, f in enumerate(facts):
        add_lit(f, "fact", ("f", i))
    for i, (prem, concl) in enumerate(rules):
        for j, l in enumerate(prem):
            add_lit(l, "prem", ("r", i, j))
        add_lit(concl, "concl", ("r", i, "c"))
    qnode = add_lit(query, "query", ("q",))
    g.add_edge(("meta",), qnode, etype="query")
    return g


def _typed_isomorphic(ex_a: dict, ex_b: dict, budget: list | None = None) -> bool:
    """Independent exact isomorphism check. Decision procedure: MRV
    backtracking over color-class-constrained candidate bijections with
    incremental hyperedge-multiset consistency. WL color classes (a sound
    isomorphism invariant) are used ONLY to constrain candidates and to
    reject early on color-multiset mismatch; no canonical labeling, orbit
    pruning, or serialization from structsig_r2 is consulted. Raises
    _GIExceeded past a step budget."""
    fa, ra, qa, da, pa, ea = structsig_r2._extract(ex_a)
    fb, rb, qb, db, pb, eb = structsig_r2._extract(ex_b)
    if da != db or ex_a["surface"] != ex_b["surface"]:
        return False
    if len(fa) != len(fb) or len(ra) != len(rb):
        return False
    steps = [0]
    BUDGET = budget[0] if budget else 1_000_000

    ca = structsig_r2._refine(structsig_r2._initial_colors(pa, ea, qa), fa, ra, qa,
                              structsig_r2._precompute_incidence(fa, ra, qa, pa, ea))
    cb = structsig_r2._refine(structsig_r2._initial_colors(pb, eb, qb), fb, rb, qb,
                              structsig_r2._precompute_incidence(fb, rb, qb, pb, eb))
    from collections import Counter, defaultdict

    if Counter(ca.values()) != Counter(cb.values()):
        return False
    b_by_color = defaultdict(list)
    for n, c in cb.items():
        b_by_color[c].append(n)

    # edge signatures under current (partial) sigma: literal color tuple
    def lit_color(l, colors):
        return (l[0], colors[("P", l[1])], colors.get(("E", l[2]), "X"), l[2] == "x")

    if sorted(lit_color(f, ca) for f in fa) != sorted(lit_color(f, cb) for f in fb):
        return False
    if sorted(
        (sorted(lit_color(p, ca) for p in prem), lit_color(c, ca)) for prem, c in ra
    ) != sorted((sorted(lit_color(p, cb) for p in prem), lit_color(c, cb)) for prem, c in rb):
        return False
    if lit_color(qa, ca) != lit_color(qb, cb):
        return False

    sigma = {}
    used = set()

    def mapped_lit(l, complete_ok):
        p = sigma.get(("P", l[1]))
        if p is None:
            return None
        t = "x" if l[2] == "x" else sigma.get(("E", l[2]))
        if t is None and l[2] != "x":
            return None
        return (l[0], p[1], t[1] if l[2] != "x" else "x")

    def consistent():
        steps[0] += 1
        if steps[0] > BUDGET:
            raise _GIExceeded("GI step budget exceeded (MRV)")
        tgt_facts = Counter(fb)
        mapped = Counter()
        for f in fa:
            m = mapped_lit(f, False)
            if m is not None:
                mapped[m] += 1
        if mapped - tgt_facts:
            return False
        tgt_rules = Counter((tuple(sorted(prem)), concl) for prem, concl in rb)
        mapped_r = Counter()
        for prem, concl in ra:
            ms = [mapped_lit(p, False) for p in prem]
            mc = mapped_lit(concl, False)
            if all(x is not None for x in ms) and mc is not None:
                mapped_r[(tuple(sorted(ms)), mc)] += 1
        if mapped_r - tgt_rules:
            return False
        return True

    nodes = list(ca.keys())
    # MRV: dynamic ordering by smallest remaining candidate class
    def order_nodes():
        def cand_count(n):
            return sum(1 for c in b_by_color[ca[n]] if c not in used)
        return sorted(nodes, key=cand_count)

    def backtrack():
        remaining = [n for n in nodes if n not in sigma]
        if not remaining:
            mq = mapped_lit(qa, True)
            return mq == qb and not (Counter(
                mapped_lit(f, True) for f in fa
            ) - Counter(fb)) and not (Counter(
                (tuple(sorted(mapped_lit(p, True) for p in prem)), mapped_lit(c, True))
                for prem, c in ra
            ) - Counter((tuple(sorted(prem)), concl) for prem, concl in rb))
        # MRV pick
        best, best_c = None, 10**9
        for n in remaining:
            cnt = sum(1 for c in b_by_color[ca[n]] if c not in used)
            if cnt < best_c:
                best, best_c = n, cnt
                if cnt <= 1:
                    break
        if best_c == 0:
            return False
        for cand in b_by_color[ca[best]]:
            if cand in used:
                continue
            sigma[best] = cand
            used.add(cand)
            if consistent() and backtrack():
                return True
            del sigma[best]
            used.discard(cand)
        return False

    return backtrack()


# --------------------------------------------------------------------------
# correctness mode
# --------------------------------------------------------------------------

def run_correctness(n: int) -> dict:
    started = time.time()
    failures = []
    adversarial = _adversarial_suite()
    for name, ok in adversarial.items():
        if not ok:
            failures.append(f"adversarial: {name}")

    per_cell = {}
    all_group_checks = []
    cross_cell = {}

    for surface in ("ID", "STRUCT"):
        for depth in range(1, 5):
            key = f"{surface}|d{depth}"
            fams = build_families(NS_CORRECTNESS, "pilot", surface, depth, range(n))
            sigs = []
            fam_reps = {}
            rename_fail = perm_fail = verr = 0
            cell_t0 = time.time()
            for fam in fams:
                errs = verify_family(fam) + counterfactual_invariance(fam)
                verr += len(errs)
                try:
                    reps = sorted(structsig_r2.variant_canonical(v) for v in fam["variants"])
                except (structsig_r2._BranchCapExceeded, structsig_r2._WallGuardExceeded) as exc:
                    failures.append(f"{key} fi{fam['family_index']}: {type(exc).__name__}")
                    continue
                fc = json.dumps({"v": "CMDR-StructSig-v1-r2-family", "orbit": reps}, sort_keys=True, separators=(",", ":"))
                sig = hashlib.sha256(fc.encode("utf-8")).hexdigest()
                sigs.append(sig)
                fam_reps.setdefault(sig, []).append(fam)
                cross_cell.setdefault(sig, []).append(key)

                # ALL-family renaming invariance
                mapping = _random_renaming_multi(fam)
                renamed = [apply_renaming(v, mapping[0], mapping[1]) for v in fam["variants"]]
                try:
                    rsig = structsig_r2.family_signature(renamed)
                except (structsig_r2._BranchCapExceeded, structsig_r2._WallGuardExceeded):
                    rsig = None
                if rsig != sig:
                    rename_fail += 1

                # ALL-family order-permutation invariance
                rng = random.Random(hash(fam["family_id"]) & 0xFFFFFFFF)
                perm_ok = True
                for v in fam["variants"]:
                    w = copy.deepcopy(v)
                    rng.shuffle(w["facts"]); rng.shuffle(w["rules"])
                    for r in w["rules"]:
                        rng.shuffle(r["premises"])
                    if structsig_r2.variant_canonical(w) != structsig_r2.variant_canonical(v):
                        perm_ok = False
                        break
                if not perm_ok:
                    perm_fail += 1

            # independent GI check on every repeated group (representative vs all)
            gi_checked = gi_pass = 0
            for sig, group in fam_reps.items():
                if len(group) < 2:
                    continue
                rep = group[0]["variants"][0]
                for other in group[1:]:
                    gi_checked += 1
                    try:
                        if _typed_isomorphic(rep, other["variants"][0]):
                            gi_pass += 1
                        else:
                            all_group_checks.append({"sig": sig[:16], "family": other["family_id"], "kind": "NOT_ISOMORPHIC"})
                    except _GIExceeded:
                        all_group_checks.append({"sig": sig[:16], "family": other["family_id"], "kind": "CHECK_EXCEEDED"})
            hard_gi_fail = [c for c in all_group_checks if c["kind"] == "NOT_ISOMORPHIC"]
            gi_exceeded = [c for c in all_group_checks if c["kind"] == "CHECK_EXCEEDED"]
            if hard_gi_fail:
                failures.append(f"{key}: {len(hard_gi_fail)} independent-GI NOT_ISOMORPHIC failures")
            if gi_exceeded:
                failures.append(f"{key}: {len(gi_exceeded)} GI checks exceeded step budget (reported, not scored)")

            sc = Counter(sigs)
            per_cell[key] = {
                "attempted": n,
                "unique": len(sc),
                "max_multiplicity": max(sc.values()),
                "verifier_errors": verr,
                "rename_invariance_failures": rename_fail,
                "permutation_invariance_failures": perm_fail,
                "independent_gi_checks": gi_checked,
                "independent_gi_pass": gi_pass,
                "cell_minutes": round((time.time() - cell_t0) / 60, 1),
            }
            if verr:
                failures.append(f"{key}: {verr} verifier errors")
            if rename_fail:
                failures.append(f"{key}: {rename_fail} renaming failures (ALL-family test)")
            if perm_fail:
                failures.append(f"{key}: {perm_fail} permutation failures (ALL-family test)")

    cross_amb = sum(1 for cells in cross_cell.values() if len(set(cells)) > 1)
    if cross_amb:
        failures.append(f"cross-cell ambiguity: {cross_amb}")

    evidence = {
        "schema_id": "E0-V061-STRUCTURAL-PREFLIGHT-R2-CORRECTNESS-v0",
        "authority": "V06-STRUCTSIG-REMEDIATION-2",
        "mode": "correctness",
        "code_git_commit": git_head(),
        "working_tree_clean_at_start": tree_clean(),
        "pilot_namespace": NS_CORRECTNESS,
        "pilot_namespace_burned": True,
        "structsig": "structsig_r2: full IR, no local symmetry shortcuts, verified-automorphism orbit pruning only",
        "adversarial_suite": adversarial,
        "per_cell": per_cell,
        "independent_gi_failures": all_group_checks[:20],
        "cross_cell_ambiguity": cross_amb,
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
        "wall_seconds": round(time.time() - started, 1),
    }
    out = EVIDENCE / "STRUCTURAL_PREFLIGHT_R2.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: evidence[k] for k in ("status", "adversarial_suite", "per_cell", "failures")}, indent=2))
    return evidence


def _random_renaming_multi(fam):
    rng = random.Random(999)
    preds, ents = set(), set()
    for v in fam["variants"]:
        for f in v["facts"]:
            preds.add(f["pred"]); ents.add(f["term"])
        for r in v["rules"]:
            for p in r["premises"] + [r["conclusion"]]:
                preds.add(p["pred"])
                if p["term"] != "x":
                    ents.add(p["term"])
    np_ = [f"MP{i:03d}" for i in range(len(preds))]
    ne = [f"ME{i:03d}" for i in range(len(ents))]
    rng.shuffle(np_); rng.shuffle(ne)
    return (dict(zip(sorted(preds), np_)), dict(zip(sorted(ents), ne)))


# --------------------------------------------------------------------------
# depletion mode
# --------------------------------------------------------------------------

def run_depletion(discovery_per_depth: int, fresh_request: int) -> dict:
    started = time.time()
    burn = set()
    discovery = {}
    for depth in range(1, 5):
        t0 = time.time()
        fams = build_families(NS_DEPLETION_DISCOVERY, "train_pool_sim", "ID", depth, range(discovery_per_depth))
        seen = set()
        curve = []
        for i, fam in enumerate(fams):
            try:
                sig = structsig_r2.family_signature(fam["variants"])
            except (structsig_r2._BranchCapExceeded, structsig_r2._WallGuardExceeded):
                continue
            seen.add(sig)
            burn.add((depth, sig))
            if (i + 1) % (discovery_per_depth // 8) == 0:
                curve.append({"n": i + 1, "unique": len(seen)})
        discovery[f"ID|d{depth}"] = {
            "families": discovery_per_depth,
            "unique_signatures": len(seen),
            "discovery_curve": curve,
            "new_signatures_in_last_quarter": len(seen) - (curve[-2]["unique"] if len(curve) >= 2 else 0),
            "minutes": round((time.time() - t0) / 60, 1),
        }
        print(f"[depletion] ID d{depth}: {discovery_per_depth} families -> {len(seen)} unique "
              f"({discovery[f'ID|d{depth}']['minutes']} min)", flush=True)

    # fresh-namespace acceptance: request families disjoint from the burn
    fresh_stats = {}
    for surface in ("ID", "STRUCT"):
        for depth in range(1, 5):
            accepted = rejected = 0
            t0 = time.time()
            for fi in range(fresh_request * 4):  # candidate pool 4x the demand
                if accepted >= fresh_request:
                    break
                fam = build_families(NS_DEPLETION_FRESH, "qual_sim", surface, depth, range(fi, fi + 1))[0]
                try:
                    sig = structsig_r2.family_signature(fam["variants"])
                except (structsig_r2._BranchCapExceeded, structsig_r2._WallGuardExceeded):
                    rejected += 1
                    continue
                if (depth, sig) in burn:
                    rejected += 1
                else:
                    accepted += 1
            fresh_stats[f"{surface}|d{depth}"] = {
                "demand": fresh_request,
                "accepted_disjoint": accepted,
                "rejected_burned_signature": rejected,
                "acceptance_rate": round(accepted / max(1, accepted + rejected), 4),
                "exhausted_pool": accepted < fresh_request,
                "minutes": round((time.time() - t0) / 60, 1),
            }
            print(f"[depletion] fresh {surface} d{depth}: accepted {accepted}/{fresh_request} "
                  f"(rej {rejected})", flush=True)

    evidence = {
        "schema_id": "E0-V061-STRUCTURAL-PREFLIGHT-R2-DEPLETION-v0",
        "authority": "V06-STRUCTSIG-REMEDIATION-2 item 6; corpus feasibility work, not qualification execution",
        "mode": "depletion",
        "code_git_commit": git_head(),
        "namespaces": {"discovery": NS_DEPLETION_DISCOVERY, "fresh": NS_DEPLETION_FRESH, "all_burned": True},
        "assumptions": [
            "discovery simulates the MSEL train-pool burn at 32,000 ID families/depth (ID surface only; STRUCT train-pool signatures not simulated - stated limitation)",
            "fresh-namespace acceptance measures the rate at which generator-drawn families are signature-disjoint from the simulated burn",
            "burn keyed by (depth, signature); cross-depth signature sharing was measured as zero in the correctness pilot, so per-depth keying matches the observed generator behavior",
        ],
        "discovery": discovery,
        "burn_total_signatures": len(burn),
        "fresh_acceptance": fresh_stats,
        "remaining_support_question": "acceptance rate at qualification-scale demand after the simulated burn",
        "wall_seconds": round(time.time() - started, 1),
    }
    out = EVIDENCE / "STRUCTURAL_DEPLETION_R2.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"discovery": {k: {kk: vv for kk, vv in v.items() if kk != 'discovery_curve'} for k, v in discovery.items()}, "burn_total": len(burn), "fresh": fresh_stats}, indent=2))
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["correctness", "depletion"], required=True)
    parser.add_argument("--families-per-depth", type=int, default=2000)
    parser.add_argument("--discovery-per-depth", type=int, default=32000)
    parser.add_argument("--fresh-request", type=int, default=2000)
    args = parser.parse_args()
    if args.mode == "correctness":
        ev = run_correctness(args.families_per_depth)
        raise SystemExit(0 if ev["status"] == "PASS" else 1)
    run_depletion(args.discovery_per_depth, args.fresh_request)


if __name__ == "__main__":
    main()
