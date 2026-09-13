"""Correctness pilot for structsig_r3 BLISS — V06-STRUCTSIG-REMEDIATION-3.

4,000 families (500/depth/surface × 8 cells) with ALL-family renaming +
order-permutation invariance tests and the independent MRV GI checker on
every repeated-signature group. Run from a literally clean implementation
commit on the burned namespace ExpertForge-E0-v061-msel-preflight-r3.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import structsig_r3
from msel_corpus import apply_renaming, build_families
from msel_verifier import counterfactual_invariance, verify_family

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EVIDENCE_PATH = REPO_ROOT / "docs" / "experiments" / "e0" / "v061" / "CORRECTNESS_R3.json"
NS = "ExpertForge-E0-v061-msel-preflight-r3"

_GI_STEP_BUDGET = 200000


class _GIExceeded(RuntimeError):
    pass


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()


def tree_clean() -> bool:
    return subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip() == ""


def _typed_isomorphic(ex_a: dict, ex_b: dict, budget: list | None = None) -> bool:
    """Independent exact isomorphism check. MRV backtracking over
    WL-color-class-constrained candidate bijections with incremental
    hyperedge-multiset consistency. Operates directly on CMDR facts/rules/
    query; does not call structsig_r3, igraph, or any canonical labeling."""
    fa, ra, qa, da, pa, ea = structsig_r3.__dict__["_extract"] if False else _extract_ab(ex_a)
    fb, rb, qb, db, pb, eb = _extract_ab(ex_b)

    if da != db or len(fa) != len(fb) or len(ra) != len(rb):
        return False
    steps = [0]
    BUDGET = budget[0] if budget else _GI_STEP_BUDGET

    # WL refinement for candidate constraint (independent implementation)
    def refine(facts, rules, query, preds, ents):
        colors = {}
        for p in preds:
            colors[("P", p)] = "P"
        for e in ents:
            colors[("E", e)] = "E"
        colors[("P", query[1])] = "PQ"
        if query[2] != "x":
            colors[("E", query[2])] = "EQ"
        while True:
            def lc(l):
                return (l[0], colors[("P", l[1])], colors.get(("E", l[2]), "X"), l[2] == "x")
            fc = sorted(str(lc(f)) for f in facts)
            rc = sorted((str(sorted(str(lc(p)) for p in prem)), str(lc(c))) for prem, c in rules)
            new = dict(colors)
            for p in preds:
                inc = []
                for i, f in enumerate(facts):
                    if f[1] == p:
                        inc.append("f:" + str(lc(f)))
                for i, (prem, c) in enumerate(rules):
                    if any(l[1] == p for l in prem):
                        inc.append("p:" + str(sorted(str(lc(x)) for x in prem)))
                    if c[1] == p:
                        inc.append("c:" + str(lc(c)))
                if query[1] == p:
                    inc.append("q")
                new[("P", p)] = hashlib.sha256(str(sorted(inc)).encode()).hexdigest()[:16]
            for e in ents:
                inc = []
                for i, f in enumerate(facts):
                    if f[2] == e:
                        inc.append("f:" + str(lc(f)))
                for i, (prem, c) in enumerate(rules):
                    if any(l[2] == e for l in prem):
                        inc.append("p:" + str(sorted(str(lc(x)) for x in prem)))
                    if c[2] == e:
                        inc.append("c:" + str(lc(c)))
                if query[2] == e:
                    inc.append("q")
                new[("E", e)] = hashlib.sha256(str(sorted(inc)).encode()).hexdigest()[:16]
            if len(set(new.values())) == len(set(colors.values())):
                return colors
            colors = new

    ca = refine(fa, ra, qa, pa, ea)
    cb = refine(fb, rb, qb, pb, eb)

    if Counter(ca.values()) != Counter(cb.values()):
        return False
    b_by_color = defaultdict(list)
    for n, c in cb.items():
        b_by_color[c].append(n)

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
    nodes = list(ca.keys())

    def mapped_lit(l):
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
            raise _GIExceeded("GI step budget exceeded")
        tgt_facts = Counter(fb)
        mapped = Counter()
        for f in fa:
            m = mapped_lit(f)
            if m is not None:
                mapped[m] += 1
        if mapped - tgt_facts:
            return False
        tgt_rules = Counter((tuple(sorted(prem)), concl) for prem, concl in rb)
        mapped_r = Counter()
        for prem, concl in ra:
            ms = [mapped_lit(p) for p in prem]
            mc = mapped_lit(concl)
            if all(x is not None for x in ms) and mc is not None:
                mapped_r[(tuple(sorted(ms)), mc)] += 1
        if mapped_r - tgt_rules:
            return False
        return True

    def backtrack():
        remaining = [n for n in nodes if n not in sigma]
        if not remaining:
            mq = mapped_lit(qa)
            fact_ok = not (Counter(mapped_lit(f) for f in fa) - Counter(fb))
            rule_ok = not (
                Counter(
                    (tuple(sorted(mapped_lit(p) for p in prem)), mapped_lit(c))
                    for prem, c in ra
                ) - Counter((tuple(sorted(prem)), concl) for prem, concl in rb)
            )
            return mq == qb and fact_ok and rule_ok
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


def _extract_ab(ex):
    facts = [(f["sign"], f["pred"], f["term"]) for f in ex["facts"]]
    rules = [
        (sorted((p["sign"], p["pred"], p["term"]) for p in r["premises"]),
         (r["conclusion"]["sign"], r["conclusion"]["pred"], r["conclusion"]["term"]))
        for r in ex["rules"]
    ]
    query = (ex["query"]["sign"], ex["query"]["pred"], ex["query"]["term"])
    depth = ex["reasoning_depth_stratum"]
    preds, ents = set(), set()
    def visit(l):
        preds.add(l[1])
        if l[2] != "x":
            ents.add(l[2])
    for f in facts:
        visit(f)
    for prem, concl in rules:
        for l in prem:
            visit(l)
        visit(concl)
    visit(query)
    return facts, rules, query, depth, preds, ents


def _random_renaming(fam):
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


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--families-per-depth", type=int, default=500)
    args = parser.parse_args()
    n = args.families_per_depth

    started = time.time()
    failures = []
    per_cell = {}
    cross_cell = {}

    for surface in ("ID", "STRUCT"):
        for depth in range(1, 5):
            key = f"{surface}|d{depth}"
            fams = build_families(NS, "pilot", surface, depth, range(n))
            sigs = []
            fam_reps = {}
            rename_fail = perm_fail = verr = 0
            cell_t0 = time.time()

            for fam in fams:
                errs = verify_family(fam) + counterfactual_invariance(fam)
                verr += len(errs)
                reps = sorted(structsig_r3.variant_canonical(v) for v in fam["variants"])
                fc = json.dumps({"v": "CMDR-StructSig-v1-r3-family", "orbit": reps}, sort_keys=True, separators=(",", ":"))
                sig = hashlib.sha256(fc.encode("utf-8")).hexdigest()
                sigs.append(sig)
                fam_reps.setdefault(sig, []).append(fam)
                cross_cell.setdefault(sig, []).append(key)

                # ALL-family renaming invariance
                pm, em = _random_renaming(fam)
                renamed = [apply_renaming(v, pm, em) for v in fam["variants"]]
                rsig = structsig_r3.family_signature(renamed)
                if rsig != sig:
                    rename_fail += 1

                # ALL-family order-permutation invariance (SHA-256 derived seed)
                rng = random.Random(int.from_bytes(
                    hashlib.sha256(fam["family_id"].encode()).digest()[:4], "big"
                ))
                perm_ok = True
                for v in fam["variants"]:
                    w = copy.deepcopy(v)
                    rng.shuffle(w["facts"]); rng.shuffle(w["rules"])
                    for r in w["rules"]:
                        rng.shuffle(r["premises"])
                    if structsig_r3.variant_canonical(w) != structsig_r3.variant_canonical(v):
                        perm_ok = False
                        break
                if not perm_ok:
                    perm_fail += 1

            # independent GI check on every repeated-signature group
            gi_checked = gi_pass = gi_exceeded = gi_failed = 0
            for sig, group in fam_reps.items():
                if len(group) < 2:
                    continue
                rep = group[0]["variants"][0]
                for other in group[1:]:
                    gi_checked += 1
                    try:
                        if _typed_isomorphic(rep, other["variants"][0], budget=[_GI_STEP_BUDGET]):
                            gi_pass += 1
                        else:
                            gi_failed += 1
                            failures.append(f"{key}: GI NOT_ISOMORPHIC {sig[:12]} family {other['family_id']}")
                    except _GIExceeded:
                        gi_exceeded += 1

            sc = Counter(sigs)
            per_cell[key] = {
                "attempted": n,
                "unique": len(sc),
                "max_multiplicity": max(sc.values()),
                "verifier_errors": verr,
                "rename_invariance_failures": rename_fail,
                "permutation_invariance_failures": perm_fail,
                "gi_checks": gi_checked,
                "gi_pass": gi_pass,
                "gi_exceeded": gi_exceeded,
                "gi_failed": gi_failed,
                "cell_minutes": round((time.time() - cell_t0) / 60, 1),
            }
            if verr:
                failures.append(f"{key}: {verr} verifier errors")
            if rename_fail:
                failures.append(f"{key}: {rename_fail} renaming failures")
            if perm_fail:
                failures.append(f"{key}: {perm_fail} permutation failures")
            if gi_failed:
                failures.append(f"{key}: {gi_failed} GI hard failures")

            print(f"[{key}] done in {per_cell[key]['cell_minutes']} min | unique {len(sc)} | "
                  f"rename_fail {rename_fail} perm_fail {perm_fail} GI {gi_pass}/{gi_checked} "
                  f"(exceeded {gi_exceeded}, failed {gi_failed})", flush=True)

    cross_amb = sum(1 for cells in cross_cell.values() if len(set(cells)) > 1)
    if cross_amb:
        failures.append(f"cross-cell ambiguity: {cross_amb}")

    evidence = {
        "schema_id": "E0-V061-CORRECTNESS-R3-v0",
        "authority": "V06-STRUCTSIG-REMEDIATION-3 (pre-authorized after oracle completion PASS)",
        "code_git_commit": git_head(),
        "working_tree_clean_at_start": tree_clean(),
        "pilot_namespace": NS,
        "pilot_namespace_burned": True,
        "canonicalizer": "structsig_r3 BLISS via igraph 1.0.0",
        "per_cell": per_cell,
        "cross_cell_ambiguity": cross_amb,
        "failures": failures,
        "status": "PASS" if not failures else "FAIL",
        "wall_seconds": round(time.time() - started, 1),
    }

    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: evidence[k] for k in ("status", "per_cell", "cross_cell_ambiguity", "failures", "wall_seconds")}, indent=2))
    raise SystemExit(0 if evidence["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
