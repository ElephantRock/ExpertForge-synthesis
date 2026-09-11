"""CMDR-StructSig-v1 (remediated r1) — exact canonical structural signature.

V06-STRUCTSIG-REMEDIATION-1. Supersedes structsig.py (74a80fd), whose
signature was ruled STRUCTURAL_PREFLIGHT_INVALID_FOR_SUPPORT_CONCLUSION for
four defects: (1) min-of-orbit family collapse, (2) set()-destroyed incidence
multiplicity, (3) 1-WL colors treated as an exact canonical labeling,
(4) truncated 64-bit internal hash channel. The generator and proof verifier
from cfe0840 are untouched.

Exact construction:
  Per VARIANT, an exact typed canonical form is produced by
  individualization-refinement (IR): full-multiset WL color refinement
  (full-length SHA-256 color values) stabilizes the partition; while any
  non-singleton color cell remains, each member of the smallest such cell is
  individualized in turn and refinement recurses (exhaustive over the
  symmetry search; hard branch cap fail-closes rather than returning a
  non-exact form). The canonical form of the variant is the lexicographically
  minimal serialization over all explored branches — an exact canonical
  labeling: two variants are isomorphic (under the contract's retained
  fields: fact/rule/query hypergraph topology, polarity, arity, direction,
  depth stratum, distractor connectivity; with names abstracted and
  semantically unordered sets sorted) iff their canonical serializations are
  identical.

  The FAMILY signature hashes a canonical JSON object containing the SORTED
  MULTISET of all three canonical variant representations — lossless over the
  counterfactual orbit (two families sharing one orbit member but differing
  in others receive different signatures).
"""

from __future__ import annotations

import hashlib
import json

_BRANCH_CAP = 5000


def _h(*parts) -> str:
    return hashlib.sha256(
        "|".join(map(str, parts)).encode("utf-8")
    ).hexdigest()


class _BranchCapExceeded(RuntimeError):
    pass


def _extract(example: dict):
    facts = [(f["sign"], f["pred"], f["term"]) for f in example["facts"]]
    rules = [
        (
            sorted((p["sign"], p["pred"], p["term"]) for p in r["premises"]),
            (r["conclusion"]["sign"], r["conclusion"]["pred"], r["conclusion"]["term"]),
        )
        for r in example["rules"]
    ]
    query = (example["query"]["sign"], example["query"]["pred"], example["query"]["term"])
    depth = example["reasoning_depth_stratum"]

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


def _initial_colors(preds, ents, query):
    colors = {}
    for p in preds:
        colors[("P", p)] = "PRED"
    for e in ents:
        colors[("E", e)] = "ENT"
    colors[("P", query[1])] = "PRED+QUERY"
    if query[2] != "x":
        colors[("E", query[2])] = "ENT+QUERY"
    return colors


def _refine(colors, facts, rules, query):
    """Full-multiset WL refinement to stability. Incident multisets keep
    multiplicity (sorted lists, never set())."""
    while True:
        fact_cols = {}
        for i, f in enumerate(facts):
            fact_cols[i] = _h("F", f[0], colors[("P", f[1])], colors.get(("E", f[2]), "X"), f[2] == "x")
        rule_cols = {}
        for i, (prem, concl) in enumerate(rules):
            rule_cols[i] = _h(
                "R",
                sorted(_h("L", l[0], colors[("P", l[1])], colors.get(("E", l[2]), "X"), l[2] == "x") for l in prem),
                _h("L", concl[0], colors[("P", concl[1])], colors.get(("E", concl[2]), "X"), concl[2] == "x"),
            )

        new = dict(colors)
        for p in {k[1] for k in colors if k[0] == "P"}:
            incident = []
            for i, f in enumerate(facts):
                if f[1] == p:
                    incident.append("fact:" + fact_cols[i])
            for i, (prem, concl) in enumerate(rules):
                if any(l[1] == p for l in prem):
                    incident.append("prem:" + rule_cols[i])
                if concl[1] == p:
                    incident.append("concl:" + rule_cols[i])
            if query[1] == p:
                incident.append("query")
            new[("P", p)] = _h("PC", colors[("P", p)], sorted(incident))
        for e in {k[1] for k in colors if k[0] == "E"}:
            incident = []
            for i, f in enumerate(facts):
                if f[2] == e:
                    incident.append("fact:" + fact_cols[i])
            for i, (prem, concl) in enumerate(rules):
                if any(l[2] == e for l in prem):
                    incident.append("prem:" + rule_cols[i])
                if concl[2] == e:
                    incident.append("concl:" + rule_cols[i])
            if query[2] == e:
                incident.append("query")
            new[("E", e)] = _h("EC", colors[("E", e)], sorted(incident))
        if len({new[k] for k in new}) == len({colors[k] for k in colors}):
            # refinement only ever splits cells (new color hashes the old),
            # so an unchanged class count means the partition is stable;
            # color VALUES keep changing under re-hashing, so value equality
            # would never terminate
            return colors
        colors = new


def _serialize(colors, facts, rules, query, depth):
    """Serialize at a DISCRETE coloring: node order = sorted final color."""
    pred_nodes = sorted(((colors[("P", p)], p) for p in {k[1] for k in colors if k[0] == "P"}))
    ent_nodes = sorted(((colors[("E", e)], e) for e in {k[1] for k in colors if k[0] == "E"}))
    p_idx = {p: i for i, (_, p) in enumerate(pred_nodes)}
    e_idx = {e: i for i, (_, e) in enumerate(ent_nodes)}

    def lit(l):
        term = "x" if l[2] == "x" else f"e{e_idx[l[2]]}"
        return [l[0], f"p{p_idx[l[1]]}", term]

    def rule_ser(prem, concl):
        return json.dumps(
            {"premises": sorted(lit(l) for l in prem), "conclusion": lit(concl)},
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        )

    canonical = {
        "v": "CMDR-StructSig-v1-r1",
        "depth": depth,
        "facts": sorted(lit(f) for f in facts),
        "rules": sorted(rule_ser(prem, concl) for prem, concl in rules),
        "query": lit(query),
    }
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _masked_edge_multiset(node, cell_set, colors, facts, rules, query):
    """For member u of cell C: the sorted multiset of descriptors of every
    hyperedge containing u, where u's own slot is recorded by role/sign and
    every OTHER member of C is masked. Non-cell content is represented by its
    stable color. If all members of C share this multiset, then any
    permutation of C maps each masked-equal edge class onto itself as a
    multiset of (class, index) pairs — the sorted serialization, which never
    records edge identity, is invariant. This unifies and generalizes the
    mutual-hyperedge-free and co-twin cases."""
    kind, name = node
    sig = []

    def lit_desc(l, own: bool):
        if kind == "P" and l[1] == name and own:
            p = "SELF"
        elif ("P", l[1]) in cell_set:
            p = "MASK"
        else:
            p = colors[("P", l[1])]
        if l[2] == "x":
            t = "x"
        elif kind == "E" and l[2] == name and own:
            t = "SELF"
        elif ("E", l[2]) in cell_set:
            t = "MASK"
        else:
            t = colors[("E", l[2])]
        return (l[0], p, t)

    for i, f in enumerate(facts):
        if (kind == "P" and f[1] == name) or (kind == "E" and f[2] == name):
            sig.append(("fact", lit_desc(f, True)))
    for i, (prem, concl) in enumerate(rules):
        for l in prem:
            if (kind == "P" and l[1] == name) or (kind == "E" and l[2] == name):
                others = sorted(str(lit_desc(x, False)) for x in prem if x is not l)
                sig.append(("prem", str(lit_desc(l, True)), others))
        if (kind == "P" and concl[1] == name) or (kind == "E" and concl[2] == name):
            sig.append(("concl", str(lit_desc(concl, True))))
    if (kind == "P" and query[1] == name) or (kind == "E" and query[2] == name):
        sig.append(("query", str(lit_desc(query, True))))
    return sorted(str(x) for x in sig)


def _cell_bulk_safe(target: list, colors, facts, rules, query) -> bool:
    """Exact bulk-individualization safety: all members share the identical
    masked-edge multiset (see _masked_edge_multiset for the invariance
    argument)."""
    cell_set = set(target)
    sigs = [_masked_edge_multiset(node, cell_set, colors, facts, rules, query) for node in target]
    return all(s == sigs[0] for s in sigs)


def _search(colors, facts, rules, query, depth, budget):
    """IR search: refine; if discrete serialize; else branch on the smallest
    non-singleton cell. Bulk-safe cells (interchangeable members — identical
    masked-edge multisets) are individualized in one branch; only genuinely
    ambiguous cells branch."""
    colors = _refine(colors, facts, rules, query)
    cells: dict[str, list] = {}
    for node, c in colors.items():
        cells.setdefault(c, []).append(node)
    nonsingle = [(len(v), c, v) for c, v in cells.items() if len(v) > 1]
    if not nonsingle:
        return {_serialize(colors, facts, rules, query, depth)}
    nonsingle.sort(key=lambda t: (t[0], t[1]))
    _, _, target = nonsingle[0]

    if _cell_bulk_safe(target, colors, facts, rules, query):
        budget[0] -= 1
        if budget[0] < 0:
            raise _BranchCapExceeded(f"IR branch cap {_BRANCH_CAP} exceeded")
        branched = dict(colors)
        for i, node in enumerate(sorted(target, key=lambda n: n[1])):
            branched[node] = _h("BULK", colors[node], node[0], i)
        return _search(branched, facts, rules, query, depth, budget)

    results = set()
    for node in sorted(target, key=lambda n: n[1]):
        budget[0] -= 1
        if budget[0] < 0:
            raise _BranchCapExceeded(f"IR branch cap {_BRANCH_CAP} exceeded")
        branched = dict(colors)
        branched[node] = _h("INDIV", colors[node], node[0])
        results |= _search(branched, facts, rules, query, depth, budget)
    return results


def variant_canonical(example: dict) -> str:
    """Exact canonical serialization of ONE variant (pivot binding included)."""
    facts, rules, query, depth, preds, ents = _extract(example)
    colors = _initial_colors(preds, ents, query)
    budget = [_BRANCH_CAP]
    leaves = _search(colors, facts, rules, query, depth, budget)
    return min(leaves)


def variant_digest(example: dict) -> str:
    return hashlib.sha256(variant_canonical(example).encode("utf-8")).hexdigest()


def family_canonical(family_variants: list[dict]) -> str:
    """Canonical JSON of the SORTED MULTISET of the three canonical variant
    representations (lossless over the counterfactual orbit)."""
    if len(family_variants) != 3:
        raise AssertionError("family_canonical expects exactly three variants")
    reps = sorted(variant_canonical(v) for v in family_variants)
    return json.dumps({"v": "CMDR-StructSig-v1-r1-family", "orbit": reps}, sort_keys=True, separators=(",", ":"))


def family_signature(family_variants: list[dict]) -> str:
    return hashlib.sha256(family_canonical(family_variants).encode("utf-8")).hexdigest()
