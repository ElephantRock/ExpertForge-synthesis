"""CMDR-StructSig-v1 (remediated r2) — exact canonical structural signature.

V06-STRUCTSIG-REMEDIATION-2. Supersedes structsig_r1 (a67defe), whose
_cell_bulk_safe() local masked-edge shortcut was falsified by the authority's
directed-cycle counterexample (all four cycle nodes share a masked local edge
multiset, yet the cycle's automorphism group is dihedral, not the full
symmetric group, so concrete-name-ordered bulk assignment breaks renaming
invariance). r1 evidence is preserved as an incident record; the generator
and proof verifier from cfe0840 remain byte-identical.

Exact construction (per variant):
  1. Typed incidence structure: predicate nodes, entity nodes, fact nodes,
     rule nodes, and a query node; typed directed incidence edges with
     premise/conclusion/polarity/variable-vs-grounded roles; depth stratum
     as a graph-level annotation. (Encoded as a colored hypergraph; refined
     as node colors + hyperedge membership.)
  2. Full-multiset Weisfeiler-Lehman refinement to a stable class count,
     full-length SHA-256 color values.
  3. Canonical labeling by individualization-refinement with NO local
     symmetry shortcuts. At each non-discrete cell EVERY member is branched
     on; the canonical serialization is the minimum over all leaves. The
     only pruning is VERIFIED-AUTOMORPHISM ORBIT PRUNING: when two branches
     produce identical leaf serializations with aligned color sequences,
     the induced node bijection is explicitly verified to preserve the
     complete incidence structure (hyperedges, roles, signs) before being
     recorded as an automorphism; branches whose target node lies in the
     orbit of an already-explored branch node (under the verified
     automorphism group) are skipped, because a verified automorphism maps
     the skipped subtree isomorphically onto the explored one. This is the
     standard automorphism-pruning of exact canonical labelers restricted
     to explicitly verified automorphisms — formally sound, and it handles
     the disconnected-twin cells that would otherwise branch factorially.
  4. Family signature = SHA-256 over the canonical JSON of the SORTED
     full multiset of the three variant canonical representations
     (preserved from r1, which the ruling accepted).
"""

from __future__ import annotations

import hashlib
import json

_BRANCH_CAP = 200000
_WALL_GUARD_SECONDS = 20.0


def _h(*parts) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()


class _BranchCapExceeded(RuntimeError):
    pass


class _WallGuardExceeded(RuntimeError):
    """Per-variant guard: exceeded time is reported, never silently inexact."""


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


def _precompute_incidence(facts, rules, query, preds, ents):
    """Branch-invariant incidence index: for each node, the list of
    (role_tag, edge_index) memberships. Roles: fact, prem, concl, query.
    Edge colors are recomputed per refinement round; membership lists are not."""
    incid = {("P", p): [] for p in preds}
    for e in ents:
        incid.setdefault(("E", e), [])
    for i, f in enumerate(facts):
        incid[("P", f[1])].append(("fact", i))
        if f[2] != "x":
            incid.setdefault(("E", f[2]), []).append(("fact", i))
    for i, (prem, concl) in enumerate(rules):
        for l in prem:
            incid[("P", l[1])].append(("prem", i))
            if l[2] != "x":
                incid.setdefault(("E", l[2]), []).append(("prem", i))
        incid[("P", concl[1])].append(("concl", i))
        if concl[2] != "x":
            incid.setdefault(("E", concl[2]), []).append(("concl", i))
    if query[1] is not None:
        incid[("P", query[1])].append(("query", -1))
    if query[2] != "x":
        incid.setdefault(("E", query[2]), []).append(("query", -1))
    return incid


def _refine(colors, facts, rules, query, incid=None):
    """Full-multiset WL refinement to stable class count (colors only split).
    Uses the precomputed incidence index when provided."""
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
        edge_col = {"fact": fact_cols, "prem": rule_cols, "concl": rule_cols}
        new = dict(colors)
        if incid is not None:
            for node, memberships in incid.items():
                incident = []
                for tag, idx in memberships:
                    if tag == "query":
                        incident.append("query")
                    else:
                        incident.append(tag + ":" + edge_col[tag][idx])
                new[node] = _h(node[0] + "C", colors[node], sorted(incident))
        else:
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
        if len(set(new.values())) == len(set(colors.values())):
            return colors
        colors = new


def _serialize(colors, facts, rules, query, depth):
    """Serialize at a DISCRETE coloring; returns (canonical_string, ordered
    (node, color) pairs for automorphism extraction)."""
    pred_nodes = sorted((colors[("P", p)], p) for p in {k[1] for k in colors if k[0] == "P"})
    ent_nodes = sorted((colors[("E", e)], e) for e in {k[1] for k in colors if k[0] == "E"})
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
        "v": "CMDR-StructSig-v1-r2",
        "depth": depth,
        "facts": sorted(lit(f) for f in facts),
        "rules": sorted(rule_ser(prem, concl) for prem, concl in rules),
        "query": lit(query),
    }
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    order = ([("P", p) for _, p in pred_nodes] + [("E", e) for _, e in ent_nodes])
    return blob, order


def _verify_automorphism(sigma: dict, facts, rules, query) -> bool:
    """Explicitly verify that node bijection sigma preserves the complete
    typed incidence structure (facts, rules with premise/conclusion roles,
    polarity, variable-vs-grounded terms, query). Only verified maps are
    ever used for pruning."""

    def map_lit(l):
        return (l[0], sigma.get(("P", l[1]), ("P", l[1]))[1], sigma.get(("E", l[2]), ("E", l[2]))[1] if l[2] != "x" else "x")

    mapped_facts = sorted(map_lit(f) for f in facts)
    if mapped_facts != sorted(facts):
        return False
    mapped_rules = sorted(
        (sorted(map_lit(p) for p in prem), map_lit(concl)) for prem, concl in rules
    )
    if mapped_rules != sorted((sorted(prem), concl) for prem, concl in rules):
        return False
    if map_lit(query) != query:
        return False
    return True


class _OrbitTracker:
    """Union-find over discovered VERIFIED automorphisms."""

    def __init__(self, nodes):
        self.parent = {n: n for n in nodes}

    def find(self, n):
        while self.parent[n] != n:
            self.parent[n] = self.parent[self.parent[n]]
            n = self.parent[n]
        return n

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def same(self, a, b):
        return self.find(a) == self.find(b)


def _search(colors, facts, rules, query, depth, budget, orbits: _OrbitTracker, incid=None):
    """Exact IR canonical labeling: branch on EVERY member of the smallest
    non-singleton cell, except members already covered by an explored branch
    node under the verified-automorphism orbit partition. Returns the set of
    (serialization, node-order) leaves; the canonical form is the minimum
    serialization."""
    import time as _time

    if _time.monotonic() > budget[1]:
        raise _WallGuardExceeded(f"IR wall guard {_WALL_GUARD_SECONDS:.0f}s exceeded")
    colors = _refine(colors, facts, rules, query, incid)
    cells: dict[str, list] = {}
    for node, c in colors.items():
        cells.setdefault(c, []).append(node)
    nonsingle = [(len(v), c, v) for c, v in cells.items() if len(v) > 1]
    if not nonsingle:
        blob, order = _serialize(colors, facts, rules, query, depth)
        return {(blob, tuple(order))}
    nonsingle.sort(key=lambda t: (t[0], t[1]))
    _, _, target = nonsingle[0]

    best_leaves: set = set()
    explored: list = []
    for node in sorted(target, key=lambda n: (n[0], n[1])):
        if any(orbits.same(node, ex) for ex in explored):
            continue  # verified automorphism maps this subtree onto an explored one
        budget[0] -= 1
        if budget[0] < 0:
            raise _BranchCapExceeded(f"IR branch cap {_BRANCH_CAP} exceeded")
        explored.append(node)
        branched = dict(colors)
        branched[node] = _h("INDIV", colors[node], node[0])
        leaves = _search(branched, facts, rules, query, depth, budget, orbits, incid)
        best_leaves |= leaves

        # automorphism harvesting: align leaves of this branch with the best
        # leaf so far; identical serializations with aligned color sequences
        # induce a node bijection that we verify explicitly before use
        if len(best_leaves) > 1:
            anchor = min(best_leaves, key=lambda t: t[0])
            for blob, order in leaves:
                if (blob, order) == anchor:
                    continue
                sigma = dict(zip(anchor[1], order))
                if len(set(sigma.values())) != len(sigma):
                    continue
                if _verify_automorphism(sigma, facts, rules, query):
                    for a, b in zip(anchor[1], order):
                        orbits.union(a, b)
    return best_leaves


def variant_canonical(example: dict) -> str:
    """Exact canonical serialization of ONE variant (pivot binding included)."""
    import time as _time

    facts, rules, query, depth, preds, ents = _extract(example)
    colors = _initial_colors(preds, ents, query)
    incid = _precompute_incidence(facts, rules, query, preds, ents)
    budget = [_BRANCH_CAP, _time.monotonic() + _WALL_GUARD_SECONDS]
    orbits = _OrbitTracker(list(colors.keys()))
    leaves = _search(colors, facts, rules, query, depth, budget, orbits, incid)
    return min(blob for blob, _ in leaves)


def variant_digest(example: dict) -> str:
    return hashlib.sha256(variant_canonical(example).encode("utf-8")).hexdigest()


def family_canonical(family_variants: list[dict]) -> str:
    if len(family_variants) != 3:
        raise AssertionError("family_canonical expects exactly three variants")
    reps = sorted(variant_canonical(v) for v in family_variants)
    return json.dumps(
        {"v": "CMDR-StructSig-v1-r2-family", "orbit": reps},
        sort_keys=True, separators=(",", ":"),
    )


def family_signature(family_variants: list[dict]) -> str:
    return hashlib.sha256(family_canonical(family_variants).encode("utf-8")).hexdigest()
