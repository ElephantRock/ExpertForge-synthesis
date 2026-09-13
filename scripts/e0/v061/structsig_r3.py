"""CMDR-StructSig-v1 (remediated r3) — BLISS-based exact canonical signature.

V06-STRUCTSIG-REMEDIATION-3. Replaces all custom IR canonical labeling with
igraph 1.0.0's Graph.canonical_permutation(color=...), which uses the BLISS
isomorphism algorithm. The custom-IR route is CLOSED/REJECTED (see
incidents/); this is the mature-package path the authority directed.

Construction (per variant):
  1. Encode the variant as a colored UNDIRECTED incidence graph with a frozen
     color vocabulary (see _COLOR_VOCABULARY). Vertex classes:
       - PREDICATE       (one per distinct predicate)
       - ENTITY          (one per distinct entity)
       - FACT_OWNER      (one per fact)
       - RULE_OWNER      (one per rule)
       - QUERY_OWNER     (exactly one)
       - LITERAL_*       (one per fact/premise/conclusion/query occurrence;
                          color encodes role + polarity + variable-vs-grounded)
     Literal-occurrence vertices connect to: their owner, their predicate
     vertex, and their entity vertex (when grounded).
     Depth is encoded via a DEPTH_n marker vertex connected to all owners.
     No concrete names, no surface label, no generator seed, no family ID,
     no fact/rule ordering.
  2. Call igraph's canonical_permutation(color=colors) — BLISS computes an
     exact canonical vertex ordering under the color constraint.
  3. Serialize: permute the vertex-color sequence to canonical order, sort
     the canonical edge set, wrap with schema tag + depth. SHA-256 the
     canonical JSON as the variant digest.
  4. Family signature = SHA-256 over canonical JSON of the SORTED multiset of
     all three variant canonical representations (unchanged from the ruling).
"""

from __future__ import annotations

import hashlib
import json

import igraph


def _h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# Frozen color vocabulary — small non-negative integers for BLISS
_COLOR_VOCABULARY = {
    "PREDICATE": 0,
    "ENTITY": 1,
    "FACT_OWNER": 2,
    "RULE_OWNER": 3,
    "QUERY_OWNER": 4,
    "LIT_FACT_PLUS_G": 5,
    "LIT_FACT_MINUS_G": 6,
    "LIT_FACT_PLUS_X": 7,
    "LIT_FACT_MINUS_X": 8,
    "LIT_PREM_PLUS_G": 9,
    "LIT_PREM_MINUS_G": 10,
    "LIT_PREM_PLUS_X": 11,
    "LIT_PREM_MINUS_X": 12,
    "LIT_CONCL_PLUS_G": 13,
    "LIT_CONCL_MINUS_G": 14,
    "LIT_CONCL_PLUS_X": 15,
    "LIT_CONCL_MINUS_X": 16,
    "LIT_QUERY_PLUS_G": 17,
    "LIT_QUERY_MINUS_G": 18,
    "LIT_QUERY_PLUS_X": 19,
    "LIT_QUERY_MINUS_X": 20,
    "DEPTH_1": 21,
    "DEPTH_2": 22,
    "DEPTH_3": 23,
    "DEPTH_4": 24,
}

_COLOR_NAMES = {v: k for k, v in _COLOR_VOCABULARY.items()}


def _lit_color(role: str, sign: str, term: str) -> int:
    grounded = "G" if term != "x" else "X"
    if sign == "+":
        polarity = "PLUS"
    elif sign == "-":
        polarity = "MINUS"
    else:
        raise ValueError(f"invalid polarity {sign!r}: must be '+' or '-'")
    key = f"LIT_{role}_{polarity}_{grounded}"
    if key not in _COLOR_VOCABULARY:
        raise ValueError(f"literal color not in frozen vocabulary: {key}")
    return _COLOR_VOCABULARY[key]


def _build_incidence_graph(example: dict):
    """Build the colored undirected incidence graph for one variant.
    Returns (igraph.Graph, list_of_colors, n_vertices)."""
    facts = example["facts"]
    rules = example["rules"]
    query = example["query"]
    depth = example["reasoning_depth_stratum"]

    vertices = []  # list of color integers
    edges = []

    def add_vertex(color: int) -> int:
        vertices.append(color)
        return len(vertices) - 1

    def connect(a: int, b: int) -> None:
        edges.append((a, b))

    # Predicate and entity vertices
    pred_ids, ent_ids = {}, {}
    all_preds, all_ents = set(), set()
    for f in facts:
        all_preds.add(f["pred"])
        if f["term"] != "x":
            all_ents.add(f["term"])
    for r in rules:
        for p in r["premises"] + [r["conclusion"]]:
            all_preds.add(p["pred"])
            if p["term"] != "x":
                all_ents.add(p["term"])
    all_preds.add(query["pred"])
    if query["term"] != "x":
        all_ents.add(query["term"])
    for p in sorted(all_preds):
        pred_ids[p] = add_vertex(_COLOR_VOCABULARY["PREDICATE"])
    for e in sorted(all_ents):
        ent_ids[e] = add_vertex(_COLOR_VOCABULARY["ENTITY"])

    # Depth marker — fail closed on invalid depth
    if depth not in (1, 2, 3, 4):
        raise ValueError(f"invalid reasoning depth {depth!r}: must be 1..4")
    depth_key = f"DEPTH_{depth}"
    depth_vid = add_vertex(_COLOR_VOCABULARY[depth_key])

    # Fact owners + literal occurrences
    for i, f in enumerate(facts):
        owner = add_vertex(_COLOR_VOCABULARY["FACT_OWNER"])
        connect(owner, depth_vid)
        lit = add_vertex(_lit_color("FACT", f["sign"], f["term"]))
        connect(lit, owner)
        connect(lit, pred_ids[f["pred"]])
        if f["term"] != "x":
            connect(lit, ent_ids[f["term"]])

    # Rule owners + premise/conclusion occurrences
    for i, r in enumerate(rules):
        owner = add_vertex(_COLOR_VOCABULARY["RULE_OWNER"])
        connect(owner, depth_vid)
        for p in r["premises"]:
            lit = add_vertex(_lit_color("PREM", p["sign"], p["term"]))
            connect(lit, owner)
            connect(lit, pred_ids[p["pred"]])
            if p["term"] != "x":
                connect(lit, ent_ids[p["term"]])
        c = r["conclusion"]
        lit = add_vertex(_lit_color("CONCL", c["sign"], c["term"]))
        connect(lit, owner)
        connect(lit, pred_ids[c["pred"]])
        if c["term"] != "x":
            connect(lit, ent_ids[c["term"]])

    # Query owner + literal
    q_owner = add_vertex(_COLOR_VOCABULARY["QUERY_OWNER"])
    connect(q_owner, depth_vid)
    ql = add_vertex(_lit_color("QUERY", query["sign"], query["term"]))
    connect(ql, q_owner)
    connect(ql, pred_ids[query["pred"]])
    if query["term"] != "x":
        connect(ql, ent_ids[query["term"]])

    g = igraph.Graph(n=len(vertices), edges=edges, directed=False)
    return g, vertices


def variant_canonical(example: dict) -> str:
    """Exact canonical serialization of ONE variant via BLISS."""
    g, colors = _build_incidence_graph(example)
    g.vs["color"] = colors
    perm = g.canonical_permutation(color=colors)
    gc = g.permute_vertices(perm)
    canon_color_names = [_COLOR_NAMES[c] for c in gc.vs["color"]]
    canon_edges = sorted(tuple(sorted(e)) for e in gc.get_edgelist())
    canonical = {
        "v": "CMDR-StructSig-v1-r3",
        "depth": example["reasoning_depth_stratum"],
        "colors": canon_color_names,
        "edges": [list(e) for e in canon_edges],
    }
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def variant_digest(example: dict) -> str:
    return hashlib.sha256(variant_canonical(example).encode("utf-8")).hexdigest()


def family_canonical(family_variants: list[dict]) -> str:
    if len(family_variants) != 3:
        raise AssertionError("family_canonical expects exactly three variants")
    reps = sorted(variant_canonical(v) for v in family_variants)
    return json.dumps(
        {"v": "CMDR-StructSig-v1-r3-family", "orbit": reps},
        sort_keys=True, separators=(",", ":"),
    )


def family_signature(family_variants: list[dict]) -> str:
    return hashlib.sha256(family_canonical(family_variants).encode("utf-8")).hexdigest()
