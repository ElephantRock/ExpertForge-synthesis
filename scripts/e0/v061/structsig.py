"""CMDR-StructSig-v1 — canonical structural signature (contract v0.6.1 SS4.3).

Procedure (per contract):
  1. retain fact/rule/query graph topology, polarity, arity, direction,
     reasoning-depth stratum, and distractor connectivity;
  2. replace entity and predicate identifiers with deterministic
     first-occurrence canonical symbols;
  3. remove lexical surface choices, slot-order nuisance, whitespace, and
     generator seed identity;
  4. sort semantically unordered fact/rule sets under the canonical symbol map;
  5. serialize canonical UTF-8 JSON and hash with SHA-256.

Two implementation decisions, both required for renaming stability:

(A) Step 2's first-occurrence symbols are assigned over a COLOR-CANONICAL
    traversal: predicates/entities are colored by iterative structural
    refinement (name-free neighborhood fingerprints over the fact/rule
    hypergraph; the query literal distinguishes its predicate and entity),
    facts/rules are ordered by refined color, and first occurrence in that
    order assigns P#/E#. Equal-color (structurally symmetric) nodes are
    serialized through their color labels, so isomorphic structures
    serialize identically regardless of tie order. A naive lexicographic
    symbol order would make signatures depend on which concrete names the
    generator drew — false distinctions from naming alone.

(B) The signature is FAMILY-level and label-invariant (SS4.3 hashes one
    signature per family; eval_STRUCT isolation is defined on family
    signatures). The three counterfactual variants share the same skeleton
    and differ only in the pivot antecedent<->conclusion binding, which is
    the family's semantic content. The family signature is therefore the
    lexicographic minimum of the three full variant signatures: same
    skeleton (any binding assignment) yields the same three-element orbit
    and hence the same minimum; renaming stability is inherited from the
    per-variant refinement; distinct skeletons yield distinct orbits.
"""

from __future__ import annotations

import hashlib
import json

_REFINEMENT_ROUNDS = 8


def _h(*parts) -> str:
    return hashlib.sha256(
        json.dumps(parts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


def _variant_signature(example: dict) -> str:
    """Full structural signature of ONE variant (pivot binding included)."""
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
    surface = example["surface"]

    preds = {p for f in facts for p in (f[1],)} | {
        p for prem, concl in rules for l in prem + [concl] for p in (l[1],)
    } | {query[1]}
    ents = {f[2] for f in facts if f[2] != "x"} | {
        l[2] for prem, concl in rules for l in prem + [concl] if l[2] != "x"
    } | ({query[2]} if query[2] != "x" else set())

    pred_color = {p: "P" for p in preds}
    ent_color = {e: "E" for e in ents}
    pred_color[query[1]] = "PQ"
    if query[2] != "x":
        ent_color[query[2]] = "EQ"

    def lit_color(l):
        return _h("L", l[0], pred_color[l[1]], ent_color.get(l[2], "X"), l[2] == "x")

    for _ in range(_REFINEMENT_ROUNDS):
        fact_cols = sorted(_h("F", lit_color(f)) for f in facts)
        rule_cols = []
        for prem, concl in rules:
            rule_cols.append(_h("R", sorted(lit_color(p) for p in prem), lit_color(concl)))
        query_col = _h("Q", lit_color(query))

        new_pred = {}
        for p in preds:
            incident = []
            for f in facts:
                if f[1] == p:
                    incident.append("fact:" + _h("F", lit_color(f)))
            for rc, (prem, concl) in zip(rule_cols, rules):
                if any(pp[1] == p for pp in prem):
                    incident.append("prem:" + rc)
                if concl[1] == p:
                    incident.append("concl:" + rc)
            if query[1] == p:
                incident.append("query")
            new_pred[p] = _h("P", pred_color[p], sorted(set(incident)))
        new_ent = {}
        for e in ents:
            incident = []
            for f in facts:
                if f[2] == e:
                    incident.append("fact:" + _h("F", lit_color(f)))
            for rc, (prem, concl) in zip(rule_cols, rules):
                if any(pp[2] == e for pp in prem):
                    incident.append("prem:" + rc)
                if concl[2] == e:
                    incident.append("concl:" + rc)
            if query[2] == e:
                incident.append("query")
            new_ent[e] = _h("E", ent_color[e], sorted(set(incident)))
        pred_color, ent_color = new_pred, new_ent

    def lit_canon(l):
        return (l[0], pred_color[l[1]], ent_color.get(l[2], "X"), l[2] == "x")

    fact_ser = sorted(_h("F", lit_canon(f)) for f in facts)
    rule_ser = sorted(
        _h("R", sorted(lit_canon(p) for p in prem), lit_canon(concl)) for prem, concl in rules
    )
    canonical = {
        "v": "CMDR-StructSig-v1",
        "depth": depth,
        "surface": surface,
        "facts": fact_ser,
        "rules": rule_ser,
        "query": _h("Q", lit_canon(query)),
    }
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def structsig_family(family_variants: list[dict]) -> str:
    """Family-level signature: lexicographic minimum of the three variant
    signatures (label-invariant, renaming-stable)."""
    if len(family_variants) != 3:
        raise AssertionError("structsig_family expects exactly three variants")
    return min(_variant_signature(v) for v in family_variants)


def variant_signatures(family_variants: list[dict]) -> list[str]:
    """All three variant signatures (diagnostic helper)."""
    return sorted(_variant_signature(v) for v in family_variants)
