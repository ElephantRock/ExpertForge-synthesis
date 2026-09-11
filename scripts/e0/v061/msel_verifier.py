"""Independent proof verifier for CMDR-MSEL families (v0.6.1 bootstrap).

Port of the v0.5 independent verifier (scripts/e0/q1_verify_and_audit.py,
independent_verify): a forward-chaining fixed-point over grounded literals
with minimal proof-depth tracking, implemented separately from the
generator's own verify_semantics so that the two must agree.
"""

from __future__ import annotations

from collections import Counter


def _ground(value: dict, ent: str | None):
    term = value["term"]
    if term == "x":
        term = ent
    return value["sign"], value["pred"], str(term)


def independent_verify(example: dict) -> tuple[str, int | None, int | None, int | None]:
    entities = {f["term"] for f in example["facts"] if f["term"] != "x"} | {example["query"]["term"]}
    depth = {_ground(f, None): 0 for f in example["facts"]}
    for _ in range(64):
        added = False
        for rule in example["rules"]:
            for ent in entities:
                premises = [_ground(p, ent) for p in rule["premises"]]
                if all(item in depth for item in premises):
                    conclusion = _ground(rule["conclusion"], ent)
                    candidate_depth = 1 + max(depth[item] for item in premises)
                    if conclusion not in depth or candidate_depth < depth[conclusion]:
                        depth[conclusion] = candidate_depth
                        added = True
        if not added:
            break
    query = _ground(example["query"], None)
    opposite = ("+" if query[0] == "-" else "-", query[1], query[2])
    q_depth = depth.get(query)
    opposite_depth = depth.get(opposite)
    if q_depth is not None and opposite_depth is not None:
        return "INVALID_BOTH", None, q_depth, opposite_depth
    if q_depth is not None:
        return "ENTAILED", q_depth, q_depth, opposite_depth
    if opposite_depth is not None:
        return "CONTRADICTED", opposite_depth, q_depth, opposite_depth
    return "UNKNOWN", None, q_depth, opposite_depth


def verify_family(family: dict) -> list:
    """Full independent check of one complete family. Returns error list."""
    errors = []
    variants = family["variants"]
    if sorted(v["gold_label"] for v in variants) != sorted(["ENTAILED", "CONTRADICTED", "UNKNOWN"]):
        errors.append([family["family_id"], "labels"])
    for attr in ("split", "surface", "reasoning_depth_stratum", "facts", "query"):
        if len({str(v[attr]) for v in variants}) != 1:
            errors.append([family["family_id"], f"shared_{attr}"])
    if len({len(v["rules"]) for v in variants}) != 1:
        errors.append([family["family_id"], "rule_count"])
    if len({len(v["rendered"]) for v in variants}) != 1:
        errors.append([family["family_id"], "rendered_length"])
    for v in variants:
        label, proof_depth, q_depth, opposite_depth = independent_verify(v)
        if label != v["gold_label"]:
            errors.append([v["sample_id"], "label", label, v["gold_label"]])
        if label != "UNKNOWN" and proof_depth != v["reasoning_depth_stratum"]:
            errors.append([v["sample_id"], "depth", proof_depth, v["reasoning_depth_stratum"]])
        if label == "UNKNOWN" and (q_depth is not None or opposite_depth is not None):
            errors.append([v["sample_id"], "unknown_derivable", q_depth, opposite_depth])
    return errors


def counterfactual_invariance(family: dict) -> list:
    """Unigram/bigram multiset invariance across the three variants."""
    errors = []

    def uni(ex):
        return Counter(ex["rendered"].split())

    def bi(ex):
        toks = ex["rendered"].split()
        return Counter(zip(toks, toks[1:]))

    variants = family["variants"]
    if not (uni(variants[0]) == uni(variants[1]) == uni(variants[2])):
        errors.append([family["family_id"], "unigram_multiset"])
    if not (bi(variants[0]) == bi(variants[1]) == bi(variants[2])):
        errors.append([family["family_id"], "bigram_multiset"])
    return errors
