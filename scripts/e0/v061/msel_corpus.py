"""CMDR-MSEL corpus generator (v0.6.1 mechanism-selection bootstrap).

Namespace-parameterized port of the v0.5 CMDR-QUAL-v1 family construction
(scripts/e0/q1_cmdr_bootstrap.py). The construction logic — family base,
counterfactual variants, rendering, semantic verification — is identical to
v0.5; only the generator namespace and corpus scale differ. This module is
authorized under V06_MECHANISM_BOOTSTRAP_AUTHORIZED (generator/verifier
scope). No R/P scoring lives here.

Namespaces:
  authoritative: ExpertForge-E0-v061-msel        (burned after selection closure)
  pilot (burned): ExpertForge-E0-v061-msel-preflight (never enters R1/R4/R16,
                  qualification, or later E0 data)
"""

from __future__ import annotations

import hashlib
import json
import random
import unicodedata

PRED_COUNT = 1024
ENTITY_COUNT = 512
TOTAL_FACTS = 12
TOTAL_RULES = 12
LABELS = ["ENTAILED", "CONTRADICTED", "UNKNOWN"]

MSEL_NAMESPACE_ROOT = "ExpertForge-E0-v061-msel"
MSEL_PILOT_NAMESPACE_ROOT = "ExpertForge-E0-v061-msel-preflight"


def seed_for(*parts: object) -> int:
    return int.from_bytes(
        hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()[:8], "big"
    )


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def cjson(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def tok_pred(i: int) -> str:
    return f"P{i % PRED_COUNT:04d}"


def tok_ent(i: int) -> str:
    return f"E{i % ENTITY_COUNT:04d}"


def lit(sign: str, pred: str, term: str = "x") -> dict:
    return {"sign": sign, "pred": pred, "term": term}


def render_lit(value: dict) -> str:
    return f"{value['sign']} {value['pred']} {value['term']}"


def render_rule(rule: dict) -> str:
    lhs = " & ".join(render_lit(x) for x in rule["premises"])
    return f"{lhs} -> {render_lit(rule['conclusion'])}"


def render(example: dict) -> str:
    lines = ["Facts :"]
    lines += [render_lit(x) for x in example["facts"]]
    lines += ["Rules :"]
    lines += [render_rule(x) for x in example["rules"]]
    lines += [
        "Query :",
        render_lit(example["query"]),
        "Choose exactly one symbol :",
        "A = ENTAILED",
        "B = CONTRADICTED",
        "C = UNKNOWN",
        "Answer :",
    ]
    return unicodedata.normalize("NFC", "\n".join(lines) + "\n")


# --------------------------------------------------------------------------- 
# Family construction (identical logic to q1_cmdr_bootstrap.family_base /
# make_variant; namespace and split names are parameters).
# ---------------------------------------------------------------------------


def ground_literal(value: dict, env_term: str | None = None) -> tuple[str, str, str]:
    term = value["term"] if env_term is None or value["term"] != "x" else env_term
    return value["sign"], value["pred"], term


def verify_semantics(example: dict) -> dict:
    depth: dict[tuple[str, str, str], int] = {}
    for fact in example["facts"]:
        depth[ground_literal(fact)] = 0
    changed = True
    while changed:
        changed = False
        for rule in example["rules"]:
            uses_x = any(p["term"] == "x" for p in rule["premises"]) or rule["conclusion"]["term"] == "x"
            candidates = {k[2] for k in depth} if uses_x else {None}
            for ent in candidates:
                premises = [ground_literal(p, ent) for p in rule["premises"]]
                if all(pk in depth for pk in premises):
                    candidate_depth = 1 + max(depth[pk] for pk in premises)
                    conclusion = ground_literal(rule["conclusion"], ent)
                    if conclusion not in depth or candidate_depth < depth[conclusion]:
                        depth[conclusion] = candidate_depth
                        changed = True
    query = ground_literal(example["query"])
    opposite = ("+" if query[0] == "-" else "-", query[1], query[2])
    q_depth = depth.get(query)
    opposite_depth = depth.get(opposite)
    if q_depth is not None and opposite_depth is not None:
        label, proof = "INVALID_BOTH", None
    elif q_depth is not None:
        label, proof = "ENTAILED", q_depth
    elif opposite_depth is not None:
        label, proof = "CONTRADICTED", opposite_depth
    else:
        label, proof = "UNKNOWN", None
    return {"label": label, "proof_depth": proof, "q_depth": q_depth, "opposite_depth": opposite_depth}


def choose_unique(rng: random.Random, k: int, exclude: set[str] | None = None, pool: int = PRED_COUNT) -> list[str]:
    excluded = set(exclude or [])
    out: list[str] = []
    while len(out) < k:
        pred = tok_pred(rng.randrange(pool))
        if pred not in excluded:
            excluded.add(pred)
            out.append(pred)
    return out


def family_base(namespace: str, split: str, surface: str, depth: int, family_index: int) -> dict:
    rng = random.Random(seed_for(namespace, split, surface, depth, family_index))
    ent = tok_ent(rng.randrange(ENTITY_COUNT))
    preds = choose_unique(rng, 40)
    qpred, neutral = preds[0], preds[1]
    chain = preds[2 : 2 + max(depth, 1)]
    d1, d2 = preds[10], preds[11]
    helpers = preds[12:20]
    distract = preds[20:]

    facts = [lit("+", chain[0], ent)]
    rules = []
    for step in range(depth - 1):
        src, dst = chain[step], chain[step + 1]
        if surface == "ID":
            rules.append({"premises": [lit("+", src, "x")], "conclusion": lit("+", dst, "x"), "template": "ID_CHAIN_1"})
        else:
            hp = helpers[step]
            facts.append(lit("+", hp, ent))
            rules.append({"premises": [lit("+", src, "x"), lit("+", hp, "x")], "conclusion": lit("+", dst, "x"), "template": "STRUCT_CHAIN_2"})
    reachable = chain[depth - 1] if depth > 1 else chain[0]

    conclusions = [lit("+", qpred, "x"), lit("-", qpred, "x"), lit("+", neutral, "x")]
    rng.shuffle(conclusions)
    pivot_helpers: list[dict] = []
    if surface == "STRUCT":
        hp = helpers[7]
        facts.append(lit("+", hp, ent))
        pivot_helpers = [lit("+", hp, "x")]

    protected = {qpred, neutral, *chain, d1, d2, *helpers}
    distractor_preds = [p for p in distract if p not in protected]
    j = 0
    while len(facts) < TOTAL_FACTS:
        pred = distractor_preds[j % len(distractor_preds)]
        j += 1
        sign = "+" if rng.randrange(2) == 0 else "-"
        entity = tok_ent((int(ent[1:]) + 1 + j) % ENTITY_COUNT)
        facts.append(lit(sign, pred, entity))

    distractor_rules = []
    while len(distractor_rules) < TOTAL_RULES - (depth - 1) - 3:
        a, b = choose_unique(rng, 2, exclude=protected | set(distractor_preds[:2]))
        sign1 = "+" if rng.randrange(2) == 0 else "-"
        sign2 = "+" if rng.randrange(2) == 0 else "-"
        if surface == "ID":
            rule = {"premises": [lit(sign1, a, "x")], "conclusion": lit(sign2, b, "x"), "template": "ID_DISTRACTOR_1"}
        else:
            c = choose_unique(rng, 1, exclude=protected | {a, b})[0]
            rule = {"premises": [lit(sign1, a, "x"), lit("+", c, "x")], "conclusion": lit(sign2, b, "x"), "template": "STRUCT_DISTRACTOR_2"}
        distractor_rules.append(rule)

    order = list(range(TOTAL_RULES))
    rng.shuffle(order)
    return {
        "ent": ent,
        "qpred": qpred,
        "neutral": neutral,
        "reachable": reachable,
        "d1": d1,
        "d2": d2,
        "pivot_helpers": pivot_helpers,
        "conclusions": conclusions,
        "facts": facts,
        "chain_rules": rules,
        "distractor_rules": distractor_rules,
        "order": order,
        "query": lit("+", qpred, ent),
    }


def make_variant(base: dict, label: str, namespace: str, split: str, surface: str, depth: int, family_index: int) -> dict:
    desired = {
        "ENTAILED": ("+", base["qpred"]),
        "CONTRADICTED": ("-", base["qpred"]),
        "UNKNOWN": ("+", base["neutral"]),
    }[label]
    desired_slot = next(i for i, c in enumerate(base["conclusions"]) if (c["sign"], c["pred"]) == desired)
    antecedents = [base["d1"], base["d2"]]
    slots: list[str | None] = [None, None, None]
    slots[desired_slot] = base["reachable"]
    remaining = [i for i in range(3) if i != desired_slot]
    if seed_for(namespace, "pivot-rem", split, surface, depth, family_index) % 2:
        antecedents.reverse()
    slots[remaining[0]], slots[remaining[1]] = antecedents

    pivots = []
    for i, conclusion in enumerate(base["conclusions"]):
        premises = [lit("+", str(slots[i]), "x")] + list(base["pivot_helpers"])
        pivots.append({
            "premises": premises,
            "conclusion": conclusion,
            "template": "ID_PIVOT_1" if surface == "ID" else "STRUCT_PIVOT_2",
        })

    raw_rules = base["chain_rules"] + pivots + base["distractor_rules"]
    if len(raw_rules) != TOTAL_RULES:
        raise AssertionError("wrong rule count")
    rules = [raw_rules[i] for i in base["order"]]
    family_id = sha256_bytes(f"{namespace}|{split}|{surface}|d{depth}|family|{family_index}".encode())[:24]
    sample_id = sha256_bytes(f"{family_id}|{label}".encode())[:24]
    example = {
        "schema_id": "CMDR-MSEL-EXAMPLE-v0",
        "sample_id": sample_id,
        "family_id": family_id,
        "namespace": namespace,
        "split": split,
        "surface": surface,
        "reasoning_depth_stratum": depth,
        "gold_label": label,
        "facts": base["facts"],
        "rules": rules,
        "query": base["query"],
    }
    text = render(example)
    example["rendered"] = text
    example["render_sha256"] = sha256_bytes(text.encode("utf-8"))
    return example


def build_families(namespace: str, split: str, surface: str, depth: int,
                   family_indices: range) -> list[dict]:
    """Generate complete counterfactual families with inline verification
    (labels must match gold; ENTAILED/CONTRADICTED minimal proof depth must
    equal the stratum; UNKNOWN must be underivable)."""
    out = []
    for fi in family_indices:
        base = family_base(namespace, split, surface, depth, fi)
        variants = [make_variant(base, label, namespace, split, surface, depth, fi) for label in LABELS]
        if len({len(v["rendered"]) for v in variants}) != 1:
            raise AssertionError(("counterfactual length mismatch", namespace, split, surface, depth, fi))
        for variant in variants:
            verified = verify_semantics(variant)
            if verified["label"] != variant["gold_label"]:
                raise AssertionError(("label mismatch", variant["sample_id"], verified, variant["gold_label"]))
            if variant["gold_label"] != "UNKNOWN" and verified["proof_depth"] != depth:
                raise AssertionError(("depth mismatch", variant["sample_id"], verified, depth))
            if variant["gold_label"] == "UNKNOWN" and (verified["q_depth"] is not None or verified["opposite_depth"] is not None):
                raise AssertionError(("unknown derivable", variant["sample_id"], verified))
            variant["verified_shortest_proof_depth"] = verified["proof_depth"]
        out.append({
            "family_id": variants[0]["family_id"],
            "namespace": namespace,
            "split": split,
            "surface": surface,
            "depth": depth,
            "family_index": fi,
            "variants": variants,
        })
    return out


def apply_renaming(example: dict, pred_map: dict[str, str], ent_map: dict[str, str]) -> dict:
    """Return a copy of an example with all predicate/entity identifiers
    renamed consistently (StructSig unit-test helper; never used for corpus
    generation)."""
    import copy

    ex = copy.deepcopy({k: v for k, v in example.items() if k not in ("rendered", "render_sha256")})

    def rl(l):
        return {"sign": l["sign"], "pred": pred_map[l["pred"]], "term": ent_map.get(l["term"], l["term"])}

    ex["facts"] = [rl(f) for f in ex["facts"]]
    ex["rules"] = [
        {"premises": [rl(p) for p in r["premises"]], "conclusion": rl(r["conclusion"])} for r in ex["rules"]
    ]
    ex["query"] = rl(ex["query"])
    return ex
