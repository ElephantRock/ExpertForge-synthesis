from __future__ import annotations

import argparse
import hashlib
import json
import random
import unicodedata
from pathlib import Path

GENERATOR_ID = "CMDR-Generator-v1-QUAL-bootstrap"
QUAL_ID = "CMDR-QUAL-v1"
NAMESPACE = "ExpertForge-E0-CMDR-QUAL-v1"
PRED_COUNT = 1024
ENTITY_COUNT = 512
TOTAL_FACTS = 12
TOTAL_RULES = 12
LABELS = ["ENTAILED", "CONTRADICTED", "UNKNOWN"]


def hbytes(s: str) -> bytes:
    return hashlib.sha256(s.encode("utf-8")).digest()


def seed_for(*parts: object) -> int:
    return int.from_bytes(hbytes("|".join(map(str, parts)))[:8], "big")


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


class FixedLex:
    def __init__(self) -> None:
        base = [
            "<PAD>", "<DECIDE>", "Facts", ":", "Rules", "Query", "Choose",
            "exactly", "one", "symbol", "A", "=", "ENTAILED", "B",
            "CONTRADICTED", "C", "UNKNOWN", "Answer", "+", "-", "x", "->", "&",
        ]
        preds = [tok_pred(i) for i in range(PRED_COUNT)]
        ents = [tok_ent(i) for i in range(ENTITY_COUNT)]
        tokens = base + preds + ents
        remaining = 4096 - len(tokens)
        if remaining < 0:
            raise RuntimeError("vocab overflow")
        tokens += [f"<RESERVED_{i:04d}>" for i in range(remaining)]
        if len(tokens) != 4096 or len(set(tokens)) != 4096:
            raise AssertionError("invalid fixed vocabulary")
        self.tokens = tokens
        self.vocab = {t: i for i, t in enumerate(tokens)}

    def encode(self, text: str, append_decide: bool = False) -> list[int]:
        tokens = text.split()
        if append_decide:
            tokens.append("<DECIDE>")
        bad = [t for t in tokens if t not in self.vocab]
        if bad:
            raise ValueError(f"unknown tokens {bad[:5]}")
        return [self.vocab[t] for t in tokens]


LEX = FixedLex()


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
    return {
        "label": label,
        "proof_depth": proof,
        "q_depth": q_depth,
        "opposite_depth": opposite_depth,
    }


def choose_unique(rng: random.Random, k: int, exclude: set[str] | None = None, pool: int = PRED_COUNT) -> list[str]:
    excluded = set(exclude or [])
    out: list[str] = []
    while len(out) < k:
        pred = tok_pred(rng.randrange(pool))
        if pred not in excluded:
            excluded.add(pred)
            out.append(pred)
    return out


def family_base(split: str, surface: str, depth: int, family_index: int) -> dict:
    rng = random.Random(seed_for(NAMESPACE, split, surface, depth, family_index))
    ent = tok_ent(rng.randrange(ENTITY_COUNT))
    preds = choose_unique(rng, 40)
    qpred, neutral = preds[0], preds[1]
    chain = preds[2:2 + max(depth, 1)]
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


def make_variant(base: dict, label: str, split: str, surface: str, depth: int, family_index: int) -> dict:
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
    if seed_for(NAMESPACE, "pivot-rem", split, surface, depth, family_index) % 2:
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
    family_id = sha256_bytes(f"{NAMESPACE}|{split}|{surface}|d{depth}|family|{family_index}".encode())[:24]
    sample_id = sha256_bytes(f"{family_id}|{label}".encode())[:24]
    example = {
        "schema_id": "CMDR-EXAMPLE-v1",
        "sample_id": sample_id,
        "family_id": family_id,
        "namespace": NAMESPACE,
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
    example["student_token_count_with_DECIDE"] = len(LEX.encode(text, append_decide=True))
    return example


def build_split(split: str, surface: str, families_per_depth: int) -> list[dict]:
    out = []
    for depth in range(1, 5):
        for family_index in range(families_per_depth):
            base = family_base(split, surface, depth, family_index)
            variants = [make_variant(base, label, split, surface, depth, family_index) for label in LABELS]
            if len({v["student_token_count_with_DECIDE"] for v in variants}) != 1 or len({len(v["rendered"]) for v in variants}) != 1:
                raise AssertionError("counterfactual length mismatch")
            for variant in variants:
                verified = verify_semantics(variant)
                if verified["label"] != variant["gold_label"]:
                    raise AssertionError(("label mismatch", variant["sample_id"], verified, variant["gold_label"]))
                if variant["gold_label"] != "UNKNOWN" and verified["proof_depth"] != depth:
                    raise AssertionError(("depth mismatch", variant["sample_id"], verified, depth))
                if variant["gold_label"] == "UNKNOWN" and (verified["q_depth"] is not None or verified["opposite_depth"] is not None):
                    raise AssertionError(("unknown derivable", variant["sample_id"], verified))
                variant["verified_shortest_proof_depth"] = verified["proof_depth"]
            out.extend(variants)
    return out


def write_text_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(cjson(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("local_data/e0_qualification_bootstrap/Q1"))
    args = parser.parse_args()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = [("train_ID", "ID", 2000), ("eval_ID", "ID", 500), ("eval_STRUCT", "STRUCT", 500)]
    manifest = {
        "schema_id": "CMDR-QUAL-MANIFEST-v1",
        "generator_id": GENERATOR_ID,
        "qualification_id": QUAL_ID,
        "namespace": NAMESPACE,
        "splits": {},
        "vocab_size": 4096,
        "max_sequence_length": 384,
    }
    all_ids: set[str] = set()
    family_split: dict[str, str] = {}
    max_tokens = 0

    for split, surface, families_per_depth in splits:
        rows = build_split(split, surface, families_per_depth)
        if len(rows) != families_per_depth * 4 * 3:
            raise AssertionError("split size mismatch")
        path = out_dir / f"{split}.jsonl"
        write_jsonl(path, rows)
        counts: dict[str, int] = {}
        for row in rows:
            key = f"{row['gold_label']}|d{row['reasoning_depth_stratum']}"
            counts[key] = counts.get(key, 0) + 1
            if row["sample_id"] in all_ids:
                raise AssertionError("duplicate sample id")
            all_ids.add(row["sample_id"])
            if row["family_id"] in family_split and family_split[row["family_id"]] != split:
                raise AssertionError("family split")
            family_split[row["family_id"]] = split
            max_tokens = max(max_tokens, row["student_token_count_with_DECIDE"])
        manifest["splits"][split] = {
            "surface": surface,
            "examples": len(rows),
            "families": len(rows) // 3,
            "sha256": sha256_bytes(path.read_bytes()),
            "cell_counts": counts,
        }

    if len(all_ids) != 36000 or max_tokens > 384:
        raise AssertionError("qualification corpus invariant failed")

    vocab = {
        "schema_id": "CMDR-Lex-v1",
        "type": "deterministic_fixed_lexical",
        "learned": False,
        "vocab_size": 4096,
        "unknown_token_allowed": False,
        "max_sequence_length": 384,
        "lexer": "exact_whitespace_split_on_canonical_rendering",
        "append_student_token": "<DECIDE>",
        "tokens": LEX.tokens,
    }
    vocab_path = out_dir / "CMDR-Lex-v1.json"
    write_text_lf(vocab_path, cjson(vocab) + "\n")
    manifest["CMDR_Lex_v1_sha256"] = sha256_bytes(vocab_path.read_bytes())
    manifest["maximum_student_token_count_with_DECIDE"] = max_tokens
    manifest_path = out_dir / "CMDR-QUAL-v1.manifest.json"
    write_text_lf(manifest_path, cjson(manifest) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
