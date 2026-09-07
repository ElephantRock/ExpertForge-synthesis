from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LABELS = ["ENTAILED", "CONTRADICTED", "UNKNOWN"]


def cjson(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_text_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def load(root: Path, name: str) -> list[dict]:
    with (root / f"{name}.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def ground(value: dict, ent: str | None = None) -> tuple[str, str, str]:
    term = value["term"]
    if term == "x":
        term = ent
    return value["sign"], value["pred"], str(term)


def independent_verify(example: dict) -> tuple[str, int | None, int | None, int | None]:
    entities = {f["term"] for f in example["facts"] if f["term"] != "x"} | {example["query"]["term"]}
    depth = {ground(f): 0 for f in example["facts"]}
    for _ in range(64):
        added = False
        for rule in example["rules"]:
            for ent in entities:
                premises = [ground(p, ent) for p in rule["premises"]]
                if all(item in depth for item in premises):
                    conclusion = ground(rule["conclusion"], ent)
                    candidate_depth = 1 + max(depth[item] for item in premises)
                    if conclusion not in depth or candidate_depth < depth[conclusion]:
                        depth[conclusion] = candidate_depth
                        added = True
        if not added:
            break
    query = ground(example["query"])
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


def template_signature(example: dict) -> tuple:
    return tuple(
        (len(rule["premises"]), tuple(p["sign"] for p in rule["premises"]), rule["conclusion"]["sign"], rule["template"])
        for rule in example["rules"]
    )


def entity_set(example: dict) -> tuple[str, ...]:
    return tuple(sorted({x["term"] for x in example["facts"] if x["term"] != "x"} | {example["query"]["term"]}))


def bigram_multiset(text: str) -> Counter:
    tokens = text.split()
    return Counter(zip(tokens, tokens[1:]))


def verify_all(splits: dict[str, list[dict]]) -> dict:
    report = {"schema_id": "E0-Q1-INDEPENDENT-VERIFIER-v0", "status": "PASS", "splits": {}, "errors": []}
    family_seen: dict[str, str] = {}
    sample_ids: set[str] = set()
    for name, rows in splits.items():
        cells: Counter = Counter()
        errors = []
        for example in rows:
            if example["sample_id"] in sample_ids:
                errors.append(["duplicate_sample_id", example["sample_id"]])
            sample_ids.add(example["sample_id"])
            if example["family_id"] in family_seen and family_seen[example["family_id"]] != name:
                errors.append(["family_cross_split", example["family_id"]])
            family_seen[example["family_id"]] = name
            label, proof_depth, q_depth, opposite_depth = independent_verify(example)
            if label != example["gold_label"]:
                errors.append(["label", example["sample_id"], label, example["gold_label"]])
            if label != "UNKNOWN" and proof_depth != example["reasoning_depth_stratum"]:
                errors.append(["depth", example["sample_id"], proof_depth, example["reasoning_depth_stratum"]])
            if label == "UNKNOWN" and (q_depth is not None or opposite_depth is not None):
                errors.append(["unknown_derivable", example["sample_id"], q_depth, opposite_depth])
            cells[(example["gold_label"], example["reasoning_depth_stratum"])] += 1
        report["splits"][name] = {
            "examples": len(rows),
            "cell_counts": {f"{k[0]}|d{k[1]}": v for k, v in sorted(cells.items())},
            "error_count": len(errors),
        }
        report["errors"] += errors[:100]
    if report["errors"]:
        report["status"] = "FAIL"
    return report


def counterfactual_audit(splits: dict[str, list[dict]]) -> dict:
    report = {
        "schema_id": "E0-Q1-COUNTERFACTUAL-AUDIT-v0",
        "status": "PASS",
        "families": 0,
        "errors": [],
        "exact_unigram_invariance_families": 0,
        "exact_bigram_invariance_families": 0,
    }
    for rows in splits.values():
        by_family: defaultdict[str, list[dict]] = defaultdict(list)
        for example in rows:
            by_family[example["family_id"]].append(example)
        for family_id, variants in by_family.items():
            report["families"] += 1
            if sorted(v["gold_label"] for v in variants) != sorted(LABELS):
                report["errors"].append([family_id, "labels"])
                continue
            for attr in ["split", "surface", "reasoning_depth_stratum"]:
                if len({v[attr] for v in variants}) != 1:
                    report["errors"].append([family_id, attr])
            if len({v["query"]["sign"] + "|" + v["query"]["pred"] + "|" + v["query"]["term"] for v in variants}) != 1:
                report["errors"].append([family_id, "query"])
            if len({len(v["facts"]) for v in variants}) != 1 or len({len(v["rules"]) for v in variants}) != 1:
                report["errors"].append([family_id, "counts"])
            if len({v["student_token_count_with_DECIDE"] for v in variants}) != 1 or len({len(v["rendered"]) for v in variants}) != 1:
                report["errors"].append([family_id, "length"])
            if len({entity_set(v) for v in variants}) != 1:
                report["errors"].append([family_id, "entity_set"])
            if len({template_signature(v) for v in variants}) != 1:
                report["errors"].append([family_id, "template_signature"])
            unigrams = [Counter(v["rendered"].split()) for v in variants]
            if unigrams[0] == unigrams[1] == unigrams[2]:
                report["exact_unigram_invariance_families"] += 1
            else:
                report["errors"].append([family_id, "unigram_multiset"])
            bigrams = [bigram_multiset(v["rendered"]) for v in variants]
            if bigrams[0] == bigrams[1] == bigrams[2]:
                report["exact_bigram_invariance_families"] += 1
    if report["errors"]:
        report["status"] = "FAIL"
    report["error_count"] = len(report["errors"])
    report["errors"] = report["errors"][:100]
    return report


def nuisance_features(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for example in rows:
        text = example["rendered"]
        facts = example["facts"]
        rules = example["rules"]
        feat = [
            len(facts),
            len(rules),
            len(text.split()),
            len(text),
            sum(f["sign"] == "+" for f in facts),
            sum(f["sign"] == "-" for f in facts),
            sum(r["conclusion"]["sign"] == "+" for r in rules),
            sum(r["conclusion"]["sign"] == "-" for r in rules),
            sum(len(r["premises"]) == 1 for r in rules),
            sum(len(r["premises"]) == 2 for r in rules),
            example["reasoning_depth_stratum"],
            1 if example["surface"] == "STRUCT" else 0,
        ]
        for pos in range(12):
            rule = rules[pos]
            feat += [len(rule["premises"]), 1 if rule["conclusion"]["sign"] == "+" else 0]
        features.append(feat)
        labels.append(example["gold_label"])
    return np.asarray(features, dtype=np.float64), np.asarray(labels)


def macro_cell_accuracy(y_true: np.ndarray, y_pred: np.ndarray, rows: list[dict]) -> float:
    values = []
    for depth in range(1, 5):
        for label in LABELS:
            idx = [i for i, row in enumerate(rows) if row["reasoning_depth_stratum"] == depth and row["gold_label"] == label]
            values.append(float(np.mean(y_pred[idx] == y_true[idx])))
    return float(np.mean(values))


def shortcut_audits(train: list[dict], eval_id: list[dict], eval_struct: list[dict]) -> dict:
    report = {"schema_id": "E0-Q1-SHORTCUT-AUDIT-v0", "gate_max": 0.38, "models": {}, "status": "PASS"}
    x_train, y_train = nuisance_features(train)
    nuisance = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=0))
    nuisance.fit(x_train, y_train)
    for name, rows in [("eval_ID", eval_id), ("eval_STRUCT", eval_struct)]:
        x, y = nuisance_features(rows)
        pred = nuisance.predict(x)
        score = macro_cell_accuracy(y, pred, rows)
        report["models"].setdefault("NUI-LOGIT-v0", {})[name] = {
            "CMDR_SMA_like": score,
            "accuracy": float(accuracy_score(y, pred)),
        }
        if score > 0.38:
            report["status"] = "FAIL"

    vectorizer = CountVectorizer(
        analyzer="word",
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        binary=False,
    )
    x_lex = vectorizer.fit_transform([row["rendered"] for row in train])
    lexical = LogisticRegression(max_iter=100, random_state=0, solver="lbfgs", tol=1e-8)
    lexical.fit(x_lex, y_train)
    report["models"]["LEX-LINEAR-v0"] = {"vocab_features": int(len(vectorizer.vocabulary_)), "solver": "multinomial_logistic_lbfgs"}
    for name, rows in [("eval_ID", eval_id), ("eval_STRUCT", eval_struct)]:
        x = vectorizer.transform([row["rendered"] for row in rows])
        y = np.asarray([row["gold_label"] for row in rows])
        pred = lexical.predict(x)
        score = macro_cell_accuracy(y, pred, rows)
        report["models"]["LEX-LINEAR-v0"][name] = {
            "CMDR_SMA_like": score,
            "accuracy": float(accuracy_score(y, pred)),
        }
        if score > 0.38:
            report["status"] = "FAIL"
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("local_data/e0_qualification_bootstrap/Q1"))
    args = parser.parse_args()
    root = args.root.resolve()
    splits = {name: load(root, name) for name in ["train_ID", "eval_ID", "eval_STRUCT"]}
    verifier = verify_all(splits)
    counterfactual = counterfactual_audit(splits)
    shortcut = shortcut_audits(splits["train_ID"], splits["eval_ID"], splits["eval_STRUCT"])
    for filename, obj in [
        ("proof_depth_audit.json", verifier),
        ("counterfactual_audit.json", counterfactual),
        ("shortcut_audit.json", shortcut),
    ]:
        write_text_lf(root / filename, cjson(obj) + "\n")
    print(json.dumps({"verifier": verifier, "counterfactual": counterfactual, "shortcut": shortcut}, indent=2))


if __name__ == "__main__":
    main()
