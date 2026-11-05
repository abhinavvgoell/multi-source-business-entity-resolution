#!/usr/bin/env python3
"""Leakage-resistant entity-resolution evaluation on Fodors-Zagats.

The public DeepMatcher pair splits reuse records across train/valid/test. This
evaluation instead splits entity classes first, then builds candidates and
labels within each split. The class column is used only as ground truth for
split construction and pair labels; it is never passed to the matcher.
"""
from __future__ import annotations

import argparse
import csv
import difflib
import json
import random
import re
import sys
import time
from urllib.request import urlretrieve
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "student_resource" / "code" / "business_entity_resolution" / "src"
sys.path.insert(0, str(SRC))
import business_entity_resolution as ber  # noqa: E402


def read_table(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def prepare(row: dict[str, str], prefix: str) -> dict:
    # Fodors-Zagats has no country field. City is included with address so that
    # the address similarities reflect the business location.
    item = ber.prep_one({
        "entity_id": f"{prefix}-{row['id']}",
        "business_name": row["name"],
        "business_address": f"{row['addr']} {row['city']}",
        "country": "",
    })
    item["class"] = row["class"]
    return item


def engineered_features(a: dict, b: dict) -> list[float]:
    """Add generic string and address-component comparisons to the base vector."""
    base = ber.features(a, b)

    def edit_similarity(x: str, y: str) -> float:
        # SequenceMatcher is normalized to [0, 1] and needs no model download.
        return difflib.SequenceMatcher(None, x, y, autojunk=False).ratio() if x or y else 1.0

    def ratio(short: str, long: str) -> float:
        return min(len(short), len(long)) / max(1, len(short), len(long))

    nums_a = set(re.findall(r"\d+[a-z]?", a["addr"]))
    nums_b = set(re.findall(r"\d+[a-z]?", b["addr"]))
    house_a = re.match(r"^(\d+[a-z]?)\b", a["addr"])
    house_b = re.match(r"^(\d+[a-z]?)\b", b["addr"])
    postal_a = {x for x in nums_a if len(x) in (5, 6) and x.isdigit()}
    postal_b = {x for x in nums_b if len(x) in (5, 6) and x.isdigit()}
    number_jaccard = len(nums_a & nums_b) / (len(nums_a | nums_b) or 1)
    unit_re = re.compile(r"\b(?:suite|ste|unit|apt|apartment|floor|fl)\s*([a-z0-9-]+)")
    unit_a, unit_b = unit_re.search(a["addr"]), unit_re.search(b["addr"])
    direction_aliases = {"n": "north", "north": "north", "s": "south", "south": "south",
                         "e": "east", "east": "east", "w": "west", "west": "west"}
    dir_a = {direction_aliases[t] for t in a["at"] if t in direction_aliases}
    dir_b = {direction_aliases[t] for t in b["at"] if t in direction_aliases}
    return base + [
        edit_similarity(a["name"], b["name"]),
        edit_similarity(a["addr"], b["addr"]),
        ratio(a["name"], b["name"]),
        ratio(a["addr"], b["addr"]),
        number_jaccard,
        float(bool(house_a and house_b and house_a.group(1) == house_b.group(1))),
        float(bool(house_a and house_b and house_a.group(1) != house_b.group(1))),
        float(bool(postal_a & postal_b)),
        float(bool(postal_a and postal_b and not postal_a & postal_b)),
        float(bool(unit_a and unit_b and unit_a.group(1) != unit_b.group(1))),
        float(bool(dir_a and dir_b and not dir_a & dir_b)),
    ]


def split_classes(a_rows, b_rows, seed: int) -> dict[str, set[str]]:
    a_classes = {r["class"] for r in a_rows}
    b_classes = {r["class"] for r in b_rows}
    matched = sorted(a_classes & b_classes)
    a_only = sorted(a_classes - b_classes)
    b_only = sorted(b_classes - a_classes)
    rng = random.Random(seed)
    groups = {"train": set(), "validation": set(), "test": set()}
    for stratum in (matched, a_only, b_only):
        values = list(stratum)
        rng.shuffle(values)
        n_train = int(0.70 * len(values))
        n_valid = int(0.15 * len(values))
        groups["train"].update(values[:n_train])
        groups["validation"].update(values[n_train:n_train + n_valid])
        groups["test"].update(values[n_train + n_valid:])
    return groups


def records_by_split(rows, prefix, groups):
    result = {s: [] for s in groups}
    for row in rows:
        for split, classes in groups.items():
            if row["class"] in classes:
                result[split].append(prepare(row, prefix))
                break
    return result


def score_candidates(targets, refs, model, cap: int, feature_fn):
    ref_by_class = defaultdict(set)
    for ref in refs:
        ref_by_class[ref["class"]].add(ref["id"])
    index, posting_cap = ber.build_index(refs)
    truth = {t["id"]: ref_by_class.get(t["class"], set()) for t in targets}
    scored = {}
    for target in targets:
        candidate_indices = ber.candidate_for(target, refs, index, posting_cap, cap)
        candidate_ids = [refs[i]["id"] for i in candidate_indices]
        scores = ber.probs(model, [feature_fn(target, refs[i]) for i in candidate_indices]) if candidate_indices else []
        scored[target["id"]] = (candidate_ids, scores)
    return truth, scored


def evaluate_split(targets, refs, model, cap: int, threshold: float, seed: int,
                   feature_fn=ber.features, precomputed=None):
    if precomputed is None:
        truth, scored = score_candidates(targets, refs, model, cap, feature_fn)
    else:
        truth, scored = precomputed
    predictions = {t["id"]: set() for t in targets}
    all_pair_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    candidate_count = candidate_tp = 0
    per_entity_f05 = []
    for target in targets:
        candidate_id_list, scores = scored[target["id"]]
        candidate_ids = set(candidate_id_list)
        candidate_count += len(candidate_ids)
        candidate_tp += len(candidate_ids & truth[target["id"]])
        predictions[target["id"]] = {
            rid for rid, score in zip(candidate_id_list, scores) if score >= threshold
        }
        expected, predicted = truth[target["id"]], predictions[target["id"]]
        tp = len(expected & predicted)
        fp = len(predicted - expected)
        fn = len(expected - predicted)
        per_entity_f05.append(ber.f05(expected, predicted))
        # Count every possible A-B pair. Pairs omitted by blocking are
        # correctly treated as predicted non-matches in end-to-end accuracy.
        all_pair_counts["tp"] += tp
        all_pair_counts["fp"] += fp
        all_pair_counts["fn"] += fn
        all_pair_counts["tn"] += len(refs) - len(expected) - fp
    c = all_pair_counts
    total_pairs = len(targets) * len(refs)
    accuracy = (c["tp"] + c["tn"]) / total_pairs if total_pairs else 0.0
    precision = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
    recall = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
    specificity = c["tn"] / (c["tn"] + c["fp"]) if c["tn"] + c["fp"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f05 = 1.25 * precision * recall / (0.25 * precision + recall) if 0.25 * precision + recall else 0.0
    rng = np.random.default_rng(seed)
    boot = rng.choice(np.asarray(per_entity_f05), size=(5000, len(per_entity_f05)), replace=True).mean(1)
    return {
        "targets": len(targets), "references": len(refs), "all_possible_pairs": total_pairs,
        "known_matches": sum(map(len, truth.values())), "candidate_pairs": candidate_count,
        "candidate_recall": candidate_tp / sum(map(len, truth.values())) if sum(map(len, truth.values())) else 0.0,
        "threshold": threshold, "confusion_matrix": c,
        "pair_accuracy": accuracy, "balanced_accuracy": (recall + specificity) / 2,
        "specificity": specificity, "positive_pair_rate": (c["tp"] + c["fn"]) / total_pairs if total_pairs else 0.0,
        "precision": precision, "recall": recall,
        "f1": f1, "f0_5": f05,
        "macro_entity_f0_5": float(np.mean(per_entity_f05)) if per_entity_f05 else 0.0,
        "macro_entity_f0_5_bootstrap_95pct_ci": [float(v) for v in np.quantile(boot, [0.025, 0.975])],
        "candidate_precision": candidate_tp / candidate_count if candidate_count else 0.0,
    }


def train_model(targets, refs, cap: int, seed: int, feature_fn=ber.features):
    index, posting_cap = ber.build_index(refs)
    ref_by_class = defaultdict(list)
    for ref in refs:
        ref_by_class[ref["class"]].append(ref)
    X, y = [], []
    rng = random.Random(seed)
    for target in targets:
        candidates = [refs[i] for i in ber.candidate_for(target, refs, index, posting_cap, cap)]
        positives = ref_by_class.get(target["class"], [])
        positive_ids = {r["id"] for r in positives}
        for ref in positives:
            X.append(feature_fn(target, ref)); y.append(1)
        negatives = [r for r in candidates if r["id"] not in positive_ids]
        rng.shuffle(negatives)
        for ref in negatives[:max(5, min(30, 3 * max(1, len(positives))))]:
            X.append(feature_fn(target, ref)); y.append(0)
    if not y or not any(y) or all(y):
        raise ValueError("Training split must contain both matching and non-matching pairs")
    model = ber.fit(X, y, steps=500, lr=0.06)
    return model, {"training_pairs": len(y), "training_matches": sum(y), "training_nonmatches": len(y) - sum(y)}


def evaluate_seed(a_raw, b_raw, seed: int, candidate_cap: int):
    start = time.monotonic()
    groups = split_classes(a_raw, b_raw, seed)
    a = records_by_split(a_raw, "A", groups)
    b = records_by_split(b_raw, "B", groups)
    variants = {}
    for variant_name, feature_fn in (("baseline", ber.features), ("engineered", engineered_features)):
        model, training_summary = train_model(a["train"], b["train"], candidate_cap, seed, feature_fn)
        # Tune a precision-weighted threshold only on entity-disjoint validation data.
        validation_scored = score_candidates(a["validation"], b["validation"], model,
                                             candidate_cap, feature_fn)
        validation_scores = []
        for threshold in np.arange(0.10, 0.991, 0.01):
            m = evaluate_split(a["validation"], b["validation"], model, candidate_cap,
                               float(threshold), seed, feature_fn, validation_scored)
            validation_scores.append((m["macro_entity_f0_5"], float(threshold)))
        best_f05 = max(score for score, _ in validation_scores)
        chosen = min(threshold for score, threshold in validation_scores if score >= best_f05 - 1e-12)
        validation = evaluate_split(a["validation"], b["validation"], model, candidate_cap,
                                    chosen, seed, feature_fn, validation_scored)
        test = evaluate_split(a["test"], b["test"], model, candidate_cap,
                              chosen, seed + 1, feature_fn)
        variants[variant_name] = {
            "training": training_summary,
            "selected_threshold": chosen,
            "validation": validation,
            "test": test,
        }
    amazon_model_path = ROOT / "student_resource" / "model" / "entity_matcher.json"
    amazon_model = json.loads(amazon_model_path.read_text(encoding="utf-8"))
    zero_shot = evaluate_split(a["test"], b["test"], amazon_model,
                               candidate_cap, amazon_model["threshold"], seed + 1)
    return {
        "seed": seed,
        "candidate_cap_per_target": candidate_cap,
        "entity_classes_per_split": {k: len(v) for k, v in groups.items()},
        "rows_per_split": {k: {"source_a": len(a[k]), "source_b": len(b[k])} for k in groups},
        "feature_engineering_variants": variants,
        "amazon_challenge_model_zero_shot_on_same_test_entities": zero_shot,
        "elapsed_seconds": round(time.monotonic() - start, 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ROOT / "portfolio" / "benchmarks" / "fodors_zagats"))
    parser.add_argument("--seeds", default="42,43,44,45,46", help="Repeated entity-disjoint split seeds")
    parser.add_argument("--candidate-cap", type=int, default=80)
    parser.add_argument("--download", action="store_true", help="Fetch the two public source tables if missing")
    parser.add_argument("--report", default=str(ROOT / "portfolio" / "fodors_zagats_evaluation.json"))
    args = parser.parse_args()
    start = time.monotonic()
    data = Path(args.data)
    required = ("tableA.csv", "tableB.csv")
    if args.download:
        data.mkdir(parents=True, exist_ok=True)
        base = "https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Fodors-Zagats/exp_data"
        for name in required:
            target = data / name
            if not target.exists():
                print(f"Downloading {name} from the DeepMatcher dataset host...", flush=True)
                urlretrieve(f"{base}/{name}", target)
    missing = [name for name in required if not (data / name).is_file()]
    if missing:
        parser.error("missing " + ", ".join(missing) + "; pass --download to fetch the public source tables")
    a_raw, b_raw = read_table(data / "tableA.csv"), read_table(data / "tableB.csv")
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        parser.error("provide at least one seed")
    runs = [evaluate_seed(a_raw, b_raw, seed, args.candidate_cap) for seed in seeds]
    metric_names = ("pair_accuracy", "balanced_accuracy", "precision", "recall", "f1", "f0_5",
                    "macro_entity_f0_5", "candidate_recall")
    summary = {}
    model_keys = ("baseline", "engineered", "amazon_challenge_model_zero_shot_on_same_test_entities")
    for model_key in model_keys:
        summary[model_key] = {}
        for metric in metric_names:
            values = [
                run["feature_engineering_variants"][model_key]["test"][metric]
                if model_key in ("baseline", "engineered") else run[model_key][metric]
                for run in runs
            ]
            summary[model_key][metric] = {
                "mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min": float(np.min(values)), "max": float(np.max(values)),
            }
    result = {
        "benchmark": "Fodors-Zagats restaurant record linkage",
        "source": "DeepMatcher public Fodors-Zagats dataset; original train/valid/test pair splits are not used",
        "protocol": "Repeated 70/15/15 stratified entity-class splits; no restaurant entity class or record crosses splits within a run; source-side all-pairs are evaluated end-to-end; threshold selected on validation macro entity F0.5",
        "label_use": "class is used only to group records and derive match labels; it is excluded from model features",
        "seeds": seeds, "candidate_cap_per_target": args.candidate_cap,
        "summary_across_entity_disjoint_runs": summary,
        "runs": runs,
        "elapsed_seconds": round(time.monotonic() - start, 2),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved report: {args.report}")


if __name__ == "__main__":
    main()
