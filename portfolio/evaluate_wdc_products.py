#!/usr/bin/env python3
"""Compare generic pair features on a strict unseen-entity WDC Products split.

WDC's 100%-unseen gold standard is the test set. Cluster IDs are used only to
keep clusters disjoint between training and validation, and are never model
features. The benchmark is product matching, not business matching.
"""
from __future__ import annotations

import argparse
import difflib
import gzip
import json
import math
import random
import re
import sys
import zipfile
from urllib.request import urlretrieve
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "student_resource" / "code" / "business_entity_resolution" / "src"))
import business_entity_resolution as ber  # noqa: E402

ARCHIVE = ROOT / "portfolio" / "benchmarks" / "wdc_80pair.zip"
OUT = ROOT / "portfolio" / "wdc_products_evaluation.json"
PREFIX = "wdcproducts80cc20rnd"
TRAIN_NAME = PREFIX + "000un_train_large.json.gz"
TEST_NAME = PREFIX + "100un_gs.json.gz"
DOWNLOAD_URL = "https://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/wdc-products/80pair.zip"


def load_jsonl_zip(zf: zipfile.ZipFile, filename: str) -> list[dict]:
    with zf.open(filename) as raw, gzip.GzipFile(fileobj=raw) as stream:
        return [json.loads(line) for line in stream]


def normalize(row: dict, side: str) -> dict:
    title = ber.norm(row.get(f"title_{side}") or "")
    desc = ber.norm(row.get(f"description_{side}") or "")
    brand = ber.norm(row.get(f"brand_{side}") or "")
    currency = (row.get(f"priceCurrency_{side}") or "").strip().upper()
    price = row.get(f"price_{side}")
    try:
        price = float(price) if price not in (None, "") else None
        if price is not None and (not math.isfinite(price) or price <= 0):
            price = None
    except (TypeError, ValueError):
        price = None
    return {
        "title": title, "tt": set(ber.words(title)), "desc": desc,
        "dt": set(ber.words(desc)), "brand": brand,
        "bt": set(ber.words(brand)), "price": price,
        "currency": currency,
    }


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / (len(a | b) or 1)


def containment(a: set[str], b: set[str]) -> float:
    return len(a & b) / (min(len(a), len(b)) or 1)


def ratio(a: str, b: str) -> float:
    return min(len(a), len(b)) / max(1, len(a), len(b))


def edit(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() if a or b else 1.0


def base_features(a: dict, b: dict) -> list[float]:
    """Simple lexical baseline, using title, description, and brand fields."""
    return [
        jaccard(a["tt"], b["tt"]), containment(a["tt"], b["tt"]),
        float(bool(a["title"] and a["title"] == b["title"])),
        jaccard(a["dt"], b["dt"]), containment(a["dt"], b["dt"]),
        float(bool(a["desc"] and a["desc"] == b["desc"])),
        float(bool(a["brand"] and b["brand"] and a["brand"] == b["brand"])),
    ]


def engineered_features(a: dict, b: dict) -> list[float]:
    """Add character, edit, numeric, field-presence, and price signals."""
    title_numbers_a = set(re.findall(r"\d+(?:\.\d+)?", a["title"]))
    title_numbers_b = set(re.findall(r"\d+(?:\.\d+)?", b["title"]))
    numeric_overlap = jaccard(title_numbers_a, title_numbers_b)
    title_grams_a, title_grams_b = ber.chargrams(a["title"]), ber.chargrams(b["title"])
    desc_grams_a, desc_grams_b = ber.chargrams(a["desc"]), ber.chargrams(b["desc"])
    brand_similarity = jaccard(a["bt"], b["bt"])
    comparable_price = (a["price"] is not None and b["price"] is not None
                        and a["currency"] and a["currency"] == b["currency"])
    if comparable_price:
        price_similarity = min(a["price"], b["price"]) / max(a["price"], b["price"])
        price_missing = 0.0
    else:
        price_similarity = 0.0
        price_missing = 1.0
    return base_features(a, b) + [
        edit(a["title"], b["title"]),
        jaccard(title_grams_a, title_grams_b), ratio(a["title"], b["title"]),
        edit(a["desc"], b["desc"]), jaccard(desc_grams_a, desc_grams_b),
        brand_similarity,
        numeric_overlap,
        float(bool(title_numbers_a and title_numbers_b and not title_numbers_a & title_numbers_b)),
        price_similarity, price_missing,
        float(bool(a["currency"] and b["currency"] and a["currency"] != b["currency"])),
        float(bool(a["desc"]) and bool(b["desc"])),
    ]


def materialize_pairs(rows: list[dict]) -> tuple[list[tuple[dict, dict]], np.ndarray, list[tuple[int, int]]]:
    pairs = [(normalize(r, "left"), normalize(r, "right")) for r in rows]
    labels = np.asarray([int(r["label"]) for r in rows], dtype=np.int8)
    clusters = [(int(r["cluster_id_left"]), int(r["cluster_id_right"])) for r in rows]
    return pairs, labels, clusters


def split_train_validation(rows: list[dict], seed: int):
    """Split pairs from 000un by product cluster, not by pair row."""
    cluster_ids = sorted({int(r["cluster_id_left"]) for r in rows} |
                         {int(r["cluster_id_right"]) for r in rows})
    rng = random.Random(seed)
    rng.shuffle(cluster_ids)
    cut = int(0.80 * len(cluster_ids))
    train_ids, valid_ids = set(cluster_ids[:cut]), set(cluster_ids[cut:])
    train = [r for r in rows if int(r["cluster_id_left"]) in train_ids
             and int(r["cluster_id_right"]) in train_ids]
    valid = [r for r in rows if int(r["cluster_id_left"]) in valid_ids
             and int(r["cluster_id_right"]) in valid_ids]
    if not train or not valid or not any(r["label"] == 1 for r in valid):
        raise ValueError("Cluster-level split did not produce both train and positive validation pairs")
    return train, valid, len(train_ids), len(valid_ids)


def make_X(pairs, fn):
    return np.asarray([fn(a, b) for a, b in pairs], dtype=np.float64)


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = scores >= threshold
    tp = int(np.sum((labels == 1) & pred)); fp = int(np.sum((labels == 0) & pred))
    fn = int(np.sum((labels == 1) & ~pred)); tn = int(np.sum((labels == 0) & ~pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f05 = 1.25 * precision * recall / (0.25 * precision + recall) if 0.25 * precision + recall else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    accuracy = (tp + tn) / len(labels) if len(labels) else 0.0
    return {
        "pairs": int(len(labels)), "positive_pairs": int(labels.sum()),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "accuracy": accuracy, "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision, "recall": recall, "f1": f1, "f0_5": f05,
        "specificity": specificity,
    }


def tune_threshold(labels, scores):
    thresholds = np.unique(np.r_[np.linspace(0.01, 0.99, 197), scores])
    metrics = [(binary_metrics(labels, scores, float(t))["f0_5"], float(t)) for t in thresholds]
    best = max(m for m, _ in metrics)
    return min(t for m, t in metrics if m >= best - 1e-12)


def run(seed: int = 42, download: bool = False) -> dict:
    if not ARCHIVE.exists():
        if not download:
            raise FileNotFoundError(f"Missing {ARCHIVE}; rerun with --download")
        ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(DOWNLOAD_URL, ARCHIVE)
    with zipfile.ZipFile(ARCHIVE) as zf:
        if zf.testzip() is not None:
            raise ValueError("WDC archive failed its CRC check")
        train_raw = load_jsonl_zip(zf, TRAIN_NAME)
        test_raw = load_jsonl_zip(zf, TEST_NAME)
    train_rows, valid_rows, n_train_clusters, n_valid_clusters = split_train_validation(train_raw, seed)
    # Official 100%-unseen gold-standard clusters are disjoint from 000un training data.
    train_clusters = {int(r["cluster_id_left"]) for r in train_raw} | {int(r["cluster_id_right"]) for r in train_raw}
    test_clusters = {int(r["cluster_id_left"]) for r in test_raw} | {int(r["cluster_id_right"]) for r in test_raw}
    overlap = train_clusters & test_clusters
    if overlap:
        raise ValueError(f"Expected unseen test clusters; found {len(overlap)} overlapping clusters")

    results = {}
    for name, feature_fn in (("lexical_baseline", base_features), ("engineered", engineered_features)):
        train_pairs, y_train, _ = materialize_pairs(train_rows)
        valid_pairs, y_valid, _ = materialize_pairs(valid_rows)
        test_pairs, y_test, _ = materialize_pairs(test_raw)
        model = ber.fit(make_X(train_pairs, feature_fn), y_train, steps=700, lr=0.04)
        validation_scores = ber.probs(model, make_X(valid_pairs, feature_fn))
        threshold = tune_threshold(y_valid, validation_scores)
        test_scores = ber.probs(model, make_X(test_pairs, feature_fn))
        results[name] = {
            "feature_count": int(len(feature_fn(*train_pairs[0]))),
            "training": {"pairs": len(train_rows), "positive_pairs": int(sum(r["label"] for r in train_rows)),
                         "product_clusters": n_train_clusters},
            "validation": {**binary_metrics(y_valid, validation_scores, threshold),
                           "product_clusters": n_valid_clusters, "selected_threshold": threshold},
            "test_unseen_products": {**binary_metrics(y_test, test_scores, threshold),
                                     "product_clusters": len(test_clusters)},
        }
    return {
        "benchmark": "WDC Products 80%-corner-case, 20%-unseen split; product matching",
        "source": "https://webdatacommons.org/largescaleproductcorpus/wdc-products/",
        "protocol": "Train/validation product clusters are disjoint; test uses official 100%-unseen gold standard; validation tunes threshold; cluster IDs are never features.",
        "seed": seed,
        "label_counts_test": {"positive": sum(int(r["label"]) for r in test_raw),
                              "negative": len(test_raw) - sum(int(r["label"]) for r in test_raw)},
        "train_test_cluster_overlap": len(overlap),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true", help="download the official WDC Products benchmark if missing")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    report = run(args.seed, args.download)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for variant, data in report["results"].items():
        m = data["test_unseen_products"]
        print(f"{variant:18} threshold={data['validation']['selected_threshold']:.3f} "
              f"acc={m['accuracy']:.4f} precision={m['precision']:.4f} "
              f"recall={m['recall']:.4f} F1={m['f1']:.4f} F0.5={m['f0_5']:.4f} "
              f"CM={m['confusion_matrix']}")
    print(f"Saved report: {args.output}")


if __name__ == "__main__":
    main()
