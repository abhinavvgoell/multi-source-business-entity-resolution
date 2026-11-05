#!/usr/bin/env python3
"""Evaluate the saved matcher on unseen S1 entities against S2/S3 references.

The evaluation targets are disjoint from the S1 rows used to fit the model and
from the earlier 100-row threshold-tuning split. By default every training
reference is available to candidate generation, as it would be at inference,
and no evaluation positives are injected into a sampled reference pool. Passing
--reference-pool-size switches to a seeded reservoir for bounded runs; that mode
retains the labeled references of evaluated entities by design, so its recall
and F0.5 are optimistic and the report records a caveat. Passing
--streaming-references keeps the full-corpus protocol but indexes only the
blocking keys the evaluated targets query, which produces the same candidates
in a fraction of the memory.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from array import array
from collections import defaultdict
from pathlib import Path

import business_entity_resolution as ber


def count_rows(path):
    """Count TSV data rows without parsing them."""
    lines = 0
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            lines += chunk.count(b"\n")
    return lines - 1


def iter_source_rows(path):
    """Yield (row_number, prepared_record) for one reference TSV."""
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        for row_number, row in enumerate(reader):
            if row:
                yield row_number, ber.prep_one(dict(zip(header, row)))


def stream_target_index(path, wanted_keys, index_cap):
    """Index only the postings this evaluation can query.

    The full-corpus protocol normalizes both multi-million-row reference tables
    in memory, which needs far more RAM than most machines have. Here the tables
    are streamed once and only the keys emitted by the evaluated targets are
    retained, so candidate selection is identical to build_index plus
    candidate_for but bounded by the target key set instead of the corpus size.
    """
    index = defaultdict(lambda: array("I"))
    for row_number, item in iter_source_rows(path):
        for key in ber.block_keys(item):
            if key not in wanted_keys:
                continue
            post = index[key]
            limit = index_cap if key[1] in ("n", "a") else max(index_cap, 2000)
            # Same acceptance rule as build_index: postings above the cap are
            # never queried, so stop growing them once they pass it.
            if len(post) <= limit:
                post.append(row_number)
    return index


def materialize_rows(path, positions):
    """Second pass keeping only the requested reference row numbers."""
    wanted = set(positions)
    rows = {}
    for row_number, item in iter_source_rows(path):
        if row_number in wanted:
            rows[row_number] = item
            if len(rows) == len(wanted):
                break
    missing = wanted - rows.keys()
    if missing:
        raise ValueError(f"Could not materialize {len(missing)} candidate reference rows.")
    return rows


def select_eval_ids(source1_path, seed, tuning_size, train_size, eval_size):
    ids = []
    with open(source1_path, encoding="utf-8", newline="") as f:
        ids = [row["entity_id"] for row in csv.DictReader(f, delimiter="\t")]
    rng = random.Random(seed)
    rng.shuffle(ids)
    start = tuning_size + train_size
    chosen = ids[start : start + eval_size]
    if len(chosen) != eval_size:
        raise ValueError(
            f"Need {eval_size} fresh evaluation entities after reserving "
            f"{start:,}; only found {len(chosen):,}."
        )
    return set(chosen)


def read_rows_for_ids(path, wanted):
    found = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            entity_id = row["entity_id"]
            if entity_id in wanted:
                found[entity_id] = ber.prep_one(row)
                if len(found) == len(wanted):
                    break
    missing = wanted - found.keys()
    if missing:
        raise ValueError(f"Could not find {len(missing)} evaluation S1 rows.")
    return found


def read_labels(path, wanted):
    labels = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            sid = row["source1_entity_id"]
            if sid in wanted:
                labels[sid] = set(filter(None, row["matched_entity_ids"].split(",")))
    missing = wanted - labels.keys()
    if missing:
        raise ValueError(f"No ground-truth rows for {len(missing)} evaluation S1 entities.")
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="dataset", help="Challenge dataset directory")
    parser.add_argument("--model", default="model/entity_matcher.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tuning-size", type=int, default=100)
    parser.add_argument("--train-size", type=int, default=3000)
    parser.add_argument("--eval-size", type=int, default=500)
    parser.add_argument(
        "--candidate-caps", default="10,20,30,40,50,60,70,80,100,120,160",
        help="Comma-separated per-source caps to compare on the same holdout",
    )
    parser.add_argument("--report", help="Optional JSON path for the metrics report")
    parser.add_argument(
        "--streaming-references", action="store_true",
        help=("Score against the complete S2/S3 corpora while holding only the posting lists "
              "and candidate rows this evaluation can query. Candidate selection matches the "
              "in-memory full protocol exactly, so the metrics are directly comparable, but the "
              "run needs a small fraction of the RAM and two streaming passes per source."),
    )
    parser.add_argument(
        "--reference-pool-size", type=int, default=0,
        help=("0 keeps the documented protocol and scores against the full S2/S3 corpora; "
              ">0 streams a seeded per-source reservoir of that size, retaining every labeled "
              "reference for the evaluated entities. Bounded mode runs on a normal machine but "
              "inflates candidate recall, so its metrics are not comparable to the full run."),
    )
    args = parser.parse_args()
    if args.reference_pool_size < 0:
        parser.error("--reference-pool-size must be zero or positive")
    caps = sorted({int(value) for value in args.candidate_caps.split(",") if value.strip()})
    if not caps or caps[0] < 1:
        parser.error("candidate caps must be positive integers")
    if args.streaming_references and args.reference_pool_size:
        parser.error("--streaming-references and --reference-pool-size are different modes")
    max_cap = max(caps)
    started = time.monotonic()

    root = Path(args.data)
    train_dir = root / "train"
    source1_path = train_dir / "train_source1.tsv"
    source1_ids = select_eval_ids(
        source1_path, args.seed, args.tuning_size, args.train_size, args.eval_size
    )
    targets = read_rows_for_ids(source1_path, source1_ids)
    truth = read_labels(train_dir / "train_ground_truth.tsv", source1_ids)
    model = json.loads(Path(args.model).read_text(encoding="utf-8"))
    predictions = {cap: {sid: set() for sid in source1_ids} for cap in caps}
    candidate_links = {cap: 0 for cap in caps}
    candidate_true_links = {cap: 0 for cap in caps}
    possible_links = 0
    reference_counts = {}

    for source_number, prefix in ((2, "S2"), (3, "S3")):
        source_path = train_dir / f"train_source{source_number}.tsv"
        if args.streaming_references:
            started_source = time.monotonic()
            reference_count = count_rows(source_path)
            key_cap = max(500, min(1200, int(reference_count * 0.00015)))
            print(
                f"Indexing the keys queried by {len(source1_ids):,} targets across "
                f"{reference_count:,} {prefix} references (posting cap {key_cap:,})...",
                flush=True,
            )
            wanted_keys = {key for target in targets.values() for key in ber.block_keys(target)}
            index = stream_target_index(source_path, wanted_keys, key_cap)
            picks = {
                sid: ber.candidate_for(targets[sid], None, index, key_cap, max_cap)
                for sid in source1_ids
            }
            del index, wanted_keys
            print(
                f"  materializing {sum(len(v) for v in picks.values()):,} candidate rows "
                f"({time.monotonic() - started_source:.1f}s so far)...",
                flush=True,
            )
            rows_by_position = materialize_rows(
                source_path, {row for rows in picks.values() for row in rows}
            )
            per_sid_rows = {sid: [rows_by_position[row] for row in rows] for sid, rows in picks.items()}
            del picks, rows_by_position
        else:
            if args.reference_pool_size:
                positives = {rid for sid in source1_ids for rid in truth[sid] if rid.startswith(prefix + "-")}
                print(
                    f"Streaming a {args.reference_pool_size:,}-row {prefix} reservoir "
                    f"(+{len(positives):,} labeled positives)...",
                    flush=True,
                )
                refs = ber.reservoir_sample(
                    source_path, positives, args.reference_pool_size, args.seed + source_number
                )
            else:
                print(f"Preparing full {prefix} reference corpus...", flush=True)
                refs = ber.prep(ber.load(source_path))
            index, key_cap = ber.build_index(refs)
            reference_count = len(refs)
            per_sid_rows = {
                sid: [refs[row_id] for row_id in ber.candidate_for(targets[sid], refs, index, key_cap, max_cap)]
                for sid in source1_ids
            }
            del index, refs
        reference_counts[prefix] = reference_count
        print(
            f"Scoring {len(source1_ids):,} held-out S1 entities against "
            f"{reference_count:,} {prefix} references (posting cap {key_cap:,})...",
            flush=True,
        )
        possible_links += len(source1_ids) * reference_count
        for i, sid in enumerate(sorted(source1_ids), 1):
            target = targets[sid]
            expected = {rid for rid in truth[sid] if rid.startswith(prefix + "-")}
            scored = per_sid_rows[sid]
            scores = (
                ber.probs(model, [ber.features(target, ref) for ref in scored])
                if scored else []
            )
            for cap in caps:
                cut_rows = scored[:cap]
                cut_ids = {ref["id"] for ref in cut_rows}
                candidate_links[cap] += len(cut_rows)
                candidate_true_links[cap] += len(cut_ids & expected)
                predictions[cap][sid].update(
                    ref["id"] for ref, score in zip(cut_rows, scores[:cap])
                    if score >= model["threshold"]
                )
            if i % 100 == 0:
                print(f"  scored {i:,}/{len(source1_ids):,} S1", flush=True)
        del per_sid_rows

    truth_total = sum(len(truth[sid]) for sid in source1_ids)
    singleton_count = sum(not truth[sid] for sid in source1_ids)
    cap_results = []
    for cap in caps:
        per_entity_f05 = []
        precisions, recalls = [], []
        tp_total = pred_total = 0
        for sid in source1_ids:
            expected, predicted = truth[sid], predictions[cap][sid]
            tp = len(expected & predicted)
            p = tp / len(predicted) if predicted else float(not expected)
            r = tp / len(expected) if expected else float(not predicted)
            per_entity_f05.append(ber.f05(expected, predicted))
            precisions.append(p)
            recalls.append(r)
            tp_total += tp
            pred_total += len(predicted)
        rng = ber.np.random.default_rng(args.seed + cap)
        samples = rng.choice(ber.np.asarray(per_entity_f05), size=(2000, len(per_entity_f05)), replace=True)
        bootstrap_means = samples.mean(axis=1)
        cap_results.append({
            "max_candidates_per_source": cap,
            "candidate_links": candidate_links[cap],
            "average_candidates_per_s1": candidate_links[cap] / len(source1_ids),
            "candidate_recall_for_known_links": candidate_true_links[cap] / truth_total if truth_total else 0.0,
            "candidate_reduction_ratio": 1.0 - candidate_links[cap] / possible_links if possible_links else 0.0,
            "macro_f05": float(sum(per_entity_f05) / len(per_entity_f05)),
            "macro_f05_bootstrap_95pct_ci": [float(x) for x in ber.np.quantile(bootstrap_means, [0.025, 0.975])],
            "macro_precision_including_empty_cases": sum(precisions) / len(precisions),
            "macro_recall_including_empty_cases": sum(recalls) / len(recalls),
            "micro_precision": tp_total / pred_total if pred_total else 0.0,
            "micro_recall": tp_total / truth_total if truth_total else 0.0,
            "micro_true_positives": tp_total,
            "predicted_links": pred_total,
        })

    reference_mode = (
        "reservoir" if args.reference_pool_size
        else "full-streamed" if args.streaming_references
        else "full"
    )
    result = {
        "evaluation_protocol": (
            f"fresh held-out S1; {reference_mode} train S2/S3 references; "
            + ("evaluation positives deliberately retained (optimistic candidate recall)"
               if reference_mode == "reservoir" else "no evaluation-positive injection")
        ),
        "reference_mode": reference_mode,
        "reference_pool_size_per_source": args.reference_pool_size or None,
        "seed": args.seed,
        "tuning_entities_reserved": args.tuning_size,
        "training_entities_reserved": args.train_size,
        "evaluation_entities": len(source1_ids),
        "reference_counts": reference_counts,
        "threshold": model["threshold"],
        "ground_truth_links": truth_total,
        "singleton_entities": singleton_count,
        "candidate_cap_sweep": cap_results,
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    if reference_mode == "reservoir":
        result["caveat"] = (
            "Bounded reference sampling retains the labeled references of the evaluated entities "
            "by design, so candidate recall and F0.5 are optimistic versus the full-corpus protocol. "
            "Use this mode for pipeline checks and paired comparisons, not as a headline metric."
        )
    print(json.dumps(result, indent=2), flush=True)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Saved report: {args.report}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
