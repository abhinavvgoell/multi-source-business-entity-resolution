#!/usr/bin/env python3
"""Build a bounded, entity-disjoint Amazon training sample and retrain/evaluate.

This avoids materializing all 1.7 GB of challenge TSVs as Python dictionaries.
The entity split follows the main trainer's seeded ordering: 100 tuning, then
the requested train block, then a disjoint evaluation block. S2/S3 reference
samples are reservoir-sampled from the full files, with every linked entity for
selected S1 records retained so evaluation positives remain available.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("student_resource/dataset"))
    ap.add_argument("--out", type=Path, default=Path("student_resource/results/retrain_sample"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-size", type=int, default=700)
    ap.add_argument("--tuning-size", type=int, default=100)
    ap.add_argument("--eval-size", type=int, default=200)
    ap.add_argument("--refs-per-source", type=int, default=20000)
    args = ap.parse_args()
    source = args.data / "train"
    dest = args.out / "train"
    dest.mkdir(parents=True, exist_ok=True)

    ids = [r["entity_id"] for r in rows(source / "train_source1.tsv")]
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    required = args.tuning_size + args.train_size + args.eval_size
    if len(ids) < required:
        raise ValueError(f"Need {required} S1 entities; found {len(ids)}")
    wanted = set(ids[:required])
    labels = {}
    for r in rows(source / "train_ground_truth.tsv"):
        sid = r["source1_entity_id"]
        if sid in wanted:
            labels[sid] = r
    if len(labels) != required:
        raise ValueError(f"Expected {required} selected labels, found {len(labels)}")
    positive_ids = {2: set(), 3: set()}
    for r in labels.values():
        for entity_id in filter(None, r["matched_entity_ids"].split(",")):
            prefix = entity_id.split("-", 1)[0]
            if prefix == "S2": positive_ids[2].add(entity_id)
            elif prefix == "S3": positive_ids[3].add(entity_id)

    # Keep target rows in the same deterministic order the trainer will shuffle.
    with (dest / "train_source1.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["entity_id", "business_name", "business_address", "country"], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for r in rows(source / "train_source1.tsv"):
            if r["entity_id"] in wanted:
                writer.writerow(r)
    with (dest / "train_ground_truth.tsv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source1_entity_id", "matched_entity_ids"], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for sid in ids[:required]:
            writer.writerow(labels[sid])

    for number in (2, 3):
        path = source / f"train_source{number}.tsv"
        reservoir = []
        positive_rows = {}
        count = 0
        local_rng = random.Random(args.seed + number)
        for row in rows(path):
            count += 1
            entity_id = row["entity_id"]
            if entity_id in positive_ids[number]:
                positive_rows[entity_id] = row
            else:
                seen_random = count - len(positive_rows)
                if len(reservoir) < args.refs_per_source:
                    reservoir.append(row)
                else:
                    j = local_rng.randrange(seen_random)
                    if j < args.refs_per_source:
                        reservoir[j] = row
        unique = {r["entity_id"]: r for r in reservoir}
        unique.update(positive_rows)
        out_path = dest / f"train_source{number}.tsv"
        with out_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["entity_id", "business_name", "business_address", "country"], delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(unique.values())
        print(f"Source {number}: scanned {count:,}; retained {len(unique):,} rows, including {len(positive_rows):,} linked positives", flush=True)

    metadata = {
        "seed": args.seed,
        "selected_s1_entities": required,
        "tuning_entities": args.tuning_size,
        "training_entities": args.train_size,
        "evaluation_entities": args.eval_size,
        "reference_reservoir_target_per_source": args.refs_per_source,
        "positive_reference_ids_retained": {f"S{k}": len(v) for k, v in positive_ids.items()},
        "protocol": "seeded disjoint S1 split; references reservoir-sampled from full source tables plus every selected S1 linked positive",
    }
    (args.out / "sample_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
