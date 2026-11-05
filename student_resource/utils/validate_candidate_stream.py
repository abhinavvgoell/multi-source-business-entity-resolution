#!/usr/bin/env python3
"""Low-memory candidate TSV validation and row-aligned match subset check.

The supplied challenge validator stores the entire candidate mapping in Python
objects. For this dataset's 258M candidate links, that is impractical. This
checker streams matching and candidate rows together, keeping only one S1 row's
ID lists in memory. It assumes the official matching validator separately
checks required S1 coverage and matching-file rules.
"""

import argparse
import itertools
import sys


def split_row(line, expected_header, path, line_number):
    text = line.rstrip("\r\n")
    if line_number == 1:
        if text != expected_header:
            raise ValueError(f"{path}: unexpected header {text!r}")
        return None
    first, tab, rest = text.partition("\t")
    if not tab:
        raise ValueError(f"{path}:{line_number}: row has no tab")
    ids = rest.split(",") if rest else []
    if any(not item for item in ids):
        raise ValueError(f"{path}:{line_number}: empty ID inside list")
    return first, ids


def validate(matching_path, candidate_path, valid_ids=None):
    errors = []
    seen_source1 = set()
    rows = candidate_count = matched_count = empty_candidates = 0
    with open(matching_path, "r", encoding="utf-8", newline="") as matches, open(
        candidate_path, "r", encoding="utf-8", newline=""
    ) as candidates:
        for line_number, pair in enumerate(
            itertools.zip_longest(matches, candidates), start=1
        ):
            match_line, candidate_line = pair
            if match_line is None or candidate_line is None:
                errors.append("matching and candidate files have different row counts")
                break
            try:
                match_row = split_row(
                    match_line, "source1_entity_id\tmatched_entity_ids", matching_path, line_number
                )
                candidate_row = split_row(
                    candidate_line,
                    "source1_entity_id\tcandidate_entity_ids",
                    candidate_path,
                    line_number,
                )
            except ValueError as exc:
                errors.append(str(exc))
                break
            if line_number == 1:
                continue
            match_s1, match_ids = match_row
            candidate_s1, candidate_ids = candidate_row
            rows += 1
            if match_s1 != candidate_s1:
                errors.append(
                    f"row {line_number}: S1 order/coverage differs: "
                    f"{match_s1!r} vs {candidate_s1!r}"
                )
                break
            if match_s1 in seen_source1:
                errors.append(f"row {line_number}: duplicate S1 ID {match_s1!r}")
                break
            seen_source1.add(match_s1)
            if len(candidate_ids) != len(set(candidate_ids)):
                errors.append(f"row {line_number}: repeated candidate ID for {candidate_s1}")
                break
            if len(match_ids) != len(set(match_ids)):
                errors.append(f"row {line_number}: repeated matched ID for {match_s1}")
                break
            invalid = [item for item in candidate_ids if not item.startswith(("S2-", "S3-"))]
            if invalid:
                errors.append(f"row {line_number}: invalid candidate ID prefix: {invalid[0]!r}")
                break
            invalid = [item for item in match_ids if not item.startswith(("S2-", "S3-"))]
            if invalid:
                errors.append(f"row {line_number}: invalid matched ID prefix: {invalid[0]!r}")
                break
            if valid_ids is not None:
                invalid = [item for item in candidate_ids if item not in valid_ids]
                if invalid:
                    errors.append(
                        f"row {line_number}: candidate ID does not exist in test data: {invalid[0]!r}"
                    )
                    break
            candidate_set = set(candidate_ids)
            missing = [item for item in match_ids if item not in candidate_set]
            if missing:
                errors.append(
                    f"row {line_number}: match {missing[0]!r} is absent from candidates"
                )
                break
            candidate_count += len(candidate_ids)
            matched_count += len(match_ids)
            empty_candidates += not candidate_ids

    if errors:
        print("FAIL — " + errors[0])
        return 1
    print(
        f"PASS — {rows:,} aligned S1 rows; {candidate_count:,} candidates; "
        f"{matched_count:,} matches are candidate subsets; "
        f"{empty_candidates:,} empty candidate lists."
    )
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matching", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--test-dir", help="Enable ID-existence checks using test_source2/3.tsv")
    args = parser.parse_args()
    try:
        valid_ids = None
        if args.test_dir:
            valid_ids = set()
            for name in ("test_source2.tsv", "test_source3.tsv"):
                with open(f"{args.test_dir}/{name}", "r", encoding="utf-8") as source:
                    next(source, None)
                    valid_ids.update(line.split("\t", 1)[0].strip() for line in source if line.strip())
            print(f"Loaded {len(valid_ids):,} valid S2/S3 IDs for existence checks.")
        return validate(args.matching, args.candidate, valid_ids)
    except (OSError, UnicodeDecodeError) as exc:
        print(f"FAIL — could not read input: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
