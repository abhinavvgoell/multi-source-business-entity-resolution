#!/usr/bin/env python3
"""Estimate candidate-link and current-match impact of per-source top-k caps."""

import argparse
import itertools


CAPS = (10, 15, 20, 25, 30, 40, 50, 60, 70, 80)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matching", required=True)
    ap.add_argument("--candidate", required=True)
    args = ap.parse_args()
    totals = {k: 0 for k in CAPS}
    retained = {k: 0 for k in CAPS}
    raw_candidates = {"S2": 0, "S3": 0}
    raw_matches = {"S2": 0, "S3": 0}
    rows = 0

    with open(args.matching, encoding="utf-8", newline="") as mf, open(
        args.candidate, encoding="utf-8", newline=""
    ) as cf:
        if mf.readline().rstrip("\r\n") != "source1_entity_id\tmatched_entity_ids":
            raise ValueError("unexpected matching header")
        if cf.readline().rstrip("\r\n") != "source1_entity_id\tcandidate_entity_ids":
            raise ValueError("unexpected candidate header")
        for line_no, (mline, cline) in enumerate(itertools.zip_longest(mf, cf), 2):
            if mline is None or cline is None:
                raise ValueError("row counts differ")
            ms1, mtab, mtext = mline.rstrip("\r\n").partition("\t")
            cs1, ctab, ctext = cline.rstrip("\r\n").partition("\t")
            if not mtab or not ctab or ms1 != cs1:
                raise ValueError(f"invalid or unaligned row {line_no}")
            matches = mtext.split(",") if mtext else []
            candidates = ctext.split(",") if ctext else []
            by_source = {"S2": [], "S3": []}
            for item in candidates:
                if item.startswith("S2-"):
                    by_source["S2"].append(item)
                elif item.startswith("S3-"):
                    by_source["S3"].append(item)
                else:
                    raise ValueError(f"invalid candidate prefix at row {line_no}: {item}")
            positions = {src: {item: i for i, item in enumerate(ids)} for src, ids in by_source.items()}
            match_ranks = {"S2": [], "S3": []}
            for item in matches:
                src = "S2" if item.startswith("S2-") else "S3" if item.startswith("S3-") else None
                if src is None or item not in positions[src]:
                    raise ValueError(f"match missing from candidate row {line_no}: {item}")
                match_ranks[src].append(positions[src][item])
                raw_matches[src] += 1
            raw_candidates["S2"] += len(by_source["S2"])
            raw_candidates["S3"] += len(by_source["S3"])
            for cap in CAPS:
                totals[cap] += min(cap, len(by_source["S2"])) + min(cap, len(by_source["S3"]))
                retained[cap] += sum(rank < cap for ranks in match_ranks.values() for rank in ranks)
            rows += 1

    print(f"Rows: {rows:,}")
    print(f"Current candidate links: {sum(raw_candidates.values()):,} (S2 {raw_candidates['S2']:,}; S3 {raw_candidates['S3']:,})")
    print(f"Current predicted matches: {sum(raw_matches.values()):,} (S2 {raw_matches['S2']:,}; S3 {raw_matches['S3']:,})")
    print("cap_per_source,candidate_links,retained_current_matches,lost_current_matches")
    for cap in CAPS:
        print(f"{cap},{totals[cap]},{retained[cap]},{sum(raw_matches.values())-retained[cap]}")


if __name__ == "__main__":
    main()
