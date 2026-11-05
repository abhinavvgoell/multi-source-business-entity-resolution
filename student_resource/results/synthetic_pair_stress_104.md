# Synthetic pair stress test: 104 cases

This run exercises the saved NumPy pair classifier and the portfolio demo decisions on 104 deterministic, hand-authored cases. It is a targeted behavior check, not an independently labeled benchmark and not an estimate of competition or real-world accuracy.

## Automotive related-name cases

At the same normalized address, all five of these comparisons return **REVIEW MATCH**:

| Reference name | Candidate name | Demo action |
| --- | --- | --- |
| garage | car mechanic | Review match |
| garage | car shop | Review match |
| garage | mechanic | Review match |
| garage | auto repair | Review match |
| garage | vehicle repair | Review match |

The demo uses transparent category cues: a vehicle term (car, auto, automotive, vehicle, automobile, motor, or truck) combined with a business/service term (such as shop, repair, mechanic, or service) indicates the automotive category. Terms such as `garage` and `mechanic` are strong category cues themselves. When both names point to the category and the normalized addresses agree, the result is review, never an automatic merge. This is a small rule-based vocabulary, not a semantic model or a trained category classifier.

The generic word `shop` is ignored when checking name conflict: for example, `coffee shop` versus `suit shop` at the same address returns **NO MATCH**, because the distinctive terms conflict. Exact co-location and related category wording can still describe separate businesses, so **REVIEW MATCH** means a person should verify the records.

## Overall results

- All 104 constructed cases received their authored expected **demo action**. This is rule-consistency agreement, not model accuracy; the cases and expected outcomes were authored together.
- For the raw classifier alone, 53 constructed clear match/non-match examples (excluding review and country-mismatch cases) yielded TP=28, FP=24, FN=0, TN=1 at cutoff 0.85: 54.7% accuracy, 53.8% precision, 100% recall, and 4.0% specificity.
- The raw classifier still rates many unrelated businesses at one address very highly. Its output is a score, not a calibrated probability. Use the separate 500-entity labeled holdout report for the dataset-based evaluation; challenge test labels were unavailable.

All synthetic metrics are diagnostics only. The selected examples are deliberately challenging and not random or representative.

## Case mix

| Scenario group | Cases | Expected demo behavior |
| --- | ---: | --- |
| Same entity with formatting or minor name variations | 25 | Possible match |
| Different business names at the same address | 24 | No match |
| Related automotive descriptions at the same address | 5 | Review match |
| Same name with conflicting address details | 25 | Review address, except equivalent formatting |
| Country mismatch | 15 | Not evaluated |
| Fuzzy name and address conflicts | 10 | Review, possible match, or no match per the authored case |

## Reproduce

From the repository root:

```powershell
python portfolio/evaluate_pair_stress.py
```

The script writes `synthetic_pair_stress_104.json` and `.csv` beside this report. Each CSV row contains both names and addresses, raw score, expected action, predicted action, and agreement.

## Limits

The browser review rules are demo-only and do not affect the competition inference pipeline or the reported holdout score. Business terms do not establish identity, and a shared address does not guarantee that two records refer to the same tenant. A real semantic matcher would need broader labeled examples and independent evaluation across business categories.
