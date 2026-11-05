# Synthetic pair stress test: 100 cases

This report runs the saved NumPy pair classifier on 100 deterministic, hand-authored examples. It is a targeted stress test of formatting changes, business-name differences, address changes, and country mismatches. It is not an independently labeled benchmark and does not estimate competition or real-world accuracy.

## Results

At the saved classifier cutoff of 0.85:

- 100 of 100 constructed cases received the expected **demo action**. These actions include `POSSIBLE MATCH`, `NO MATCH`, `REVIEW ADDRESS`, and `NOT EVALUATED`.
- For the raw classifier alone, on the 53 constructed cases designated as clear match/non-match (excluding review and country-mismatch cases), the confusion matrix was TP=28, FP=24, FN=0, TN=1. This gives 54.7% accuracy, 53.8% precision, 100% recall, and 4.0% specificity on this synthetic subset.
- Many intentionally different businesses at the same address got raw scores near 1.0. This confirms that the learned pair score alone is not a reliable final identity decision for these stress examples.

The 100% demo-action agreement measures whether the transparent UI rules produced the action that was specified when each synthetic scenario was written. It is not model accuracy. The synthetic labels and rules were authored together, and the examples are not random or representative. Treat the raw metrics above as diagnostics only.

## Case mix

| Scenario group | Cases | Expected demo behavior |
| --- | ---: | --- |
| Same entity with formatting or minor name variations | 25 | Possible match |
| Different business names at the same address | 24 | No match |
| Related automotive terms (`garage` / `car mechanic`) at the same address | 1 | Review match |
| Same name with conflicting address details | 25 | Review address, except equivalent address formatting |
| Country mismatch | 15 | Not evaluated |
| Fuzzy name and address conflicts | 10 | Review, possible match, or no match based on the authored case |

## Reproduce

From the repository root, run:

```powershell
python portfolio/evaluate_pair_stress.py
```

The script writes `synthetic_pair_stress_100.json` and `.csv` beside this report. Each row includes the inputs, raw score, expected action, predicted action, and whether the actions agree.

## Interpretation and limits

The raw model output is a classifier score, not a calibrated probability. Address and country review decisions in the browser are explicit demo safeguards layered over the score; they are not used by the challenge inference pipeline. Real model quality should be assessed on an independently labeled holdout set. The separate fresh 500-entity holdout report remains the relevant dataset-based evaluation; neither report measures performance on the unavailable challenge test labels.
