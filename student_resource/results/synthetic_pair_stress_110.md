# Synthetic pair stress test: 110 cases

This run exercises the saved NumPy pair classifier and portfolio decision flow on 110 deterministic, hand-authored cases. It targets name variations, unrelated businesses at one address, category-related names, address conflicts, and country differences. It is not an independently labeled benchmark and does not estimate competition or real-world accuracy.

## General category review behavior

When both distinct names map to the same category and their normalized addresses agree, the demo shows **REVIEW MATCH**. It does not automatically merge them. The suite includes:

| Category | Example names | Cases |
| --- | --- | ---: |
| Automotive | garage / car mechanic, car shop, mechanic, auto repair, vehicle repair | 5 |
| Coffee and cafés | coffee shop / cafe, cappuccino, espresso | 3 |
| Nightlife | pub / bar, nightclub, tavern | 3 |

The UI uses an editable alias vocabulary with category groups across several business types (including automotive, coffee/cafés, nightlife, restaurants, bakeries, apparel, beauty, lodging, pharmacy, healthcare, grocery, bookstores, florists, fitness, banking, legal services, and education). For example, `coffee shop` versus `suit shop` remains **NO MATCH** because `shop` is treated as a generic word and the distinctive category terms do not agree.

This is a transparent, rule-based category vocabulary—not a general semantic model. Names that use unfamiliar or ambiguous wording may not map to a category. Even two businesses in the same category at one address can be separate tenants, so the result requests review instead of declaring identity.

## Results

- All 110 constructed cases received their authored expected **demo action**. This is a check of the implemented rules against scenarios written for those rules, not accuracy.
- For the raw classifier alone, 53 constructed clear match/non-match cases (excluding review and country-mismatch cases) yielded TP=28, FP=24, FN=0, TN=1 at cutoff 0.85: 54.7% accuracy, 53.8% precision, 100% recall, and 4.0% specificity.
- The raw classifier still gives many unrelated businesses at one address very high scores. Its output is not a calibrated probability. Use the separate labeled 500-entity holdout for dataset-level evaluation; challenge test labels were unavailable.

These synthetic numbers are diagnostics only. The examples are deliberately selected, not random or representative.

## Case mix

| Scenario group | Cases | Expected demo behavior |
| --- | ---: | --- |
| Same entity with formatting or minor name variations | 25 | Possible match |
| Different business names at the same address | 24 | No match |
| Related automotive names at the same address | 5 | Review match |
| Related coffee/café names at the same address | 3 | Review match |
| Related nightlife names at the same address | 3 | Review match |
| Same name with conflicting address details | 25 | Review address, except equivalent formatting |
| Country mismatch | 15 | Not evaluated |
| Fuzzy name and address conflicts | 10 | Review, possible match, or no match per the authored case |

## Reproduce

From the repository root:

```powershell
python portfolio/evaluate_pair_stress.py
```

The script writes `synthetic_pair_stress_110.json` and `.csv` beside this report. The CSV records both names and addresses, raw model score, expected action, predicted action, and agreement.

## Limits

The category review logic affects only the browser demo; it does not change the trained competition model, challenge outputs, or reported holdout score. Broad semantic coverage would require a semantic model and independently labeled examples for evaluation.
