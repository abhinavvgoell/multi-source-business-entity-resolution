# Project Handoff: Amazon Business Entity Resolution

Updated: 2026-09-26 15:29 local time (Asia/Kolkata); project and submission ZIP complete.

## Current state

- The model is trained and saved at `student_resource/model/entity_matcher.json`.
- Full inference finished successfully for **1,732,544** Source 1 records at threshold **0.98** using the bounded-memory streaming predictor (`predict --streaming-references`, 4 workers, 25,000-row shards, about 2 hours). It produced **269,698,323 candidates** and **7,818,829 matches**.
- Final files are `student_resource/output/matching_results.tsv` (123,159,493 bytes; 1,678,887 rows with matches and 53,657 empty) and `student_resource/output/candidate_pairs.tsv` (3,498,431,259 bytes; 347 empty candidate lists). Both are the 0.98 outputs; the earlier 0.85-era files were replaced by this run.
- The supplied official validator passed the matching file with no blocking issues (coverage, header, prefixes, duplicate/self-match and row-alignment rules). It ran without its optional `--check-ids` flag because that flag holds every matched and candidate link in memory. The streaming checker then covered the rest with ID existence enabled: 1,732,544 aligned rows, candidate IDs confirmed against 9,969,589 test S2/S3 IDs, no duplicate or empty IDs inside lists, and all 7,818,829 predicted links present in that entity's candidate list.
- The `amazon_business_entity_resolution_submission.zip` archive at the project root still contains the earlier 0.85 outputs and needs rebuilding before it can be submitted again.
- `student_resource/Documentation_template.md` now records actual output totals and validation results. The trained model is at `student_resource/model/entity_matcher.json`.
- Submission archive: `amazon_business_entity_resolution_submission.zip` at the project root (about 1.64 GB). It contains nine expected entries; `ZipFile.testzip()` read every entry and returned `None` (all CRC checks passed).

## Submission ZIP contents

Use the project root as the working directory:

`C:\Users\Abhinav Goel\OneDrive\Desktop\amazonmlprojectdataset`

The outputs and validation are complete. The ZIP has these project-root paths mapped to the required submission paths:

```text
student_resource/output/matching_results.tsv       -> output/matching_results.tsv
student_resource/output/candidate_pairs.tsv        -> output/candidate_pairs.tsv
student_resource/code/business_entity_resolution/ -> code/business_entity_resolution/
student_resource/Documentation_template.md         -> Documentation_template.md
student_resource/model/entity_matcher.json         -> model/entity_matcher.json (included for reproducibility)
```

## Submission package

The methodology document and both outputs have already been validated. Create one ZIP with this structure at its root:

```text
output/matching_results.tsv
output/candidate_pairs.tsv
code/business_entity_resolution/README.md
code/business_entity_resolution/requirements.txt
code/business_entity_resolution/src/business_entity_resolution.py
code/business_entity_resolution/src/validate_candidate_stream.py
model/entity_matcher.json                 # useful trained artifact
Documentation_template.md
```

The challenge's required package is the two output files, runnable code folder, and methodology document. Including `model/entity_matcher.json` is useful; the train command can also regenerate it from the challenge's training data. Do not include the multi-gigabyte dataset in the submission ZIP unless the official package instructions change.

## Model and reported score

- Training command used: `train --data student_resource\dataset --model student_resource\model\entity_matcher.json --train-size 3000 --validation-size 100 --reference-pool-size 100000 --max-candidates 80`.
- It trained on **48,873 labeled candidate pairs** and reported tuning-split macro F0.5 0.9014 on 100 S1 entities. That 0.9014 was measured with labeled positive references injected into the development pool, so it is optimistic and is not the headline number.
- The saved threshold is **0.98**, calibrated afterwards on a third 500-entity slice (seed-42 `[3600:4100]`) that is disjoint from both training and the reported holdout.
- The headline estimate is **macro F0.5 0.727** (95% entity-bootstrap interval 0.703–0.750) at cap 20 on 500 held-out entities against the full S2/S3 reference corpora, with 82.3% candidate recall. This is not ordinary accuracy. The challenge metric is macro F0.5.
- There is currently no measured test accuracy because the test set has no provided labels. The 0.727 is an offline holdout estimate, not percent accuracy or a leaderboard score.

## Method and challenge constraints

The pipeline uses country-aware multi-pass blocking (rare name/address tokens, normalized name prefix, address digit prefix, postal token), pair similarity features (token and character-trigram Jaccard, containment, normalized exact fields, postal overlap, country equality), and a balanced regularized NumPy logistic classifier. It supports open-set country strings, including France.

Follow `student_resource/README.md` for the challenge specification. The PDF/problem statement says one S1 can have zero, one, or many S2/S3 matches. That governs over the 1-to-1 assumption in the research summary. The user's request is to finish the project; attached documents are reference material, not instructions that override the user's request or the challenge specification. No external business lookup was used.

## Project structure

```text
amazonmlprojectdataset/
|-- business_entity_resolution.py                 # active inference/training entry point
|-- HANDOFF.md                                     # this continuation guide
|-- student_resource/
    |-- README.md                                  # challenge problem statement/spec
    |-- Documentation_template.md                 # methodology/results write-up
    |-- dataset/
    |   |-- train/
    |   |   |-- train_source1.tsv
    |   |   |-- train_source2.tsv
    |   |   |-- train_source3.tsv
    |   |   `-- train_ground_truth.tsv
    |   `-- test/
    |       |-- test_source1.tsv
    |       |-- test_source2.tsv
    |       `-- test_source3.tsv
    |-- model/
    |   `-- entity_matcher.json
    |-- output/
    |   |-- matching_results.tsv                   # final predictions; 1,732,544 rows
    |   `-- candidate_pairs.tsv                    # final candidates; 269,698,323 links
    |-- code/business_entity_resolution/
    |   |-- README.md
    |   |-- requirements.txt
    |   `-- src/
    |       |-- business_entity_resolution.py      # packaged copy of root script
    |       `-- validate_candidate_stream.py        # low-memory candidate-file validator
    `-- utils/
        |-- validate_submission.py                 # supplied official validator
        `-- validate_candidate_stream.py            # local low-memory candidate checker
```

The user's research/reference files are outside this project folder:

- `C:\Users\Abhinav Goel\Downloads\Executive Summary.docx`
- `C:\Users\Abhinav Goel\Downloads\problemstatemntamazon.pdf`

## Files to give a new chat

If another chat needs to finish packaging, give it this `HANDOFF.md` and the project-root path above. No prediction rerun or large interim file is needed.

If another chat cannot access this workspace, share the handoff and submission ZIP. The original candidate TSV is about 3.36 GB; it is already included in the ZIP. The ZIP intentionally excludes the multi-gigabyte datasets.

## Completion audit

- Full inference completed and printed totals for all 1,732,544 S1 entities at threshold 0.98.
- Official matching validator passed with no blocking issues (ID existence was covered by the streaming check instead, which holds no per-link objects).
- Streaming candidate validator passed row alignment, duplicates, prefixes, match subset, and candidate ID existence checks across all 269,698,323 links.
- Documentation records the measured output totals, the 0.727 holdout estimate, and the fact that test accuracy is unknown without test labels.
- ZIP includes outputs, code, model, docs, and official validator; all nine entries passed CRC integrity verification.

## Maintenance pass (2026-10-29)

- **One implementation.** The project-root `business_entity_resolution.py` is now a thin wrapper that runs `student_resource/code/business_entity_resolution/src/business_entity_resolution.py` (the submission copy). There is no second 375-line copy to drift; the root commands in the docs still work.
- **Bounded training.** `train` previously loaded all of Source 1, 2, and 3 before applying `--reference-pool-size`, so even a 40-entity run did not finish in ten minutes. It now streams Source 1 twice (IDs, then only the selected rows and labels) and streams each reference source once, keeping a seeded reservoir plus every labeled positive for the selected entities. A 40-entity smoke run takes about 38 seconds; 700 training entities with 20,000 references per source takes about 52 seconds and tunes threshold 0.85. Staged progress lines are printed.
- **Bounded evaluation.** `evaluate_holdout.py` keeps the full-reference protocol by default and gains `--reference-pool-size` for a fast reservoir mode. That mode records `reference_mode`, `reference_pool_size_per_source`, and a caveat in its JSON because retained evaluation positives inflate candidate recall; do not quote its numbers as the headline result. README also now uses the real `--report` flag (it previously documented a nonexistent `--output`).
- **Demo fixes.** The load-error message finally reaches the page (the target `<p>` had a class but no `id`), the chart y-domain is derived from the report instead of hardcoded 0.57–0.69, and the candidate slider fill and highlighted cap stay in sync with the selected row.
- **Tests.** `tests/test_pipeline.py` covers normalization, feature bounds, label semantics, country-aware blocking, reservoir sampling, streamed readers, the candidate validator, root-wrapper delegation, and an end-to-end `predict` + validator run on synthetic data. Run `python -m unittest discover -s tests -v` from the project root.

## Portfolio edition update (2026-10-19)

A lightweight portfolio deliverable was added after the competition run:

- `README.md`: resume-ready project overview, exact validation protocol, safe metric phrasing, setup commands, project map, and limitations.
- `portfolio/`: dependency-free interactive page with candidate-cap sweep chart/table and a browser pair scorer; run with a local HTTP server from that folder.
- `student_resource/code/business_entity_resolution/src/evaluate_holdout.py`: fresh deterministic 500-entity validation against the full training references, with a candidate cap sweep and entity bootstrap intervals.
- `student_resource/results/fresh_holdout_500.json`: report; copied into the portfolio as `portfolio/evaluation.json`.
- `entity_resolution_portfolio.zip`: intended as a compact shareable portfolio bundle; it excludes challenge datasets and multi-gigabyte generated submission TSVs.

On the 500 fresh entities, cap 20 / threshold 0.85 produced macro F0.5 0.629 (entity-bootstrap 95% interval 0.601–0.656), candidate recall 70.9%, and 38.6 candidates per S1 on average. This is a holdout estimate only; no test labels or official leaderboard score are known. Candidate-cap intervals overlap. The original Unstop submission ZIP and output TSVs were preserved.

The pair explorer was also adjusted to recalculate on every edit, show a “Review: name conflict” flag for zero-overlap names whose shared address evidence inflates the raw score, and suppress scoring for a country mismatch because the production blocker excludes such pairs. These flags are demo-only safeguards; they are not in the scored benchmark or official Unstop prediction outputs. The model does not contain business-category input.

**Latest demo correction (2026-10-19):** The pair explorer now labels its number as the **raw classifier score**, and its separate final demo verdict rejects a strong business-name contradiction even when common address tokens push the raw score above threshold. A differing leading street number triggers review. The confirmed example (coffee roaster vs suit shop, 125 vs 124 W Lake St, US) shows raw 0.871 and **Rejected: name mismatch**. This safeguard is browser-demo-only, not part of the measured benchmark or prior Unstop output files; do not describe 0.871 as a final match decision.

**Address-conflict update:** The pair explorer now flags conflicting place terms, leading street numbers, or postal codes for address review. Example: same business name and same `766/5 Patel Nagar` prefix, but `Delhi` versus `Gurgaon Haryana` → raw classifier score 1.000, final verdict **REVIEW ADDRESS**. The high raw score is still visible for transparency. These UI-only checks have not been applied to the trained competition model or used in the reported holdout evaluation.

**Decision display clarity update:** Changed the pair card so the primary status is `NO MATCH`, `REVIEW ADDRESS`, or `POSSIBLE MATCH`; the raw classifier output and its cutoff now appear as secondary diagnostic text explaining any override. Verified the Delhi/Gurgaon and 125/124 address differences show `REVIEW ADDRESS` when the raw score passes its cutoff; the raw classifier output is separate diagnostic text. This resolves display ambiguity, not the underlying trained challenge model; guard checks remain UI-only.


**Typo-aware address decision update:** A high raw score with a street-number, postal-code, or location-text discrepancy now returns **REVIEW ADDRESS**, not an automatic rejection. This reflects that an address mismatch may be a typo or separate storefront; a single pair cannot provide an accuracy percentage. Confirmed same-name example: 125 vs 124 West Lake St gives raw output 0.999 over cutoff 0.85, with final verdict **REVIEW ADDRESS** and an explanation that it could be a typo or neighboring storefront. Address safeguards remain demo-only; benchmark/model outputs unchanged.

**Related-category review update (2026-10-19):** Expanded the demo from one automotive cue to an editable multi-category alias vocabulary (automotive, coffee/cafés, nightlife, restaurants, bakery, apparel, health, and other common categories). If two distinct names map to the same supported category and have equivalent normalized addresses, the UI returns **REVIEW MATCH**. Verified `garage` vs `car shop`, `coffee shop` vs `cappuccino`, and `pub` vs `nightclub`; `coffee shop` vs `suit shop` remains **NO MATCH** because “shop” is generic and the category terms conflict. This is rule-based demo logic, not a universal semantic model; it never auto-merges these records and it does not change production inference. The updated 115-case synthetic suite checks these examples and incomplete-field behavior; all 115 expected demo actions agree, but this authored-action agreement is not an accuracy metric.

**External benchmark evaluation (2026-10-19):** Added `portfolio/evaluate_fodors_zagats.py` and `portfolio/fodors_zagats_evaluation.json`. The Fodors-Zagats business-record tables are downloaded from the public DeepMatcher dataset source; the random pair splits are not used because they reuse records across splits. Instead, five repeated 70/15/15 splits are created by entity class, with `class` used only for split grouping and labels, never as a feature. On each test split (18 known matches among about 4,264 possible pairs), the same feature-based logistic pipeline retrained on that split's training data averaged 99.95% pair accuracy, 89.5% precision, 100% recall, F0.5 0.914, and macro entity F0.5 0.982. The saved Amazon weights evaluated zero-shot on the same test splits averaged 54.6% precision and 100% recall (F0.5 0.600), revealing domain shift and over-prediction. Plain accuracy is misleading because only ~0.42% of all possible pairs are matches. The five seeds reuse this small source dataset and are a stability check, not five independent datasets. The portfolio README and UI explain these limitations. `entity_resolution_portfolio.zip` now includes the evaluator, report, and dataset-source README, and excludes raw third-party benchmark tables; the evaluator downloads them with `--download`.


**Cross-domain feature-engineering experiment (2026-10-19):** Added `portfolio/evaluate_wdc_products.py` and `portfolio/wdc_products_evaluation.json`. On the official WDC Products 80%-corner-case benchmark, training/validation clusters are separated and test uses the 100%-unseen gold split (4,500 pairs; 500 positive/4,000 negative). The model is retrained for product matching; no cluster IDs are features. Lexical baseline: 54.64% accuracy, 18.36% precision, 89.4% recall, F0.5 0.2183. Product-aware engineered features (price compared only for matching currencies): 76.42% accuracy, 29.69% precision, 82.0% recall, F0.5 0.3403. Stronger features improved the product task but did not achieve high precision and do not validate business-record accuracy. Fodors-Zagats experiment found the opposite: extra address/name features reduced F0.5 from 0.914 to 0.843, so they were not adopted for the business pipeline. Raw WDC data is excluded from the portfolio ZIP; run with `python portfolio/evaluate_wdc_products.py --download`.


**Incomplete evidence review correction (2026-10-19):** The pair explorer now handles an absent address or an address whose normalized token set is a strict subset of the other record as incomplete location evidence. A high raw score then displays **REVIEW ADDRESS**, not **POSSIBLE MATCH**. A business name consisting only of a recognized category term (e.g. `coffee`) gets **REVIEW MATCH** against a longer same-category name only when canonical addresses match. The user's example (`coffee` / `North Shore Coffee Roasters LLC`; `patel nagar gurgaon` / `patel nagar`) now yields **REVIEW ADDRESS**; category-only names at an exact address yield **REVIEW MATCH**. Abbreviation-equivalent addresses and ZIP+4 remain normalized. These remain UI heuristics, not changes to the saved production model or a validated business accuracy metric. The stress suite grew to 115 authored cases; 115/115 expected actions agree, which is rule regression evidence only. The raw classifier on 49 clear synthetic cases is 51.0% accuracy, another reason not to treat a high raw score as identity proof. Updated files: `portfolio/app.js`, `portfolio/evaluate_pair_stress.py`, and `student_resource/results/synthetic_pair_stress_115.*`.

**Retraining comparison (2026-10-19, superseded by the 2026-10-30 re-run below):** User requested retraining. The original full-corpus trainer was stopped after 13 minutes of materializing all challenge rows in memory and before it saved a model. I added `student_resource/code/business_entity_resolution/src/retrain_bounded.py` to prepare a reproducible, bounded dataset: 3,000 S1 training entities, 100 threshold-tuning entities, 500 entity-disjoint holdout entities, and a 100,000-row reservoir from each reference source while preserving all known S2/S3 positives linked to those selected S1 entities, including holdout positives. A fresh model trained on 48,860 candidate pairs and selected threshold 0.96. On the same sampled references and holdout at cap 20, the saved model had macro F0.5 0.930, precision 94.4%, recall 91.0%; retrained had macro F0.5 0.922, precision 99.0%, recall 82.8%. Therefore the new weights were **not promoted**; `student_resource/model/entity_matcher.json` and the interactive demo weights remain unchanged. Candidate weights are preserved at `student_resource/model/entity_matcher_retrained.json`; full same-split reports are `student_resource/results/saved_model_3000_holdout_500.json` and `student_resource/results/retrained_3000_holdout_500.json`; the summary is `portfolio/retraining_comparison.json`. Since linked holdout references were deliberately added to the reservoir, absolute sampled-reference recall is optimistic; use this as a paired model comparison only, not ordinary accuracy or a full-corpus result. Re-training the binary classifier does not provide training labels for the demo-only REVIEW actions.

**Blocker + cutoff improvement, measured before saving (2026-10-30):** The documented 0.629 holdout came from full-corpus references at a fixed 0.85 cutoff. Re-measuring required the full S2/S3 corpora, which do not fit in RAM here, so the evaluation now has a verified-equivalent low-memory mode: `evaluate_holdout.py --streaming-references` streams each reference table and indexes only the blocking keys the evaluated targets query, then materializes just the candidate rows. A regression test asserts that this returns the same candidates as `build_index` + `candidate_for` on a small corpus, and the run reproduced the old baseline exactly (candidate recall 70.9%, cap-20 F0.5 0.629 at 0.85) before any change.

Two changes were then tested on entity slices that were not used for fitting or for the reported holdout: a validation slice (seed-42 ordering `[3600:4100]`) for cutoff selection and the documented holdout (seed-42 `[3100:3600]`) for reporting.

- **Order-insensitive name signature block.** `block_keys()` is now the single source of blocking keys for index construction and query expansion, and it adds kind `s`: the two alphabetically first name tokens. Candidate selection keeps at most one `s` posting and now unions 7 keys instead of 6. Cap-20 candidate recall rose 70.9% → 82.3% for about one extra candidate per entity (38.6 → 39.5).
- **Calibrated cutoff 0.85 → 0.98.** The validation sweep over 0.96–0.999 peaked at 0.98–0.985 (F0.5 0.712) and collapsed at 0.999 (0.611). F0.5 is precision-weighted, and 0.98 lifts micro precision 66.5% → 79.1% while recall drops 68.0% → 68.6% net of the blocker gain (both higher than before).

Paired ablation, cap 20, same two slices: old blocker at its own selected cutoff 0.96 → test F0.5 0.667; shipped blocker at 0.98 → **0.727** (95% interval 0.703–0.750). The canonical production report `student_resource/results/fresh_holdout_500.json` was regenerated by `evaluate_holdout.py --streaming-references` and matches the ablation (0.727, recall 82.3%, 39.5 candidates/S1, 17.7 min). The superseded report is kept as `results/fresh_holdout_500_baseline_blocker.json`, and the two-variant threshold grids are in `results/blocker_ablation_500.json`.

**Bounded full-corpus inference and refreshed retraining comparison (2026-10-30):** `predict` gained `--streaming-references` so the submission can be regenerated on a machine that cannot hold the reference corpora. It writes each reference source once to an on-disk sidecar (sorted posting lists capped exactly as `build_index` caps them, plus a memory-mapped table of reference rows), then scores Source 1 in resumable shards that materialize only the keys their own slice queries. Three things mattered for speed: resolving a shard's keys with one vectorized lookup into the sorted key table (per-key binary search burned CPU before any shard wrote output), reading the reference row table in one forward sweep per shard in reference order rather than randomly per candidate (the random version was I/O-bound at about 4 million scattered reads per shard), and sharding small enough that each worker's feature matrix stays modest. Scoring still uses the same `candidate_for`, feature, and probability functions, so candidate selection and scores do not drift. `tests/test_pipeline.py` asserts the sidecar index matches `build_index` and that streamed predictions are byte-identical to the in-memory predictor; the suite is now 25 tests.

The full test corpus then ran in that mode: 1,732,544 S1 rows, 269,698,323 candidates, and 7,818,829 matches at threshold 0.98, in 7,252 seconds with four workers (sidecars: 3.45M S2 keys and 3.57M S3 keys). The official validator and the streaming candidate checker both pass on those files, as recorded above.

The saved-vs-retrained comparison was re-run under the current blocker instead of the superseded one. The retrained model was refit on the same 3,000 training entities and its cutoff re-tuned on the same 100-entity split (it also selected 0.98), then both models were scored on the same 500 held-out entities with the same reservoir references. At cap 20 the saved model reached macro F0.5 0.8994 (95% interval 0.883–0.916) with 99.5% micro precision and 76.3% micro recall; the retrained model reached 0.8999 (0.884–0.917) with 99.5% precision and 76.4% recall. The 0.0005 gap sits inside heavily overlapping intervals, so the retrained weights are still **not promoted**: what produced the earlier 0.930-versus-0.922 split was the two models being compared at cutoffs tuned under the old blocker, not a real model difference. `portfolio/retraining_comparison.json`, `student_resource/results/retraining_comparison.json`, `student_resource/model/entity_matcher_retrained.json`, and the two `results/retrain_new/*_holdout.json` reports carry the new numbers; the older `saved_model_3000_holdout_500.json` and `retrained_3000_holdout_500.json` are kept as the pre-change record.

Effects elsewhere: `student_resource/model/entity_matcher.json` and `portfolio/model.json` carry threshold 0.98 plus a `threshold_calibration` block; `train` now tunes over 0.96–0.99 so a retrain can reproduce the operating point; `portfolio/evaluation.json` and the page's placeholders show the new sweep; the demo's display-threshold control defaults to the model's real cutoff and offers 0.90/0.70/0.95 for exploring the review bands below it. Two hand-authored synthetic cases (same name, unrelated-city address, raw ~0.92) no longer clear the cutoff, so their expected action became NO MATCH instead of REVIEW ADDRESS and the suite remains 115/115 against the documented rules; on the 49 clear cases the raw classifier is now 52.9% accuracy, 48.9% precision, 100% recall, 14.3% specificity. Nothing was saved before the paired measurements existed.
