# Multi-source Business Entity Resolution

A scalable entity-resolution system for linking business records across three noisy sources with no shared identifier. Built for the Amazon ML Challenge and extended into a reproducible portfolio project with an interactive web demo.

[![Deploy Portfolio Demo](https://github.com/abhinavvgoell/multi-source-business-entity-resolution/actions/workflows/deploy-pages.yml/badge.svg)](https://github.com/abhinavvgoell/multi-source-business-entity-resolution/actions/workflows/deploy-pages.yml)
[![Live Demo](https://img.shields.io/badge/Live_Demo-GitHub_Pages-brightgreen?logo=github)](https://abhinavvgoell.github.io/multi-source-business-entity-resolution/)

👉 **[Launch the Live Interactive Demo](https://abhinavvgoell.github.io/multi-source-business-entity-resolution/)**

## What it does

- Generates a small candidate set per Source 1 record using country-aware token blocks, rare-token overlap, name prefixes, an order-insensitive name signature, address-number prefixes, and postal tokens.
- Scores candidate pairs with interpretable name, address, postal, and country similarity features and a NumPy logistic model.
- Supports multiple matches per Source 1 entity and writes the challenge TSV formats.
- Provides a fresh held-out evaluation, candidate-cap trade-off report, streaming output validator, and browser-based pair scoring demo.

## Evaluation

A fixed model threshold of 0.98 was evaluated on 500 Source 1 entities reserved separately from the 3,000 training entities and the 100-entity split used during training. The held-out entities were matched against the full supplied Source 2 and Source 3 training references. No known held-out positives were injected into candidate generation. The cutoff itself was chosen on a third, disjoint 500-entity slice, so it was never fitted to the entities being reported.

At a cap of 20 candidates per target source per Source 1 record, the run produced 19,762 candidates (39.5 per Source 1 on average), retrieved 82.3% of known links before classification, and achieved macro F0.5 of **0.727** (entity-bootstrap 95% interval **0.703–0.750**). The sweep across caps 10–160 is in `portfolio/evaluation.json` and `student_resource/results/fresh_holdout_500.json`. Candidate recall keeps improving with larger caps but F0.5 is flat from 10 to 160, so cap 20 is the operating point.

Two changes moved this result up from the earlier 0.629 at cutoff 0.85, and both were measured on disjoint entity slices rather than tuned on the reported holdout:

- **Order-insensitive name signature key.** Each record now also indexes the two alphabetically first name tokens, which joins records that share distinctive name words but no 4-character name prefix. It raises candidate recall from 70.9% to 82.3% at the same cap and adds about one candidate per entity.
- **Calibrated cutoff.** F0.5 is precision-weighted, and the saved 0.85 cutoff was leaving precision on the table. The holdout F0.5 curve is flat over 0.975–0.99 and falls away at 0.995, so 0.98 is a plateau point rather than a knife-edge maximum.

A paired ablation on the same two slices (`student_resource/results/blocker_ablation_500.json`) separates the two: the old blocker at its own best cutoff (0.96) reaches 0.667, and the shipped blocker at 0.98 reaches 0.727. `student_resource/results/fresh_holdout_500_baseline_blocker.json` keeps the superseded 0.629 report for comparison.

This is a sample-based offline validation estimate, not the competition test score. The confidence intervals across caps overlap. The competition test labels are unavailable, so test accuracy or leaderboard score cannot be calculated from this project. For entity resolution, F0.5 and candidate recall are more informative than plain accuracy because non-matching pairs vastly outnumber matches.

### Retraining comparison

I retrained the Amazon pair classifier from scratch on 3,000 disjoint Source 1 training entities (48,934 labeled candidate pairs), tuned its cutoff on the same 100 entities used for the saved model, and compared both models on the same 500 held-out entities. To keep the run practical, each reference source used a deterministic 100,000-row reservoir sample plus every labeled reference linked to the selected entities, including the holdout entities. On this identical sampled-reference evaluation at 20 candidates per target source, the saved model scored macro F0.5 **0.899** (95% entity-bootstrap interval **0.883–0.916**), micro precision **99.5%**, and micro recall **76.3%**; the retrained model scored macro F0.5 **0.900** (**0.884–0.917**), micro precision **99.5%**, and micro recall **76.4%**. Both selected cutoff 0.98. The 0.0005 gap is far inside overlapping intervals, so the retrained model was **not promoted** and `student_resource/model/entity_matcher.json` remains the default. Its weights are kept as an experiment at `student_resource/model/entity_matcher_retrained.json`.

An earlier version of this comparison reported 0.930 versus 0.922 and read as a genuine precision-for-recall trade. Re-running it under the current blocker and cutoff calibration dissolved the gap: that split came from scoring the two models at cutoffs tuned under the previous blocker, not from a difference in the weights. The honest conclusion is that re-fitting changed nothing measurable on this holdout, and that a paired model comparison has to be re-measured whenever the blocking stage or the cutoff calibration moves.

These comparison metrics are not ordinary pair accuracy, and they are not directly comparable to the 0.727 full-reference result above. In this bounded comparison, linked reference rows for the holdout entities were deliberately retained so the sample could include their known matches; this makes its absolute candidate recall and scores optimistic versus a naturally sampled reference search. The paired comparison is useful for testing whether retraining helps under the same conditions, not as a replacement full-corpus estimate. Details and confidence intervals are in `portfolio/retraining_comparison.json`; the bounded data-preparation script is `student_resource/code/business_entity_resolution/src/retrain_bounded.py`, and the two reports are `student_resource/results/retrain_new/saved_model_holdout.json` and `.../retrained_model_holdout.json`. Retraining the binary match/non-match model also does not create a supervised review class. The demo's `REVIEW MATCH` / `REVIEW ADDRESS` actions remain separate, transparent UI rules.

### Public, entity-disjoint restaurant benchmark

To measure behavior on a second labeled dataset, the project now includes a reproducible Fodors-Zagats restaurant linkage benchmark. I rebuilt five 70/15/15 train/validation/test splits by restaurant entity class because the dataset's published random pair splits reuse the same records across splits. The entity class is used only for grouping and evaluation labels; it is never a model feature. Within each run, no restaurant record or entity class crosses the split boundary. The evaluation runs the candidate blocker and pair classifier end to end, tuning the threshold on validation data only.

Across five repeated splits, a model trained on each split's Fodors-Zagats training data averaged **99.95% all-pairs accuracy**, **89.5% precision**, **100% recall**, **0.944 F1**, **0.914 F0.5**, and **0.982 macro entity F0.5**. Each test split contains only 18 known matches among roughly 4,264 possible pairs (about 0.42% positives), so all-pairs accuracy is inflated by the many non-matches; precision, recall, F0.5, and the confusion counts are the more useful measures. The mean test precision ranged from 81.8% to 100% across the five split seeds, reflecting the small benchmark. The same source tables are reused across split seeds, so the five runs are a stability check, not five independent datasets or a pooled confidence interval.

For comparison, the saved Amazon-challenge weights transferred zero-shot to the same entity-disjoint test splits averaged **54.6% precision and 100% recall** (mean macro entity F0.5 **0.905**). This exposes a real domain-shift limitation: the Amazon model over-predicts matches on this restaurant data. The higher 0.982 score belongs to models retrained on the benchmark training partition, not to the saved Amazon model. Neither number predicts accuracy on a different dataset without labeled evaluation there.

Reproduce the benchmark from the project root (the two source tables download from the public source if missing):

```powershell
python portfolio/evaluate_fodors_zagats.py --download
```

Full per-seed metrics and confusion matrices are saved in `portfolio/fodors_zagats_evaluation.json`. Dataset source and split notes are in `portfolio/benchmarks/README.md`.

### Feature-engineering experiments and a second benchmark

I compared the original compact feature set with a richer version adding character edit similarity, string-length ratios, numeric address-token overlap, house-number and postal conflicts, unit conflicts, and direction conflicts. On the same five entity-disjoint Fodors-Zagats runs, the richer version **reduced** mean F0.5 from **0.914 to 0.843** and precision from **89.5% to 81.5%** (recall stayed at 100%). The added features were therefore not adopted for the business matcher. This is a useful negative result: more features do not automatically improve the model, and the restaurant benchmark is too small to justify a blanket feature change.

To get a larger, harder feature experiment, the portfolio also evaluates the WDC Products 80%-corner-case benchmark. Its 19,835 training pairs are split by product cluster into train and validation groups, and evaluation uses the official **100%-unseen-product** gold test set (4,500 pairs: 500 matches and 4,000 non-matches). The logistic model is trained on WDC training data; it is not the saved Amazon business model. On this product benchmark, generic title/description/brand lexical features achieved **54.6% accuracy, 18.4% precision, 89.4% recall, and 0.218 F0.5**. Adding edit, character n-gram, numeric, brand-token, field-presence, and currency-aware price signals raised results to **76.4% accuracy, 29.7% precision, 82.0% recall, and 0.340 F0.5**. The richer set reduces false positives while keeping more recall than the baseline, but absolute precision remains low. The benchmark deliberately emphasizes difficult pairs and unseen products, so these scores should not be compared directly with Fodors-Zagats or represented as business-resolution accuracy. WDC Products documents strict record-disjoint splits and its unseen-entity variants on the [official benchmark page](https://webdatacommons.org/largescaleproductcorpus/wdc-products/index.html).

Reproduce from the project root (the archive is not bundled in the lightweight portfolio ZIP; `--download` fetches it if needed):

```powershell
python portfolio/evaluate_wdc_products.py --download
```

The script saves detailed counts, thresholds, and confusion matrices to `portfolio/wdc_products_evaluation.json`. Feature names and dataset provenance are documented in `portfolio/benchmarks/README.md`. To measure **business** accuracy rather than cross-domain behavior, the next useful benchmark must contain labeled business pairs with realistic name/address changes (for example, known duplicates and manually reviewed hard negatives). WDC Products alone cannot answer that business-specific question.

## Run the portfolio demo

The UI is static and has no external runtime dependencies. You can try the **[Live Demo](https://abhinavvgoell.github.io/multi-source-business-entity-resolution/)** directly on GitHub Pages, or serve it locally:

```powershell
cd portfolio
python -m http.server 8000
```

Open `http://localhost:8000`. Use **Pair explorer** to enter two records and inspect the raw classifier score, feature signals, and final demo verdict. The score is not a calibrated probability. The browser demo illustrates pair scoring and does not run the large-scale blocker.

## Synthetic pair stress test

The saved pair model and demo decision flow were exercised on 115 deterministic, hand-authored cases. All 115 constructed cases received the intended **demo action**, which is a rule-consistency check, not accuracy. On the 51 clear synthetic match/non-match cases, the raw classifier at cutoff 0.98 had 52.9% accuracy, 48.9% precision, 100% recall, and 14.3% specificity; these deliberately targeted examples are diagnostics, not representative performance estimates. Two hand-authored cases were re-labelled when the cutoff moved: a shared name at an address in an unrelated city now falls below the cutoff and is declined instead of queued for review. The browser demo uses an editable category vocabulary across multiple business types. Related or category-only names such as `coffee shop` / `cappuccino`, `pub` / `nightclub`, or `garage` / `car shop` at the same normalized address are marked **REVIEW MATCH**. A high raw score with a missing address or a strictly less-detailed address now results in **REVIEW ADDRESS**. See the [detailed stress report](student_resource/results/synthetic_pair_stress_115.md), [CSV](student_resource/results/synthetic_pair_stress_115.csv), and [JSON](student_resource/results/synthetic_pair_stress_115.json).

Reproduce the 115 cases from the repository root with:

```powershell
python portfolio/evaluate_pair_stress.py
```

## Reproduce training and challenge inference

Requires Python 3.10+ and NumPy. The challenge data is expected under `student_resource/dataset/` with train and test TSV folders.

```powershell
cd student_resource
python code/business_entity_resolution/src/business_entity_resolution.py train --data dataset --model model/entity_matcher.json
python code/business_entity_resolution/src/business_entity_resolution.py predict --data dataset/test --model model/entity_matcher.json --output output --max-candidates 80
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

To reproduce the held-out evaluation and candidate-cap sweep:

```powershell
python code/business_entity_resolution/src/evaluate_holdout.py --data dataset --model model/entity_matcher.json --streaming-references --report results/fresh_holdout_500.json
```

Add `--streaming-references` to run that protocol on a normal machine: it keeps the complete S2/S3 corpora as references but indexes only the blocking keys the evaluated targets query, then materializes just the candidate rows. Candidate selection is identical to the in-memory path, so the metrics are directly comparable, and the run needs a fraction of the RAM (about 17 minutes for 500 entities on a laptop).

`train` and `evaluate_holdout.py` both stream the reference tables. Their `--reference-pool-size` flag bounds the reference pool: by default `train` keeps a seeded reservoir per source (200,000 rows) plus every labeled positive for the selected Source 1 entities, and `evaluate_holdout.py` keeps every reference (the protocol behind the 0.727 result). Passing a positive `--reference-pool-size` to the evaluator switches to the same bounded reservoir; that mode is for pipeline checks and paired comparisons, because retaining labeled evaluation references inflates candidate recall, and its report records the caveat.

The supplied TSVs are multi-gigabyte, but training no longer needs them in RAM: it streams each reference source once and normalizes only the retained rows. Both are intentionally excluded from the lightweight portfolio ZIP. Use the complete challenge dataset to reproduce the reported evaluation.

Full-corpus inference has the same problem at a larger scale, so `predict` accepts `--streaming-references`:

```powershell
python code/business_entity_resolution/src/business_entity_resolution.py predict --streaming-references --data dataset/test --model model/entity_matcher.json --output output --max-candidates 80 --workers 4 --shard-rows 25000
```

That mode writes each reference source to an on-disk index once (posting lists capped exactly as `build_index` caps them, plus a memory-mapped table of the reference rows), then scores Source 1 in shards that materialize only the keys their own slice queries. References are read in one forward sweep per shard rather than randomly per candidate, which matters on a machine whose free RAM cannot cache a multi-gigabyte index. Completed shards are kept, so an interrupted run resumes instead of restarting. `tests/test_pipeline.py` asserts that this mode reproduces the in-memory predictor byte for byte.

That mode produced the shipped submission on a 16 GB laptop: 1,732,544 Source 1 rows, 269,698,323 candidate links, and 7,818,829 matched links at cutoff 0.98, in about two hours with four workers. The official validator passes the matching file with no blocking issues, and the streaming candidate checker confirms all 269.7 million candidate links and every matched link against the test sources.

Run the regression tests (no challenge data required) from the project root:

```powershell
python -m unittest discover -s tests -v
```

## Project layout

```text
portfolio/                         Interactive portfolio UI, model JSON, evaluation JSON, stress test runner
student_resource/code/.../src/     Training, inference, holdout evaluation, validation tools
student_resource/model/            Trained logistic model
student_resource/results/          Fresh holdout and synthetic stress test reports
student_resource/dataset/          Challenge-provided data (large; not in portfolio ZIP)
student_resource/output/           Challenge submission TSVs (large; kept separately)
tests/                             Fast regression tests for blocking, scoring, streaming readers, and validators
```

The project root `business_entity_resolution.py` is a thin wrapper that runs the packaged implementation in `student_resource/code/business_entity_resolution/src/`, so the submission copy is the single source of truth and the two cannot drift.

## Resume bullet

Built a three-source business entity-resolution pipeline with rare-token blocking and an interpretable logistic pair scorer; on 500 fresh held-out entities against full reference corpora, reached macro F0.5 0.727 (95% entity-bootstrap interval 0.703–0.750) while scoring an average of 39.5 candidates per entity at a 20-per-source cap.

## Limitations

The fresh holdout contains 500 entities from the supplied training split; the cutoff was calibrated on a third 500-entity slice and the model was fitted on a disjoint 3,000 entities. The cap sweep is exploratory and interval estimates overlap. This is one fixed split, not repeated seeds, so the interval reflects entity sampling only and not split-to-split variance. The interactive demo scores individual pairs and does not demonstrate the full blocking pipeline. No claim is made about unseen challenge test accuracy or winning placement.

## Pair explorer behavior

The browser demo recalculates while fields are edited. It shows the saved classifier score for same-country pairs and raises a review flag when there is no meaningful business-name overlap, because shared street tokens alone can otherwise drive an implausibly high score. Cross-country pairs are shown as not scored: the production candidate blocker is country-aware, and the learned country-match coefficient is zero because eligible training pairs already share country. These browser-only conflict flags are transparent demo safeguards; they are not part of the measured challenge model or the competition output files. The model has no business-category feature, so it cannot infer that one business is a suit shop and another is a coffee shop from category labels.

The pair explorer's displayed final verdict is a separate conservative demo rule layered over the saved model's raw score: strong name contradictions veto an automatic match, names in the same supported business category at an equivalent address trigger review, and address conflicts trigger review. The category list is explicit and editable; it is not a general semantic model. These safeguards are not part of the trained competition model or the reported holdout benchmark; the UI labels the raw score and final demo verdict separately.

The browser pair explorer treats conflicting place tokens (for example Delhi vs Gurgaon/Haryana), differing leading street numbers, differing postal codes, and an address that is absent or omits details present in the other record as an address discrepancy. If the raw score is high, it returns REVIEW ADDRESS rather than auto-matching or auto-rejecting because such differences can be data-entry errors or missing details. A name consisting only of a recognized business category (for example `coffee`) prompts REVIEW MATCH when paired with a longer name in the same category at the same canonical address. These checks are transparent UI heuristics, not a geocoder or validated rules in the competition model; the displayed raw classifier score can still be high and is deliberately labeled separately.

### Interpreting the pair explorer

The primary result is now the final demo verdict (`NO MATCH`, `REVIEW MATCH`, `REVIEW ADDRESS`, or `POSSIBLE MATCH`). The raw classifier output appears separately as diagnostic context and may exceed its cutoff while the final verdict requests review. For example, `garage` versus `car mechanic` at the same normalized address returns `REVIEW MATCH`; Delhi versus Gurgaon/Haryana returns `REVIEW ADDRESS`. The raw score is not a calibrated probability. These demo-only overrides are not used by the saved competition inference code or the holdout benchmark.

