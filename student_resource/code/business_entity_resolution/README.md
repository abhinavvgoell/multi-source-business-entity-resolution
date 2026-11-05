# Business entity resolution

A Python/NumPy pipeline for candidate generation, pair scoring, model training, challenge inference, and a disjoint holdout evaluation. See the repository-root `README.md` for the portfolio summary and resume-safe results.

## Commands

Run from `student_resource/`. Requirements: Python 3.10+ and NumPy.

```powershell
python code/business_entity_resolution/src/business_entity_resolution.py train --data dataset --model model/entity_matcher.json
python code/business_entity_resolution/src/evaluate_holdout.py --data dataset --model model/entity_matcher.json --eval-size 500 --streaming-references --report results/fresh_holdout_500.json
python code/business_entity_resolution/src/business_entity_resolution.py predict --streaming-references --data dataset/test --model model/entity_matcher.json --output output --max-candidates 80 --workers 4 --shard-rows 25000
python code/business_entity_resolution/src/business_entity_resolution.py predict --streaming-references --data dataset/test --model model/entity_matcher.json --output output --max-candidates 80 --workers 4 --shard-rows 25000
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## Bounded training and evaluation

`train` streams each Source 2 / Source 3 table exactly once and normalizes only the rows it keeps: a seeded reservoir of `--reference-pool-size` rows (default 200,000) per source plus every labeled positive for the selected Source 1 entities. It prints staged progress and never materializes a whole multi-gigabyte reference table. A 700-entity / 20,000-row-per-source run completes in about a minute on a laptop.

`evaluate_holdout.py` defaults to the documented full-reference protocol. Add `--reference-pool-size N` to use the same bounded reservoir; that mode is fast but retains labeled evaluation references by design, so the report records an explicit caveat and should not be quoted as the headline metric.

## Bounded inference

`predict --streaming-references` scores the full test corpus without holding a reference index in memory. For each reference source it writes a sidecar once: posting lists capped by the same rule `build_index` uses, a hash index over the blocking keys, and the normalized reference rows behind an offset table. Source 1 is then split into shards (`--shard-rows`, default 25,000): each worker materializes postings only for the keys its slice queries, computes candidates with the unchanged `candidate_for`, and reads that shard's references in one forward sweep of the sidecar instead of one seek per candidate. The two per-source shard outputs are merged in Source 1 order into the challenge files.

Peak memory is a shard's feature matrix plus its postings, not the corpus: a 25,000-row shard peaks near 300 MB, and completed shards on disk make a stopped run resumable. `tests/test_pipeline.py` runs both modes on the same synthetic corpus and asserts the two output files are byte-identical, so the bounded path cannot drift from the documented one.

## Fresh holdout protocol

`evaluate_holdout.py` reserves deterministic, non-overlapping Source 1 groups using seed 42: 100 entities for the model's threshold tuning, 3,000 for training, and 500 for final reporting by default. It uses all available training Source 2 and Source 3 records as references and does not inject known positives into the held-out candidate pools. The saved model and its threshold remain fixed during the candidate-cap sweep. The report includes macro F0.5, entity-bootstrap 95% intervals, candidate recall before classification, micro precision/recall, and candidate volume.

The 500-entity report gives macro F0.5 0.727 (95% bootstrap interval 0.703–0.750) at cap 20, threshold 0.98. This is an offline estimate on labeled challenge training data, not accuracy or an Unstop test/leaderboard score. Candidate-cap intervals are correlated, so the sweep is not evidence of a clear winner between neighbouring caps. See `student_resource/results/fresh_holdout_500.json` for the full sweep.

The cutoff in the saved model was calibrated on a separate 500-entity slice (seed-42 ordering `[3600:4100]`) rather than on the reported entities, and the holdout F0.5 curve is flat over 0.975–0.99, so 0.98 is a plateau point. `results/blocker_ablation_500.json` scores the previous blocker and the shipped one on the same two slices; the earlier 0.629 report is preserved as `results/fresh_holdout_500_baseline_blocker.json`.

## Method and resource profile

Blocking unions country-aware, frequency-filtered rare-token name/address postings, normalized-name prefixes, an order-insensitive name signature (the two alphabetically first name tokens), address digit prefixes, and postal tokens, then ranks and caps candidates per target source. Index construction and query expansion share one key function, so a reference and a target cannot disagree about which blocks exist. A logistic model uses ten string and record-comparison features. Predictions allow zero, one, or many matches for an S1 record.

`--streaming-references` runs the full-corpus protocol in bounded memory: the reference tables are streamed to index only the keys the evaluated targets query, and a second pass materializes just the candidate rows. Candidate selection matches the in-memory path, so the metrics stay comparable.

To reproduce the saved-versus-retrained comparison, retrain on the same split and score both models on the same bounded references (each model uses its own cutoff):

```powershell
python code/business_entity_resolution/src/business_entity_resolution.py train --data dataset --model model/entity_matcher_retrained.json --train-size 3000 --validation-size 100 --reference-pool-size 100000
python code/business_entity_resolution/src/evaluate_holdout.py --data dataset --model model/entity_matcher.json --reference-pool-size 100000 --report results/retrain_new/saved_model_holdout.json
python code/business_entity_resolution/src/evaluate_holdout.py --data dataset --model model/entity_matcher_retrained.json --reference-pool-size 100000 --report results/retrain_new/retrained_model_holdout.json
```

`predict --streaming-references` extends the same idea to submission generation. It writes each reference source once to an on-disk sidecar (posting lists capped exactly as `build_index` caps them, plus a memory-mapped table of reference rows), then scores Source 1 in resumable shards, reading the reference table in one forward sweep per shard. Completed shards are kept. The full test corpus ran this way in about two hours with four workers: 1,732,544 Source 1 records, 269,698,323 candidate links, 7,818,829 matched links at threshold 0.98. `tests/test_pipeline.py` asserts that the sidecar postings equal `build_index`'s and that streamed predictions are byte-identical to the in-memory predictor.

For a low-memory validation of the large candidate output, run `python code/business_entity_resolution/src/validate_candidate_stream.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test`. It checks row alignment, duplicate IDs, allowed source prefixes, match-subset consistency, and candidate ID existence while streaming the candidate TSV.

The challenge TSV data is multiple gigabytes; inference against the full reference corpora still requires substantial RAM and disk. Dataset and generated TSVs are omitted from the portfolio ZIP; the original Unstop submission archive remains separate.
