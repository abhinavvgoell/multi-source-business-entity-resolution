# ML Challenge 2026 Business Entity Resolution Methodology

**Team Name:** Individual Submission  
**Team Members:** Abhinav Goel  
**Submission Date:** 2026-09-26

## 1. Executive Summary

The solution generates country-aware candidate pairs from the supplied business records and scores them with a logistic classifier trained on labeled matches and difficult non-matches. It tunes the decision threshold against held-out Source 1 entities using macro F0.5, which gives precision extra weight and rewards correct empty predictions for singletons.

## 2. Methodology

### 2.1 Problem Analysis

Source 1 is the reference set; each Source 1 entity may match zero, one, or many Source 2 and Source 3 records. The provided brief specifies two training countries and adds France at test time, so country is treated as an open string and never used to exclude test entities. The executive summary's 1-to-1 assumption conflicts with the problem statement and is not used.

### 2.2 Solution Strategy

**Approach Type:** Multi-pass blocking plus pairwise classifier.  
**Core Innovation:** Use several complementary name/address blocks, then retain an explicit empty prediction where evidence is weak.

## 3. Candidate Generation

Blocking unions seven country-aware blocks: rare normalized business-name tokens, rare normalized address tokens, a four-character normalized name prefix, an order-insensitive name signature (the two alphabetically first name tokens), an address digit prefix, exact postal-like tokens, and the exact country label. Very common name and address tokens are suppressed by a frequency filter, ranking prefers rarer shared keys, and candidates are capped per Source 1 record per target source. A reference record and a query record expand keys through the same function, so they cannot disagree about which blocks exist. Candidate pairs are the records actually scored at inference and are written to `output/candidate_pairs.tsv`. Spelling variation and punctuation noise are handled by normalization plus the token, prefix, signature, and digit blocks rather than by fuzzy join keys, which keeps the index small enough for roughly five million reference rows per source.

During model development, a seeded reference pool per source limits memory and runtime; known positive references for sampled Source 1 entities are added to the pool. Full-corpus evaluations and inference instead stream the reference tables once and index only the keys the targets actually query, so the complete S2/S3 corpora are used without materializing them in RAM.

## 4. Matching Model

**Features:** normalized name token Jaccard, address token Jaccard, character trigram Jaccard for name and address, token containment for both fields, normalized exact name/address equality, postal-token overlap, and country equality.  
**Model:** regularized, class-balanced logistic regression implemented with NumPy gradient descent.  
**Threshold selection:** sweep a fixed set of cutoffs on held-out Source 1 entities and choose the cutoff with the highest mean per-entity F0.5.

## 5. Results and Error Analysis

The model was fitted on 3,000 Source 1 training entities and its cutoff selected on a separate 100-entity split. The reported estimate uses 500 further Source 1 entities, matched against the full Source 2 and Source 3 training corpora with no known held-out positive injected into candidate generation, so candidate recall is measured honestly rather than handed to the blocker.

At a cap of 20 candidates per target source the run produced 19,762 candidates (39.5 per Source 1 record), retrieved **82.3%** of known links before classification, and reached **macro F0.5 0.727** (entity-bootstrap 95% interval **0.703–0.750**) at threshold **0.98**. The cutoff itself was calibrated on a third, disjoint 500-entity slice, so it was never fitted on the entities being reported. Two measured changes account for the improvement over the earlier 0.629 result at cutoff 0.85 under the same protocol: the order-insensitive name-signature block (candidate recall 70.9% to 82.3% at the same cap) and the recalibrated precision-weighted cutoff (F0.5 is flat over 0.975–0.99, so 0.98 is a plateau point, not a knife edge). A paired ablation of the two blockers on the same slices is in `results/blocker_ablation_500.json`; the superseded report is kept as `results/fresh_holdout_500_baseline_blocker.json`.

This is a sample-based offline estimate on the supplied training split, not the competition test score or ordinary accuracy. Candidate caps are correlated, so overlapping intervals across caps are not evidence of a winner between neighbouring settings. Likely false positives are businesses sharing generic name or address tokens; likely false negatives are trade-name substitutions, transliterations, abbreviations the token blocks do not share, and links lost to the candidate cap. The test labels are unavailable, so leaderboard accuracy cannot be computed here.

## 6. Conclusion

The pipeline provides reproducible blocking, pair scoring, threshold tuning, and challenge-format output using only the supplied data. Its held-out estimate is sample-based; final performance should be checked on a larger validation sample and challenge leaderboard.

## Appendix A. Code Artefacts

The runnable entry point is `code/business_entity_resolution/src/business_entity_resolution.py`. The README provides the exact train, predict, and validation commands. `src/validate_candidate_stream.py` provides low-memory candidate validation. The trained coefficient file is written under `model/`; inference writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

## Appendix B. Additional Results

Full test inference completed for all **1,732,544** Source 1 entities using threshold **0.98**. `matching_results.tsv` contains 1,732,544 rows: 1,678,887 with at least one predicted match and 53,657 empty lists. The output contains **7,818,829 predicted links**. `candidate_pairs.tsv` contains 1,732,544 aligned rows and **269,698,323 candidate links**, including 347 empty candidate lists — an average of 155.7 candidates per Source 1 record across both target sources.

The run used the bounded-memory streaming path, which writes each reference corpus once to an on-disk index (posting lists capped exactly as the in-memory index caps them, plus a memory-mapped table of reference rows) and then scores Source 1 in resumable shards, reading the reference index in one forward sweep per shard. It completed in about two hours with four worker processes, on a machine whose free RAM could not hold the multi-gigabyte index. The choice between the two candidate-generation blocks, and the saved cutoff, were both measured on entity slices disjoint from the reported holdout rather than on the test set.

The official challenge validator passed the matching output with no blocking issues (required S1 coverage, header, prefixes, duplicate and self-match rules, and row alignment), running without its optional ID-existence check because that check loads every matched and candidate link into memory. A streaming check then covered the remaining rules: 1,732,544 aligned S1 rows, every candidate ID confirmed present in the test sources (9,969,589 valid S2/S3 IDs loaded), no empty or duplicate IDs inside lists, and all 7,818,829 predicted links confirmed to be subsets of that entity's candidate list.

The reported development metric remains **macro F0.5 = 0.727** on 500 held-out entities against full references, not test accuracy. The test set has no labels; its accuracy/F0.5 is unknown until leaderboard evaluation. Do not substitute literature estimates from the research summary for measured dataset results.
