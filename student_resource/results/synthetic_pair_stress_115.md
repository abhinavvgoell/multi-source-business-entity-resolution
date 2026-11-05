# 115-case synthetic pair stress evaluation

This deterministic, hand-authored suite checks UI decision-rule consistency; it is not a real-world accuracy estimate.

- Expected demo actions: 115/115 (100.0%)
- Clear synthetic match/non-match cases: 51
- Raw classifier accuracy on those constructed cases: 52.9%
- Raw classifier precision / recall / specificity: 48.9% / 100.0% / 14.3%

These cases were authored from expected behaviors and are not representative sampled data. Agreement with the expected UI actions is not model accuracy. The new incomplete-field checks mark a high-scoring pair for address review when one address is absent or is a strict token subset of the other.
