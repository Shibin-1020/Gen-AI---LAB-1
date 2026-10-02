<!-- AUTO-DRAFT: template. src/error_analysis.py replaces this file with the 20 selected test errors of my best model after the full run. Review each one and put the notes in your own words; once you delete this line the file is never overwritten again. -->
# Task 2.2.4 - Manual review of 20 errors

_Pending the full training run._ After `run_pipeline.py` finishes, this file lists 20 test-set errors of my best
model (highest validation macro-F1):

| Group | Count | Selection rule |
|---|---|---|
| Confident false positives | 5 | true negative, highest P(positive) |
| Confident false negatives | 5 | true positive, lowest P(positive) |
| Near-threshold errors | 5 | misclassified, P(positive) closest to 0.5 |
| Slice-specific failures | 5 | most confident errors in the slice with the highest error rate |

Each error gets the review text, its probability, an error type (mixed sentiment, negation, truncation,
explicit rating ignored, label noise / sarcasm, weak sentiment) and evidence. A testable fix is proposed
together with how to measure it.
