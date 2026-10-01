<!-- AUTO-DRAFT: template. src/analyze_failures.py replaces this file with a draft built from the real generated samples of the full run. Review that draft and put the observations in your own words; once you delete this line the file is never overwritten again. -->
# Task 1.4 - Sequence model failure analysis

_Pending the full training run._ After `run_pipeline.py` (or the notebook) finishes on the GPU, this file is
replaced by a draft with three failure cases taken from my model's actual generations
(`outputs/<run_id>/samples.md`), one per failure type:

1. **Repetition** - highest repeated-4-gram rate (or word/character stutter).
2. **Broken grammar / invented words** - words that never occur in my training split, doubled words, unbalanced quotes.
3. **Loss of coherence / hallucinated entities** - character-name drift, stories that never end, text beyond the 256-char context.

Each case gives the generated snippet, the failure type, and an observation backed by the measured evidence.
