# Reproducibility evidence

* `raw_logs/<task>/<member>/<run_id>/` - **unedited** logs written by the training/evaluation code during the run
  (append-only; a resumed run appends a `RESUMED` line, nothing is ever rewritten). Do not clean these up.
  - Task 1: `train.log`, `train_steps.csv` (every optimizer step), `epochs.csv` (every epoch), `eval.log`
* `manifests/<task>/<member>/<run_id>.json` - written automatically for every run: full config + hash,
  git commit, Python/PyTorch/CUDA versions and `pip freeze`, GPU/CPU model, data hashes, and the SHA-256 of
  each checkpoint, so every number in the report maps to one checkpoint and one log.
