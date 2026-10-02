# Task 3 · Shibin Thomas — how to run (GPU Lab, Windows PowerShell)

From-scratch CycleGAN, Monet (A) ↔ Photo (B). Run from the **repo root** with the `.venv` active.

```
task3_gan/
├── data/                    monet_jpg/ (300) · photo_jpg/ (7,038) · Part3_Evaluation_Script.ipynb  (images git-ignored)
└── shibin_thomas/
    ├── configs/             cyclegan_v1.yaml (baseline) · cyclegan_v2.yaml (improved) · smoke.yaml
    ├── src/                 data.py · models.py · diffaug.py · train.py · translate.py · gan_metrics.py · human_audit.py
    │                        run_pipeline.py · utils.py · task3_cyclegan.ipynb
    ├── tests/               test_task3.py
    ├── checkpoints/         <run_id>/G_AB.pt, G_BA.pt (+ last_full.pt for resuming, git-ignored)
    ├── outputs/             pred_A2B/ · pred_B2A/ · human_audit/ · <run_id>/ (plots, samples, metrics)
    ├── evaluate_local.py    all metrics + submission.csv (instructor protocol)
    ├── submission.csv       Kaggle file (ID, FID, MiFID)
    ├── kaggle_leaderboard.json   fill in after submitting
    ├── full_metrics_report.csv   every metric, both directions (also as metrics_report.csv)
    ├── failure_analysis.md · results.md
```

## Steps
```powershell
cd $HOME\Gen-AI---LAB-1
.\.venv\Scripts\Activate.ps1
git pull
pip install torchvision lpips --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
python task3_gan/shibin_thomas/tests/test_task3.py
python task3_gan/shibin_thomas/src/run_pipeline.py --config task3_gan/shibin_thomas/configs/smoke.yaml
python task3_gan/shibin_thomas/src/run_pipeline.py
```

What each command does:
1. **`test_task3.py`**: 13 unit tests. Expect `13 tests passed`.
2. **Smoke config**: a tiny model on 24 + 48 images for 2 short epochs, then every evaluation step. It
   downloads the Inception and AlexNet weights used for **measuring** once. Takes 1–3 minutes, and its files
   are git-ignored.
3. **Full run**: 40 epochs of 2,000 unpaired pairs, then translation of the 300 Monets and the first 300
   photos, all metrics, `submission.csv`, and the human-audit pack.
   * The log prints the steps and images/sec at the start, so you can estimate the runtime.
   * If the GPU is shared with another job, training is slower; the GPU state at start is recorded in the
     log and manifest.

**If the run stops partway:** rerun it with `--resume latest`. It continues from the last finished epoch.

## Improved run (v2)
`configs/cyclegan_v2.yaml` keeps the v1 architecture and adds:
* DiffAugment on the discriminators;
* an EMA of the generator weights;
* λ_identity 2.5 instead of 5;
* 60 epochs instead of 40;
* best-epoch selection on held-out photos.

`results.md` §5b explains each change. Commit and push v1 first: v2 rewrites `outputs/pred_A2B`, `pred_B2A` and
`submission.csv`. Then run:
```powershell
python task3_gan/shibin_thomas/tests/test_task3.py
python task3_gan/shibin_thomas/src/run_pipeline.py --config task3_gan/shibin_thomas/configs/smoke.yaml
python task3_gan/shibin_thomas/src/run_pipeline.py --config task3_gan/shibin_thomas/configs/cyclegan_v2.yaml
```
* Interrupted: add `--resume latest` to the last command.
* If v2 scores worse than v1, put v1's images back without retraining:
  `python task3_gan/shibin_thomas/src/run_pipeline.py --skip-train cyclegan_v1_20261001-201801`
* `evaluate_local.py` and the notebook automatically use the run whose images are in `outputs/pred_*`, so
  they need no extra arguments after either run.

## After the full run
1. **Notebook outputs.** This re-uses the run and does not retrain:
   ```powershell
   jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 task3_gan/shibin_thomas/src/task3_cyclegan.ipynb
   ```
2. **Kaggle.**
   1. Upload `task3_gan/shibin_thomas/submission.csv` to the class competition under your team.
   2. Fill in `kaggle_leaderboard.json`: team name, public and private score, rank.
   3. Run `python task3_gan/shibin_thomas/evaluate_local.py` again, so the scores appear in the report.
3. **Human audit** (2 raters).
   1. Each rater opens `task3_gan/shibin_thomas/outputs/human_audit/images/S01.png` … `S30.png` and fills in
      their own `rater1.csv` or `rater2.csv` independently.
   2. Then run:
      ```powershell
      python task3_gan/shibin_thomas/src/human_audit.py score
      python task3_gan/shibin_thomas/evaluate_local.py
      ```
4. **Commit and push.** `git add -A`, `git commit`, `git push`. The images are git-ignored; the predictions,
   generator checkpoints, logs and metrics are committed.

| Problem | Fix |
|---|---|
| CUDA out of memory (shared GPU) | `--set training.batch_size=2` |
| Too slow | `--set training.samples_per_epoch=1000` (halves the run) |
| Windows DataLoader error | `--set training.num_workers=0` |
