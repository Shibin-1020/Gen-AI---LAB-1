# Task 2 · Shibin Thomas — how to run (GPU Lab, Windows PowerShell)

Three sentiment models on Yelp polarity (no pretrained embeddings): `baseline_meanpool`, `exp1_textcnn`,
`exp2_bigru_attn`. Run everything from the **repo root**, with the `.venv` from Task 1 activated.

```
shibin_thomas/
├── configs/            data.yaml (preprocessing) · baseline_meanpool.yaml · exp1_textcnn.yaml · exp2_bigru_attn.yaml · data_smoke.yaml
├── src/                data_prep.py · models.py · train.py · evaluate.py · error_analysis.py · run_pipeline.py · utils.py
│                       task2_yelp_sentiment.ipynb
├── tests/              test_task2.py
├── data_processed/     main/: vocab.json, meta.json, split_indices.npz, slices_*.npz  (+ X_*.npy etc., git-ignored)
├── checkpoints/        <run_id>/<model>/best_model.pt
├── outputs/            <run_id>/ comparison, plots, predictions, error candidates · eda/main/ EDA plots
├── metrics_report.csv  every metric, every model
├── comparison.md       side-by-side table + slices + McNemar
├── failure_analysis.md 20-error review (draft written by error_analysis.py)
└── results.md          preprocessing + architecture/embedding justification + results + analysis
```

## Steps (PowerShell, one line at a time)

```powershell
cd $HOME\Gen-AI---LAB-1
.\.venv\Scripts\Activate.ps1
git pull
pip install -r requirements.txt
python task2_sentiment/shibin_thomas/tests/test_task2.py
python task2_sentiment/shibin_thomas/src/run_pipeline.py --smoke
python task2_sentiment/shibin_thomas/src/run_pipeline.py
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 task2_sentiment/shibin_thomas/src/task2_yelp_sentiment.ipynb
git add -A
git status
git commit -m "Task 2: full run - 3 models, metrics, logs, manifest, notebook outputs"
git push
```
If `Activate.ps1` is blocked, first run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`.

What each command does:
1. **`test_task2.py`**: the 8 unit tests (no data needed). Expect `8 tests passed`.
2. **`--smoke`**: downloads 4,000 train and 4,000 test reviews, trains each model for 1 epoch and runs every
   metric. Takes about a minute. Its files are git-ignored.
3. **Full run**:
   1. downloads Yelp polarity (560K + 38K reviews) once;
   2. EDA and preprocessing, which takes several minutes;
   3. trains the 3 models;
   4. evaluates them on the 38K test reviews and draws the plots;
   5. drafts the 20-error review.
4. **Notebook**: fills the notebook outputs from the finished run. It does not train again.

**If the run stops partway:** rerun it with `--resume latest`. Models that already finished are skipped.

| Problem | Fix |
|---|---|
| CUDA out of memory | `--set training.batch_size=128` |
| Hugging Face blocked | put `train.csv` / `test.csv` from `yelp_review_polarity_csv.tgz` in `task2_sentiment/data/raw/`, then add `--set data.raw_train=task2_sentiment/data/raw/train.csv data.raw_test=task2_sentiment/data/raw/test.csv` |
| Re-evaluate only | `--skip-train <run_id>` |

Do **not** edit files in `reproducibility/raw_logs/`.
