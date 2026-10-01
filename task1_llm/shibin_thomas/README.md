# Task 1 · Shibin Thomas — how to run on the GPU Lab

Everything is config-driven ([`configs/gpt_char_v1.yaml`](configs/gpt_char_v1.yaml)) and runs from the **repo root**.
No paths need editing.

```
shibin_thomas/
├── configs/            gpt_char_v1.yaml (main run) · smoke.yaml (pipeline check)
├── src/                data_prep.py · model.py · train.py · evaluate.py · analyze_failures.py
│                       run_pipeline.py (one command) · utils.py · task1_gpt_tinystories.ipynb
├── tests/              test_model.py (causal mask, no prebuilt modules, LayerNorm, schedule, ...)
├── data_processed/     my split: vocab.json, meta.json, split_indices.npz, example_sequences.txt (+ *.bin, git-ignored)
├── checkpoints/        <run_id>/best_model.pt  (+ last_full.pt for resuming, git-ignored)
├── outputs/            <run_id>/ plots, samples, metrics.json, failure_candidates.md
├── metrics_report.csv  every Task 1 metric (written by evaluate.py)
├── failure_analysis.md 3 failure cases (draft written by analyze_failures.py)
└── results.md          architecture + hyperparameter justification + metrics table
```

## Windows lab PC (PowerShell): quick version
```powershell
winget install --id Git.Git -e          # only if `git` is missing; then close and reopen PowerShell
git clone -b claude/zen-mccarthy-mpamlz https://github.com/Shibin-1020/Gen-AI---LAB-1.git
cd Gen-AI---LAB-1
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128   # RTX 50xx (Blackwell) needs a CUDA 12.8+ build
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python task1_llm/shibin_thomas/tests/test_model.py
python task1_llm/shibin_thomas/src/run_pipeline.py --config task1_llm/shibin_thomas/configs/smoke.yaml
python task1_llm/shibin_thomas/src/run_pipeline.py --config task1_llm/shibin_thomas/configs/gpt_char_v1.yaml
```
There is no `tmux` on Windows. Leave that PowerShell window open while training, and set the PC
not to sleep. If the run stops, rerun the last command with `--resume latest`.

## Step by step (Linux)

**1. Set up the environment (once per machine)**
```bash
git clone <repo-url> && cd <repo-folder>
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
# PyTorch with the CUDA build that matches the lab driver (check `nvidia-smi`), e.g. CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"   # must print True + GPU
```

**2. Check that everything works (~1–3 min)**
```bash
python task1_llm/shibin_thomas/tests/test_model.py
python task1_llm/shibin_thomas/src/run_pipeline.py --config task1_llm/shibin_thomas/configs/smoke.yaml
```
The smoke test downloads 3,000 stories, trains a tiny model for 2 epochs and runs the full evaluation.
Its files are git-ignored and never used in the report.

**3. Full run: 100K/10K split, 10 epochs.** Start it inside `tmux` (or with `nohup`) so it keeps running
if your connection drops:
```bash
tmux new -s task1
python task1_llm/shibin_thomas/src/run_pipeline.py --config task1_llm/shibin_thomas/configs/gpt_char_v1.yaml
#   detach: Ctrl-b then d      re-attach: tmux attach -t task1
```
What happens:
1. Downloads the TinyStories train split (~2.1M stories, ~2 GB) once.
2. Builds my split.
3. Trains for 10 epochs, logging every step and evaluating after every epoch.
4. Evaluates the best checkpoint and writes the plots, samples and `metrics_report.csv`.
5. Fills in the metrics table in `results.md` and writes the draft `failure_analysis.md`.

The log prints `steps/epoch`, `total_steps` and `tok/s` at the start, so you can estimate the runtime
(total characters ≈ 10 × train_tokens).

**4. Did the session end or the machine get wiped mid-run?** Training checkpoints after every epoch. Copy
`task1_llm/shibin_thomas/checkpoints/<run_id>/` and `reproducibility/raw_logs/task1_llm/shibin_thomas/<run_id>/`
to your own storage regularly. Then, on a new session (after putting those folders back):
```bash
python task1_llm/shibin_thomas/src/run_pipeline.py --config task1_llm/shibin_thomas/configs/gpt_char_v1.yaml --resume latest
```

**5. Fill the notebook outputs.** This re-uses the finished run; it does not train again:
```bash
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 \
    task1_llm/shibin_thomas/src/task1_gpt_tinystories.ipynb
```

**6. Review the write-ups, then commit and push**
* `failure_analysis.md`: read the draft, put the observations in your own words, and delete the first
  `<!-- AUTO-DRAFT ... -->` line.
* `results.md` §5: add 2–3 sentences about the loss curves. §7: add your teammates' rows.
```bash
git add -A        # .gitignore already excludes raw data, *.bin, last_full.pt and smoke runs
git status        # expect: checkpoints/<run_id>/best_model.pt, outputs/<run_id>/, metrics_report.csv,
                  #         reproducibility/raw_logs/... and manifests/..., data_processed/main/*.json|npz
git commit -m "Task 1: full 10-epoch run <run_id> with metrics, logs and manifest"
git push
```

## If something goes wrong
| Problem | Fix (overrides are recorded in the log and manifest) |
|---|---|
| CUDA out of memory | `--set training.batch_size=32 training.grad_accum_steps=2` (same effective batch of 64) |
| Hugging Face download blocked | download `TinyStories-train.txt` from the dataset page into `task1_llm/data/raw/`, then add `--set data.raw_path=task1_llm/data/raw/TinyStories-train.txt` |
| Older GPU without bf16 | nothing to do: `precision: auto` switches to fp16 with a GradScaler |
| Only want to re-evaluate | `python task1_llm/shibin_thomas/src/run_pipeline.py --config ... --skip-train <run_id>` |

Do **not** edit or delete files in `reproducibility/raw_logs/`. They are the evidence trail for grading.
