# DATA266 Lab 1 — LLM Pretraining · Sentiment Classification · CycleGAN Style Transfer

Project repository for DATA266 Lab 1 (Fall 2026). The repository contains the code, experiments,
reproducibility artifacts, and combined report for all three tasks.

| Task | Folder | Status |
|---|---|---|
| 1 — GPT-style character LLM from scratch (TinyStories) | [`task1_llm/`](task1_llm/) | implementation and results included |
| 2 — Yelp Polarity sentiment classification | [`task2_sentiment/`](task2_sentiment/) | multiple model implementations and results included |
| 3 — CycleGAN Monet ↔ Photo (Kaggle) | [`task3_gan/`](task3_gan/) | training, evaluation, and audit artifacts included |
| Final report | [`report/DATA266_Lab1_Report_Team_09.pdf`](report/DATA266_Lab1_Report_Team_09.pdf) | combined team report (rebuild: `python report/build_report.py`) |

```
.
├── README.md                  this file: setup, reproduction, where results live
├── requirements.txt
├── task1_llm/   data/ · shared_eval/ · <member>/
├── task2_sentiment/   data/ · <member>/
├── task3_gan/   data/ · <member>/
├── reproducibility/
│   ├── raw_logs/<task>/<member>/<run_id>/     unedited training/eval logs (evidence trail)
│   └── manifests/<task>/<member>/<run_id>.json   versions, hardware, git commit, config, checkpoint SHA-256
└── report/   DATA266_Lab1_Report_Team_[Team Number].pdf
```

Each member's folder in each task follows the same layout:
`src/` (code + task notebook), `data_processed/`, `checkpoints/`, `outputs/`,
`metrics_report.csv`, `failure_analysis.md` and `results.md`.

## Setup
Python ≥ 3.10.
```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu121   # pick the build matching your CUDA driver
pip install -r requirements.txt
```

## Smoke test (one command)
Run this from the repository root. It reproduces a task pipeline at small scale: it downloads 3,000
TinyStories, preprocesses them, trains for 2 epochs, evaluates every metric, and writes the failure candidates.
It takes a few minutes on a CPU and about one minute on a GPU.
```bash
python task1_llm/<member_name>/src/run_pipeline.py --config task1_llm/<member_name>/configs/smoke.yaml
```
Unit tests (no data needed): `python task1_llm/<member_name>/tests/test_model.py`

## Reproducing a full run
| Member | Task 1 command |
|---|---|
| Project run | `python task1_llm/<member_name>/src/run_pipeline.py --config task1_llm/<member_name>/configs/gpt_char_v1.yaml` |

| Member | Task 2 command |
|---|---|
| Project run | `python task2_sentiment/<member_name>/src/run_pipeline.py` (smoke: add `--smoke`) |

| Member | Task 3 command |
|---|---|
| Project run | v1: `python task3_gan/<member_name>/src/run_pipeline.py` · v2: add the appropriate YAML configuration (smoke: use the task's smoke configuration) |

Detailed GPU Lab instructions and task-specific run notes are available in the README files inside each task folder.

## Where results live
| What | Where |
|---|---|
| Metrics (every required metric) | `task*/<member>/metrics_report.csv` |
| Architecture + hyperparameter justification | `task*/<member>/results.md` |
| Failure / error analysis | `task*/<member>/failure_analysis.md` |
| Loss curves, samples, plots | `task*/<member>/outputs/<run_id>/` |
| Model weights used in the report | `task*/<member>/checkpoints/<run_id>/best_model.pt` |
| Raw logs / manifests | `reproducibility/raw_logs/...`, `reproducibility/manifests/...` |

## Rules we follow
* No personal file paths, credentials or API keys in any file. All paths are relative to the repo root,
  and runs are driven by YAML configs (override with `--set key=value`; overrides are logged).
* Raw logs are append-only and are never edited after a run.
* Task 1 uses no prebuilt Transformer or attention modules. This is checked by
  `task1_llm/<member_name>/tests/test_model.py`.
