# DATA266 Lab 1 — LLM Pretraining · Sentiment Classification · CycleGAN Style Transfer

Team repository for DATA266 Lab 1 (Fall 2026). Every member builds and trains their own models.
Each member's work lives in a named folder inside each task folder.

| Task | Folder | Status |
|---|---|---|
| 1 — GPT-style character LLM from scratch (TinyStories) | [`task1_llm/`](task1_llm/) | Shibin and Denisha complete; Denisha val BPC 0.9842 on Tesla T4 |
| 2 — Yelp Polarity sentiment classification | [`task2_sentiment/`](task2_sentiment/) | Shibin and Denisha complete; Denisha best BiGRU accuracy 95.19% on Tesla T4 |
| 3 — CycleGAN Monet ↔ Photo (Kaggle) | [`task3_gan/`](task3_gan/) | Both independent runs and official FID/MiFID results recorded; human audit still needs Rater 2 |

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
└── report/   DATA266_Lab1_Report_Team_09.pdf
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
Run this from the repository root. It reproduces a member's full pipeline at small scale: it downloads 3,000
TinyStories, preprocesses them, trains for 2 epochs, evaluates every metric, and writes the failure candidates.
It takes a few minutes on a CPU and about one minute on a GPU.
```bash
python task1_llm/shibin_thomas/src/run_pipeline.py --config task1_llm/shibin_thomas/configs/smoke.yaml
```
Unit tests (no data needed): `python task1_llm/shibin_thomas/tests/test_model.py`

## Reproducing a member's full run
| Member | Task 1 command |
|---|---|
| Shibin Thomas | `python task1_llm/shibin_thomas/src/run_pipeline.py --config task1_llm/shibin_thomas/configs/gpt_char_v1.yaml` |

| Member | Task 2 command |
|---|---|
| Shibin Thomas | `python task2_sentiment/shibin_thomas/src/run_pipeline.py` (smoke: add `--smoke`) |

| Member | Task 3 command |
|---|---|
| Shibin Thomas | v1: `python task3_gan/shibin_thomas/src/run_pipeline.py` · v2: add `--config task3_gan/shibin_thomas/configs/cyclegan_v2.yaml` (smoke: `--config task3_gan/shibin_thomas/configs/smoke.yaml`) |

Detailed GPU Lab instructions (including resume after a session ends):
[`task1_llm/shibin_thomas/README.md`](task1_llm/shibin_thomas/README.md).

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
  `task1_llm/shibin_thomas/tests/test_model.py`.

## Team
| Member | Task 1 | Task 2 | Task 3 |
|---|---|---|---|
| Shibin Thomas | [`task1_llm/shibin_thomas`](task1_llm/shibin_thomas/) | [`task2_sentiment/shibin_thomas`](task2_sentiment/shibin_thomas/) | [`task3_gan/shibin_thomas`](task3_gan/shibin_thomas/) |
| Denisha Ketan Tank | [`task1_llm/member_denisha`](task1_llm/member_denisha/) | [`task2_sentiment/member_denisha`](task2_sentiment/member_denisha/) | [`task3_gan/member_denisha`](task3_gan/member_denisha/) |
