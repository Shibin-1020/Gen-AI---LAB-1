"""One command for the whole Task 1 pipeline: preprocess -> train -> evaluate -> failure candidates.

    # smoke test (~minutes on CPU; downloads 3,000 stories if needed)
    python task1_llm/shibin_thomas/src/run_pipeline.py --config task1_llm/shibin_thomas/configs/smoke.yaml

    # full run (GPU): 100K/10K split, 10 epochs
    python task1_llm/shibin_thomas/src/run_pipeline.py --config task1_llm/shibin_thomas/configs/gpt_char_v1.yaml

    # GPU session ended mid-run? continue from the last completed epoch
    python task1_llm/shibin_thomas/src/run_pipeline.py --config ... --resume latest
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from data_prep import prepare_data
from evaluate import evaluate
from train import train
from utils import SRC_DIR, load_config


def main() -> None:
    p = argparse.ArgumentParser(description="Task 1 full pipeline")
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. training.batch_size=32")
    p.add_argument("--resume", default=None, help="run_id or 'latest' to continue training")
    p.add_argument("--skip-train", default=None, metavar="RUN_ID", help="only evaluate an existing run")
    a = p.parse_args()

    cfg = load_config(a.config, a.set)
    prepare_data(cfg)
    run_id = a.skip_train or train(cfg, a.resume)
    evaluate(cfg, run_id)
    subprocess.run([sys.executable, str(SRC_DIR / "analyze_failures.py"), "--config", a.config,
                    "--run-id", run_id, "--set", *a.set], check=True)
    print(f"\nPipeline finished for run {run_id}")


if __name__ == "__main__":
    main()
