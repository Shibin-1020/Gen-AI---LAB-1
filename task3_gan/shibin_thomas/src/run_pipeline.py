"""One command for Task 3: train CycleGAN -> translate (pred_A2B / pred_B2A) -> evaluate + submission.csv
-> build the blinded human-audit pack.

    python task3_gan/shibin_thomas/src/run_pipeline.py --config task3_gan/shibin_thomas/configs/smoke.yaml
    python task3_gan/shibin_thomas/src/run_pipeline.py                       # full run (cyclegan_v1.yaml)
    python task3_gan/shibin_thomas/src/run_pipeline.py --resume latest       # continue after an interruption
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from train import train
from translate import export
from utils import MEMBER_DIR, load_config, repo_path

sys.path.insert(0, str(MEMBER_DIR))
from evaluate_local import evaluate  # noqa: E402

DEFAULT_CONFIG = "task3_gan/shibin_thomas/configs/cyclegan_v1.yaml"


def run_pipeline(config: str = DEFAULT_CONFIG, overrides=None, resume=None, skip_train=None) -> str:
    cfg = load_config(config, overrides or [])
    run_id = skip_train or train(cfg, resume)
    export(cfg, run_id)
    evaluate(cfg, run_id)
    from human_audit import make
    make(pred_root=repo_path(cfg["export"]["out_dir"]), monet_dir=cfg["data"]["monet_dir"],
         photo_dir=cfg["data"]["photo_dir"])
    print(f"\nPipeline finished for run {run_id}")
    return run_id


def main() -> None:
    p = argparse.ArgumentParser(description="Task 3 full pipeline")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. training.batch_size=2")
    p.add_argument("--resume", default=None, help="run id or 'latest'")
    p.add_argument("--skip-train", default=None, metavar="RUN_ID", help="only translate + evaluate an existing run")
    a = p.parse_args()
    run_pipeline(a.config, a.set, a.resume, a.skip_train)


if __name__ == "__main__":
    main()
