"""One command for the whole Task 2 pipeline:
preprocess + EDA -> train baseline, experiment 1, experiment 2 -> evaluate + compare -> 20-error review.

    # smoke test (minutes on CPU; downloads 4,000 train + 4,000 test reviews if needed)
    python task2_sentiment/shibin_thomas/src/run_pipeline.py --smoke

    # full run (GPU): all three models on the full dataset
    python task2_sentiment/shibin_thomas/src/run_pipeline.py

    # session ended mid-run? models that already finished are skipped
    python task2_sentiment/shibin_thomas/src/run_pipeline.py --resume latest
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime

from data_prep import prepare_data
from evaluate import evaluate_run
from train import load_model_config, train_model
from utils import (SRC_DIR, config_hash, environment_info, get_device, git_info, hardware_info, latest_run_id,
                   rel, repo_path, run_dirs, sha256_file, update_manifest)

CFG_DIR = "task2_sentiment/shibin_thomas/configs"
DEFAULT_MODELS = [f"{CFG_DIR}/baseline_meanpool.yaml", f"{CFG_DIR}/exp1_textcnn.yaml",
                  f"{CFG_DIR}/exp2_bigru_attn.yaml"]


def run_pipeline(models=None, overrides=None, smoke=False, resume=None, skip_train=None) -> str:
    """Preprocess -> train every model -> evaluate -> 20-error review. Returns the run id."""
    models = list(models or DEFAULT_MODELS)
    overrides = list(overrides or [])
    if smoke:
        overrides = [f"data_config={CFG_DIR}/data_smoke.yaml", "training.epochs=1", "smoke=true"] + overrides
    cfgs = [load_model_config(p, overrides) for p in models]
    if len({c["data_config"] for c in cfgs}) != 1:
        raise ValueError("all models of one run must share the same data config")
    prefix = "smoke_yelp3" if smoke else "yelp3"

    processed = prepare_data(cfgs[0]["data"])
    if skip_train:
        run_id = skip_train
    elif resume:
        run_id = latest_run_id(prefix) if resume == "latest" else resume
    else:
        run_id = f"{prefix}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    dirs = run_dirs(run_id)
    device = get_device(cfgs[0].get("device", "auto"))
    if not skip_train and not resume:
        update_manifest(run_id, git=git_info(), hardware=hardware_info(device), environment=environment_info(),
                        data={"processed_dir": rel(processed), "meta": rel(processed / "meta.json")},
                        models={c["name"]: {"config_file": c["_config_path"], "config_hash": config_hash(c),
                                            "role": c.get("role"), "overrides": c["_overrides"]} for c in cfgs},
                        raw_logs=rel(dirs["raw_logs"]), outputs=rel(dirs["outputs"]),
                        checkpoints=rel(dirs["checkpoints"]))

    if not skip_train:
        for cfg in cfgs:
            if (dirs["outputs"] / cfg["name"] / "train_summary.json").exists():
                print(f"[pipeline] {cfg['name']} already finished in {run_id} - skipping")
                continue
            train_model(cfg, run_id, processed)
        ckpts = {c["name"]: {"path": rel(p), "sha256": sha256_file(p)}
                 for c in cfgs for p in [dirs["checkpoints"] / c["name"] / "best_model.pt"]}
        update_manifest(run_id, checkpoints_files=ckpts,
                        train_summaries={c["name"]: rel(dirs["outputs"] / c["name"] / "train_summary.json")
                                         for c in cfgs})

    evaluate_run(cfgs, run_id, processed)
    cmd = [sys.executable, str(SRC_DIR / "error_analysis.py"), "--run-id", run_id]
    subprocess.run(cmd + (["--smoke"] if smoke else []), check=True)
    print(f"\nPipeline finished for run {run_id}")
    return run_id


def main() -> None:
    ap = argparse.ArgumentParser(description="Task 2 full pipeline")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS, help="model config files (first = baseline)")
    ap.add_argument("--set", nargs="*", default=[], help="overrides for every model config, e.g. training.epochs=3")
    ap.add_argument("--smoke", action="store_true", help="tiny data + 1 epoch: pipeline check only")
    ap.add_argument("--resume", default=None, help="run id or 'latest': skip models that already finished")
    ap.add_argument("--skip-train", default=None, metavar="RUN_ID", help="only evaluate an existing run")
    a = ap.parse_args()
    run_pipeline(a.models, a.set, a.smoke, a.resume, a.skip_train)


if __name__ == "__main__":
    main()
