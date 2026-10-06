"""One-command Task 1 runner for VSCode, Jupyter, and Colab terminals.

The smoke test always runs first. Full training resumes from the newest local
checkpoint and then produces the report, manifest, and validation artifacts.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


MEMBER_DIR = Path(__file__).resolve().parent
REPO_ROOT = MEMBER_DIR.parents[1]
DATA_PATH = REPO_ROOT / "task1_llm" / "data" / "TinyStories-train.jsonl"
CHECKPOINT_DIR = MEMBER_DIR / "checkpoints"
LOG_DIR = REPO_ROOT / "reproducibility" / "raw_logs"
LOG_PATH = LOG_DIR / "task1_denisha.log"


def run(command: list[str], cwd: Path) -> None:
    print("\n$", " ".join(str(x) for x in command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="refuse to resume when checkpoints already exist")
    args = parser.parse_args()
    run([sys.executable, "src/smoke_test.py"], MEMBER_DIR)
    if args.smoke_only:
        print("Smoke test passed; full training was intentionally skipped.")
        return
    if not DATA_PATH.exists():
        print(f"Missing {DATA_PATH}; downloading the approved TinyStories split.")
        run([sys.executable, "reproducibility/prepare_data.py", "--task", "task1", "--out", str(REPO_ROOT)], REPO_ROOT)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if args.fresh and any(CHECKPOINT_DIR.glob("*.pt")):
        raise SystemExit(f"Checkpoint directory is not empty: {CHECKPOINT_DIR}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", "src/train.py", "--config", "config.yaml", "--checkpoint-dir", str(CHECKPOINT_DIR), "--resume-latest"]
    with LOG_PATH.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=MEMBER_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = process.wait()
    if code != 0:
        raise SystemExit(f"Training failed with exit code {code}; post-run steps were not started.")
    run([sys.executable, "src/plot_history.py"], MEMBER_DIR)
    run([sys.executable, "src/write_report.py"], MEMBER_DIR)
    manifest_dir = REPO_ROOT / "reproducibility" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "reproducibility/collect_manifest.py", "--task", "task1", "--command", "python src/run_task1.py", "--dataset", "roneneldan/TinyStories train split", "--checkpoint", str(CHECKPOINT_DIR / "gpt_from_scratch.pt"), "--out", str(manifest_dir / "task1_denisha.json")], REPO_ROOT)
    run([sys.executable, "reproducibility/validate_results.py", "--task", "task1", "--root", str(REPO_ROOT)], REPO_ROOT)
    print("Task 1 training, reporting, manifest, and validation completed.")


if __name__ == "__main__":
    main()
