"""Windows/VS Code entry point for the complete Task 3 pipeline.

Run from any directory:
    python task3_gan/member_denisha/run_task3.py

The script always runs the smoke test first. Full training starts only when
the smoke test passes. It resumes from the latest checkpoint when available.
The approved course images can be placed in task3_gan/data/, or supplied as
the course ZIP with --dataset-zip.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile


MEMBER_DIR = Path(__file__).resolve().parent
REPO_ROOT = MEMBER_DIR.parents[1]
DATA_ROOT = MEMBER_DIR.parent / "data"
CHECKPOINT_DIR = MEMBER_DIR / "checkpoints"
LOG_DIR = REPO_ROOT / "reproducibility" / "raw_logs"
LOG_PATH = LOG_DIR / "task3_denisha.log"
CONFIG = MEMBER_DIR / "config.yaml"


def run(command: list[str], *, cwd: Path) -> None:
    print("\n$", " ".join(str(x) for x in command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def extract_dataset(zip_path: Path) -> None:
    """Extract only the two approved domains from the supplied course ZIP."""
    if not zip_path.is_file():
        raise SystemExit(f"Dataset ZIP not found: {zip_path}")
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    wanted = {"monet_jpg", "photo_jpg"}
    extracted = 0
    with ZipFile(zip_path) as archive:
        for info in archive.infolist():
            parts = Path(info.filename).parts
            domain_index = next((i for i, part in enumerate(parts) if part in wanted), None)
            if domain_index is None or info.is_dir():
                continue
            relative = Path(*parts[domain_index:])
            if relative.name.startswith("._") or relative.name == ".DS_Store" or relative.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            destination = (DATA_ROOT / relative).resolve()
            if DATA_ROOT.resolve() not in destination.parents:
                raise SystemExit(f"Unsafe dataset path in ZIP: {info.filename}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            extracted += 1
    if extracted == 0:
        raise SystemExit("The ZIP did not contain monet_jpg/photo_jpg image files.")
    print(f"Extracted {extracted} approved images into {DATA_ROOT}")


def verify_data() -> None:
    expected = {
        "monet_jpg": DATA_ROOT / "monet_jpg",
        "photo_jpg": DATA_ROOT / "photo_jpg",
    }
    missing = []
    for name, folder in expected.items():
        images = [p for p in folder.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}] if folder.exists() else []
        print(f"{name}: {len(images)} images — {folder}")
        if not images:
            missing.append(str(folder))
    if missing:
        raise SystemExit(
            "Approved image data is missing. Add the course-provided Kaggle data to:\n"
            f"  {DATA_ROOT / 'monet_jpg'}\n"
            f"  {DATA_ROOT / 'photo_jpg'}\n"
            "or rerun with --dataset-zip /path/to/dataset.zip.\n"
            "Do not use manually edited or externally sourced generated images."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--dataset-zip", type=Path, help="Course-provided ZIP containing monet_jpg/ and photo_jpg/")
    parser.add_argument("--checkpoint-dir", type=Path, default=CHECKPOINT_DIR)
    parser.add_argument("--require-gpu", action="store_true", help="stop instead of silently falling back to CPU")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="refuse to resume and require an empty checkpoint folder")
    args = parser.parse_args()
    if args.config.is_absolute():
        config = args.config
    else:
        # Commands are normally launched with cwd=MEMBER_DIR.  Prefer that
        # location, while retaining compatibility with repo-root invocations.
        config = MEMBER_DIR / args.config
        if not config.exists():
            config = REPO_ROOT / args.config
    if not config.is_file():
        raise SystemExit(f"Config file not found: {config}")
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir.is_absolute() else REPO_ROOT / args.checkpoint_dir
    if args.dataset_zip:
        extract_dataset(args.dataset_zip.expanduser().resolve())
    verify_data()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if args.fresh and any(checkpoint_dir.glob("*.pt")):
        raise SystemExit(f"Checkpoint folder is not empty: {checkpoint_dir}. Use another folder or remove it deliberately.")
    run([sys.executable, "src/smoke_test.py", "--config", str(config)], cwd=MEMBER_DIR)
    if args.smoke_only:
        print("Smoke test passed; full training was intentionally skipped.")
        return
    if args.require_gpu:
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("CUDA GPU is unavailable; full training was refused. Select the correct NVIDIA GPU/interpreter first.")
        print(f"CUDA GPU confirmed: {torch.cuda.get_device_name(0)}", flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    train_command = [
        sys.executable, "-u", "src/train.py", "--config", str(config),
        "--checkpoint-dir", str(checkpoint_dir), "--resume-latest",
    ]
    print(f"\nTraining log: {LOG_PATH}")
    with LOG_PATH.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(train_command, cwd=MEMBER_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = process.wait()
    if code != 0:
        raise SystemExit(f"Training failed with exit code {code}; evaluation was not started.")
    run([sys.executable, "src/plot_history.py"], cwd=MEMBER_DIR)
    run([sys.executable, "src/evaluate_metrics.py", "--real", "../data/monet_jpg", "--fake", "outputs/pred_B2A", "--out", "outputs/metrics_B2A.json"], cwd=MEMBER_DIR)
    run([sys.executable, "src/evaluate_metrics.py", "--real", "../data/photo_jpg", "--fake", "outputs/pred_A2B", "--out", "outputs/metrics_A2B.json"], cwd=MEMBER_DIR)
    run([sys.executable, "src/evaluate_submission.py", "--real-monet", "../data/monet_jpg", "--real-photo", "../data/photo_jpg", "--gen-a2b", "outputs/pred_A2B", "--gen-b2a", "outputs/pred_B2A", "--out", "outputs/submission.csv", "--json-out", "outputs/submission_metrics.json"], cwd=MEMBER_DIR)
    run([sys.executable, "src/evaluate_lpips.py", "--real", "../data/monet_jpg", "--recon", "outputs/recon_A", "--out", "outputs/lpips_A.json"], cwd=MEMBER_DIR)
    run([sys.executable, "src/evaluate_lpips.py", "--real", "../data/photo_jpg", "--recon", "outputs/recon_B", "--out", "outputs/lpips_B.json"], cwd=MEMBER_DIR)
    print("\nTRAINING AND AUTOMATED EVALUATION COMPLETE.")
    print("Next required step: two human raters must fill outputs/human_audit.csv before audit_agreement.py and final validation.")


if __name__ == "__main__":
    main()
