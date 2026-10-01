"""Download the raw TinyStories dataset into task1_llm/data/raw/ (shared by all members).

The Hugging Face `train` split of roneneldan/TinyStories (~2.1M stories) is streamed
and written as JSON Lines, one story per line: {"text": "..."}. The line order is
the order of the HF split, so every member's split (stored as line indices in their
own data_processed/ folder) can be rebuilt exactly from this file.

Usage (from the repo root):
    python task1_llm/data/download_tinystories.py                      # full train split
    python task1_llm/data/download_tinystories.py --max-stories 3000   # small file for smoke tests

The raw .jsonl files are git-ignored (too large for GitHub); a small sidecar
*.meta.json with the source, split, row count and SHA-256 is written next to each one
and IS committed so the exact raw file can be verified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
REPO_ROOT = DATA_DIR.parent.parent
DATASET_NAME = "roneneldan/TinyStories"


def default_output(split: str, max_stories: int | None) -> Path:
    suffix = f"_first{max_stories}" if max_stories else ""
    return DATA_DIR / "raw" / f"tinystories_{split}{suffix}.jsonl"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download(split: str = "train", max_stories: int | None = None, out: Path | None = None) -> Path:
    from datasets import load_dataset  # imported lazily so the rest of the repo works without it

    out = Path(out) if out else default_output(split, max_stories)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".jsonl.partial")

    print(f"Streaming {DATASET_NAME} [{split}] -> {out.resolve().relative_to(REPO_ROOT).as_posix()}")
    ds = load_dataset(DATASET_NAME, split=split, streaming=True)
    t0, n = time.time(), 0
    with open(tmp, "w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps({"text": row["text"]}, ensure_ascii=False) + "\n")
            n += 1
            if n % 100_000 == 0:
                print(f"  {n:,} stories ({time.time() - t0:.0f}s)")
            if max_stories and n >= max_stories:
                break
    tmp.replace(out)

    meta = {
        "dataset": DATASET_NAME,
        "split": split,
        "max_stories": max_stories,
        "num_rows": n,
        "sha256": sha256_file(out),
        "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "format": 'JSON Lines, one {"text": story} object per line, original HF order',
    }
    with open(out.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Done: {n:,} stories in {time.time() - t0:.0f}s, sha256={meta['sha256'][:12]}...")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="train", choices=["train", "validation"])
    p.add_argument("--max-stories", type=int, default=None, help="stop after N stories (smoke tests)")
    p.add_argument("--out", type=Path, default=None, help="output path (default: task1_llm/data/raw/...)")
    args = p.parse_args()
    download(args.split, args.max_stories, args.out)


if __name__ == "__main__":
    main()
