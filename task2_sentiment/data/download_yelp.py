"""Download the raw Yelp Review Polarity dataset into task2_sentiment/data/raw/ (shared by all members).

Source: Hugging Face `fancyzhx/yelp_polarity` (Zhang et al., 2015): 560,000 train and 38,000 test reviews,
label 0 = negative (1-2 stars), 1 = positive (4-5 stars). Each split is written as JSON Lines,
one {"text": ..., "label": 0|1} object per line, in the original order.

Usage (from the repo root):
    python task2_sentiment/data/download_yelp.py                    # full train + test
    python task2_sentiment/data/download_yelp.py --max-rows 4000    # small shuffled files for smoke tests

Fallback without Hugging Face: download yelp_review_polarity_csv.tgz (train.csv / test.csv with
label 1|2 and no header, e.g. from the fast.ai datasets mirror), extract it into data/raw/ and point
the data config at the .csv files -- the preprocessing reads that format too.

Raw files are git-ignored; a small *.meta.json sidecar (rows, label counts, SHA-256) is committed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
REPO_ROOT = DATA_DIR.parent.parent
DATASET_NAMES = ["fancyzhx/yelp_polarity", "yelp_polarity"]


def default_output(split: str, max_rows: int | None) -> Path:
    suffix = f"_first{max_rows}" if max_rows else ""
    return DATA_DIR / "raw" / f"yelp_polarity_{split}{suffix}.jsonl"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _load_stream(split: str):
    from datasets import load_dataset  # lazy import

    last_err = None
    for name in DATASET_NAMES:
        try:
            return name, load_dataset(name, split=split, streaming=True)
        except Exception as e:  # try the next mirror name
            last_err = e
    raise RuntimeError(f"could not load Yelp polarity from Hugging Face: {last_err}")


def download(split: str = "train", max_rows: int | None = None, out: Path | None = None) -> Path:
    out = Path(out) if out else default_output(split, max_rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".jsonl.partial")
    name, ds = _load_stream(split)
    if max_rows:                                   # smoke files: shuffle so both classes appear
        ds = ds.shuffle(seed=0, buffer_size=50_000)
    print(f"Streaming {name} [{split}] -> {out.resolve().relative_to(REPO_ROOT).as_posix()}")
    t0, n, labels = time.time(), 0, Counter()
    with open(tmp, "w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps({"text": row["text"], "label": int(row["label"])}, ensure_ascii=False) + "\n")
            labels[int(row["label"])] += 1
            n += 1
            if n % 100_000 == 0:
                print(f"  {n:,} rows ({time.time() - t0:.0f}s)")
            if max_rows and n >= max_rows:
                break
    tmp.replace(out)
    meta = {"dataset": name, "split": split, "max_rows": max_rows, "num_rows": n,
            "label_counts": {str(k): v for k, v in sorted(labels.items())}, "sha256": sha256_file(out),
            "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "format": 'JSON Lines, one {"text": str, "label": 0|1} per line; 0 = negative, 1 = positive'}
    with open(out.with_suffix(".meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Done: {n:,} rows, labels {dict(labels)}, {time.time() - t0:.0f}s")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=["train", "test", "both"], default="both")
    p.add_argument("--max-rows", type=int, default=None, help="stop after N rows (smoke tests)")
    a = p.parse_args()
    for split in (["train", "test"] if a.split == "both" else [a.split]):
        download(split, a.max_rows)


if __name__ == "__main__":
    main()
