"""Task 1.1 -- Data preprocessing (character level).

Steps
  1. Load the shared raw TinyStories file (task1_llm/data/raw/*.jsonl).
  2. Clean each story: normalise typographic unicode (curly quotes, dashes, ellipsis)
     to ASCII, normalise whitespace, and keep only stories made of printable ASCII
     + newline. Stories shorter than `min_story_chars` are dropped.
  3. Draw MY OWN random split from the cleaned pool with my own seed:
     100,000 training stories and 10,000 validation stories (disjoint).
  4. Build `char_to_idx` / `idx_to_char` from the TRAINING split only.
     Special ids: 0 = <eos> (end of story), 1 = <unk> (char unseen in training).
  5. Encode every story to integers, append <eos>, and concatenate into one long
     stream per split (uint8, since the vocabulary is < 256) -> train.bin / val.bin.
  6. Fixed-length input/target sequences for autoregressive LM are cut from the stream
     by `CharStream`: x = s[i : i+T], y = s[i+1 : i+T+1] (target = input shifted by one).

Run:  python task1_llm/shibin_thomas/src/data_prep.py --config task1_llm/shibin_thomas/configs/gpt_char_v1.yaml
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from utils import config_hash, load_config, read_json, rel, repo_path, sha256_file, write_json

EOS_TOKEN, UNK_TOKEN = "<eos>", "<unk>"
EOS_ID, UNK_ID = 0, 1

UNICODE_TO_ASCII = {
    "‘": "'", "’": "'", "‚": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "″": '"',
    "–": "-", "—": "-", "‒": "-", "−": "-",
    "…": "...", " ": " ", "​": "", "\t": " ", "\r": "",
}
_TRANS = str.maketrans(UNICODE_TO_ASCII)
ALLOWED_CHARS = {chr(i) for i in range(32, 127)} | {"\n"}
_MULTI_NL = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r"[ ]{2,}")


# --------------------------------------------------------------------------- cleaning
def clean_story(text: str, min_chars: int) -> tuple[str | None, str]:
    """Return (cleaned_text, status). cleaned_text is None when the story is rejected."""
    if not isinstance(text, str):
        return None, "malformed"
    t = text.translate(_TRANS)
    t = _MULTI_SPACE.sub(" ", t)
    t = "\n".join(line.strip() for line in t.split("\n"))
    t = _MULTI_NL.sub("\n\n", t).strip()
    if len(t) < min_chars:
        return None, "too_short"
    if not set(t) <= ALLOWED_CHARS:
        return None, "non_ascii"
    return t, "ok"


def iter_raw(path: Path):
    """Yield (story_index, text). Supports the shared JSONL file ({"text": ...} per line) and, as a
    fallback, the official TinyStories .txt release where stories are separated by <|endoftext|>."""
    if path.suffix == ".txt":
        yield from _iter_txt(path)
        return
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            try:
                yield line_no, json.loads(line).get("text")
            except (json.JSONDecodeError, AttributeError):
                yield line_no, None


def _iter_txt(path: Path, sep: str = "<|endoftext|>", chunk: int = 1 << 24):
    idx, buf = 0, ""
    with open(path, encoding="utf-8", errors="replace") as f:
        while block := f.read(chunk):
            buf += block
            *stories, buf = buf.split(sep)
            for s in stories:
                if s.strip():
                    yield idx, s.strip()
                    idx += 1
    if buf.strip():
        yield idx, buf.strip()


# --------------------------------------------------------------------------- vocab
def build_vocab(texts: list[str]) -> tuple[dict[str, int], dict[int, str], Counter]:
    counts = Counter()
    for t in texts:
        counts.update(t)
    char_to_idx = {EOS_TOKEN: EOS_ID, UNK_TOKEN: UNK_ID}
    for i, ch in enumerate(sorted(counts), start=2):
        char_to_idx[ch] = i
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    return char_to_idx, idx_to_char, counts


def encode(text: str, char_to_idx: dict[str, int]) -> list[int]:
    return [char_to_idx.get(ch, UNK_ID) for ch in text]


def decode(ids, idx_to_char: dict[int, str], stop_at_eos: bool = False) -> str:
    out = []
    for i in ids:
        i = int(i)
        if i == EOS_ID:
            if stop_at_eos:
                break
            out.append("\n<eos>\n")
        else:
            out.append(idx_to_char.get(i, "?") if i != UNK_ID else "�")
    return "".join(out)


def load_vocab(processed_dir: Path) -> tuple[dict[str, int], dict[int, str]]:
    v = read_json(processed_dir / "vocab.json")
    return v["char_to_idx"], {int(k): c for k, c in v["idx_to_char"].items()}


def encode_split(texts: list[str], char_to_idx: dict[str, int]) -> tuple[np.ndarray, int]:
    ids: list[int] = []
    for t in texts:
        ids.extend(encode(t, char_to_idx))
        ids.append(EOS_ID)
    arr = np.asarray(ids, dtype=np.uint8 if len(char_to_idx) < 256 else np.uint16)
    return arr, int((arr == UNK_ID).sum())


# --------------------------------------------------------------------------- main step
def _ensure_raw(cfg: dict) -> Path:
    raw = repo_path(cfg["data"]["raw_path"])
    if raw.exists():
        return raw
    if not cfg["data"].get("download_if_missing", False):
        raise FileNotFoundError(f"{rel(raw)} not found; run task1_llm/data/download_tinystories.py first")
    import importlib.util
    spec = importlib.util.spec_from_file_location("dl", repo_path("task1_llm/data/download_tinystories.py"))
    dl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl)
    return dl.download("train", cfg["data"].get("download_max_stories"), raw)


def prepare_data(cfg: dict, force: bool = False, log=print) -> Path:
    d = cfg["data"]
    out_dir = repo_path(d["processed_dir"])
    data_key = {k: d[k] for k in ("raw_path", "n_train_stories", "n_val_stories", "min_story_chars", "split_seed")}
    meta_path = out_dir / "meta.json"
    if not force and meta_path.exists() and (out_dir / "train.bin").exists():
        meta = read_json(meta_path)
        if meta.get("data_config") == data_key:
            log(f"[data] reusing processed data in {rel(out_dir)} (same data config)")
            return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = _ensure_raw(cfg)

    # Pass 1: clean everything, remember which raw lines are usable.
    log(f"[data] pass 1: cleaning {rel(raw)}")
    status = Counter()
    valid_lines: list[int] = []
    for line_no, text in iter_raw(raw):
        _, s = clean_story(text, d["min_story_chars"])
        status[s] += 1
        if s == "ok":
            valid_lines.append(line_no)
    n_total = sum(status.values())
    log(f"[data] raw stories={n_total:,} ok={status['ok']:,} non_ascii={status['non_ascii']:,} "
        f"too_short={status['too_short']:,} malformed={status['malformed']:,}")

    n_tr, n_va = d["n_train_stories"], d["n_val_stories"]
    if n_tr + n_va > len(valid_lines):
        raise ValueError(f"need {n_tr + n_va} clean stories but only {len(valid_lines)} available")

    # My own random split (seeded), disjoint by construction.
    rng = np.random.default_rng(d["split_seed"])
    chosen = rng.choice(np.asarray(valid_lines, dtype=np.int64), size=n_tr + n_va, replace=False)
    train_lines, val_lines = chosen[:n_tr], chosen[n_tr:]
    assert not set(train_lines.tolist()) & set(val_lines.tolist())

    # Pass 2: gather the chosen stories (in split order).
    pos = {int(l): i for i, l in enumerate(chosen)}
    texts: list[str | None] = [None] * len(chosen)
    for line_no, text in iter_raw(raw):
        if line_no in pos:
            texts[pos[line_no]] = clean_story(text, d["min_story_chars"])[0]
    train_texts, val_texts = texts[:n_tr], texts[n_tr:]

    char_to_idx, idx_to_char, char_counts = build_vocab(train_texts)
    train_ids, train_unk = encode_split(train_texts, char_to_idx)
    val_ids, val_unk = encode_split(val_texts, char_to_idx)
    train_ids.tofile(out_dir / "train.bin")
    val_ids.tofile(out_dir / "val.bin")
    np.savez_compressed(out_dir / "split_indices.npz", train_lines=train_lines, val_lines=val_lines)
    write_json(out_dir / "vocab.json", {
        "char_to_idx": char_to_idx,
        "idx_to_char": {str(i): c for i, c in idx_to_char.items()},
        "special_tokens": {EOS_TOKEN: EOS_ID, UNK_TOKEN: UNK_ID},
        "vocab_size": len(char_to_idx),
        "train_char_counts": dict(char_counts.most_common()),
    })

    lens_tr = np.array([len(t) for t in train_texts])
    lens_va = np.array([len(t) for t in val_texts])
    T, B = cfg["model"]["block_size"], cfg["training"]["batch_size"] * cfg["training"].get("grad_accum_steps", 1)
    meta = {
        "data_config": data_key,
        "raw_file": rel(raw),
        "raw_sha256": sha256_file(raw),
        "raw_status_counts": dict(status),
        "vocab_size": len(char_to_idx),
        "dtype": str(train_ids.dtype),
        "train": {"stories": n_tr, "tokens": int(train_ids.size), "unk_tokens": train_unk,
                  "story_chars_mean": float(lens_tr.mean()), "story_chars_median": float(np.median(lens_tr)),
                  "story_chars_p95": float(np.percentile(lens_tr, 95)), "story_chars_max": int(lens_tr.max())},
        "val": {"stories": n_va, "tokens": int(val_ids.size), "unk_tokens": val_unk,
                "story_chars_mean": float(lens_va.mean()), "story_chars_median": float(np.median(lens_va))},
        "sequences_per_epoch_estimate": int((train_ids.size - 1) // T),
        "optimizer_steps_per_epoch_estimate": int((train_ids.size - 1) // T // B),
        "config_hash": config_hash(cfg),
    }
    write_json(meta_path, meta)

    # Human-readable example of the fixed-length (input, target) pairs.
    with open(out_dir / "example_sequences.txt", "w", encoding="utf-8") as f:
        f.write(f"block_size T={T}; target = input shifted by one character\n\n")
        for k in range(3):
            i = k * T
            x, y = train_ids[i:i + T], train_ids[i + 1:i + T + 1]
            f.write(f"--- sequence {k} ---\nx ids[:20] = {x[:20].tolist()}\ny ids[:20] = {y[:20].tolist()}\n")
            f.write(f"x text: {decode(x, idx_to_char)!r}\ny text: {decode(y, idx_to_char)!r}\n\n")

    log(f"[data] vocab={len(char_to_idx)} train_tokens={train_ids.size:,} val_tokens={val_ids.size:,} "
        f"val_unk={val_unk} steps/epoch~{meta['optimizer_steps_per_epoch_estimate']:,} -> {rel(out_dir)}")
    return out_dir


# --------------------------------------------------------------------------- sequences
class CharStream:
    """A split's token stream kept on the device; yields fixed-length (x, y) batches.

    Epoch definition: the stream is cut into non-overlapping windows of length T,
    starting at a random offset in [0, T) that changes every epoch, and the windows are
    shuffled. Every character is therefore a prediction target exactly once per epoch.
    """

    def __init__(self, path: Path, block_size: int, device: torch.device, dtype=np.uint8):
        arr = np.fromfile(path, dtype=dtype)
        self.data = torch.from_numpy(arr.astype(np.int64) if dtype != np.uint8 else arr).to(device)
        self.T = block_size
        self.device = device
        self._offsets = torch.arange(block_size + 1, device=device)

    def __len__(self) -> int:
        return self.data.numel()

    def num_sequences(self, offset: int = 0) -> int:
        return (self.data.numel() - 1 - offset) // self.T

    def _gather(self, starts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[starts[:, None] + self._offsets].long()
        return chunk[:, :-1], chunk[:, 1:]

    def epoch_starts(self, gen: torch.Generator) -> torch.Tensor:
        offset = int(torch.randint(0, self.T, (1,), generator=gen))
        starts = offset + self.T * torch.arange(self.num_sequences(offset))
        return starts[torch.randperm(starts.numel(), generator=gen)]

    def batches(self, starts: torch.Tensor, batch_size: int, drop_last: bool = True):
        n = starts.numel()
        stop = n - n % batch_size if drop_last else n
        for i in range(0, stop, batch_size):
            yield self._gather(starts[i:i + batch_size].to(self.device))

    def eval_starts(self, max_seqs: int | None = None, seed: int = 0) -> torch.Tensor:
        starts = self.T * torch.arange(self.num_sequences())
        if max_seqs is not None and max_seqs < starts.numel():
            g = torch.Generator().manual_seed(seed)
            starts = starts[torch.randperm(starts.numel(), generator=g)[:max_seqs]].sort().values
        return starts


def main() -> None:
    p = argparse.ArgumentParser(description="Task 1 data preprocessing")
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. data.n_train_stories=1000")
    p.add_argument("--force", action="store_true", help="rebuild even if processed data exists")
    a = p.parse_args()
    prepare_data(load_config(a.config, a.set), force=a.force)


if __name__ == "__main__":
    main()
