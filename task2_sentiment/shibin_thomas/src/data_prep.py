"""Task 2.1 -- Data analysis and preprocessing for Yelp polarity.

1. EDA: class distribution / balance and review-length distribution (raw words) per class, per split.
2. Missing / malformed entries: null or non-string text, invalid labels, reviews that are empty after
   normalisation, exact duplicates inside train, and train reviews that also appear in test (leakage)
   are counted and removed (test is never modified except for malformed rows).
3. Text preprocessing (each step is a config switch, all on by default):
     normalise    HTML entities, the literal "\\n" / '\\"' sequences of the Yelp dump, URLs
     lowercase
     contractions "didn't" -> "did not", "they're" -> "they are" (so negations survive step 4)
     punctuation  every character except a-z, 0-9 and whitespace removed (digits kept: "5 stars")
     stopwords    scikit-learn English list MINUS negations and contrast words (not, no, never,
                  nothing, without, but, however, ...): removing "not" would turn "not good" into
                  "good" and flip the sentiment
     stemming     NLTK Snowball (Porter2) stemmer: "loved", "loving", "loves" -> "love"
4. Tokenisation: whitespace split of the processed text; vocabulary built on MY training split only
   (min frequency, max size); 0 = <pad>, 1 = <unk>.
5. Encoding: fixed length `max_len` token ids; long reviews keep the first `head_tokens` and the last
   (max_len - head_tokens) tokens ("head+tail"), because reviews often state the verdict at the end.
   The embeddings themselves are learned from scratch by each model (nn.Embedding, random init).

Split: official TEST split (38K) = test set for every member; my own stratified validation split
(`val_size`, seed `split_seed`) is carved out of the official TRAIN split.

Run:  python task2_sentiment/shibin_thomas/src/data_prep.py --data-config task2_sentiment/shibin_thomas/configs/data.yaml
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

from utils import MEMBER_DIR, load_config, read_json, rel, repo_path, sha256_file, write_json

PAD_ID, UNK_ID = 0, 1
PAD, UNK = "<pad>", "<unk>"

KEEP_WORDS = {  # removed from the stopword list on purpose: they carry sentiment / polarity shifts
    "no", "not", "nor", "never", "nothing", "nobody", "none", "neither", "cannot", "without", "against",
    "but", "however", "although", "though", "yet", "except", "very", "too", "few", "less", "least",
    "more", "most", "much", "only", "again", "off", "down", "up", "over", "well",
}
CONTRACTIONS = [
    (re.compile(r"\bwon't\b"), "will not"), (re.compile(r"\bcan't\b"), "can not"),
    (re.compile(r"\bain't\b"), "is not"), (re.compile(r"\bshan't\b"), "shall not"),
    (re.compile(r"n't\b"), " not"), (re.compile(r"'re\b"), " are"), (re.compile(r"'ll\b"), " will"),
    (re.compile(r"'ve\b"), " have"), (re.compile(r"'m\b"), " am"), (re.compile(r"'d\b"), " would"),
    (re.compile(r"'s\b"), ""),
]
URL_RE = re.compile(r"https?://\S+|www\.\S+")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]+")
WS_RE = re.compile(r"\s+")


# --------------------------------------------------------------------------- cleaning
def normalize_raw(text) -> str | None:
    """Undo dump artefacts; returns None for missing / non-string entries."""
    if not isinstance(text, str):
        return None
    t = text.replace("\\n", " ").replace('\\"', '"').replace("\r", " ").replace("\n", " ")
    t = html.unescape(t)
    return WS_RE.sub(" ", t).strip()


class Preprocessor:
    def __init__(self, cfg: dict):
        self.lowercase = cfg.get("lowercase", True)
        self.contractions = cfg.get("expand_contractions", True)
        self.remove_punct = cfg.get("remove_punctuation", True)
        self.remove_stop = cfg.get("remove_stopwords", True)
        self.stem = cfg.get("stemming", "snowball")
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        self.stopwords = set(ENGLISH_STOP_WORDS) - KEEP_WORDS if cfg.get("keep_negations", True) \
            else set(ENGLISH_STOP_WORDS)
        self._stem_cache: dict[str, str] = {}
        if self.stem == "snowball":
            from nltk.stem.snowball import SnowballStemmer
            self._stemmer = SnowballStemmer("english")
        elif self.stem in (None, "none", False):
            self._stemmer = None
        else:
            raise ValueError(f"unknown stemming option {self.stem!r}")

    def _stem_word(self, w: str) -> str:
        s = self._stem_cache.get(w)
        if s is None:
            s = self._stem_cache[w] = self._stemmer.stem(w)
        return s

    def tokens(self, text: str) -> list[str]:
        t = URL_RE.sub(" ", text)
        if self.lowercase:
            t = t.lower()
        if self.contractions:
            t = t.replace("’", "'")
            for rx, rep in CONTRACTIONS:
                t = rx.sub(rep, t)
        if self.remove_punct:
            t = NON_ALNUM_RE.sub(" ", t)
        toks = t.split()
        if self.remove_stop:
            toks = [w for w in toks if w not in self.stopwords]
        if self._stemmer is not None:
            toks = [self._stem_word(w) for w in toks]
        return toks

    def steps(self, text: str) -> dict[str, str]:
        """Intermediate result after every step (for the notebook / report)."""
        out = {"raw": text}
        t = normalize_raw(text) or ""
        out["normalised"] = t
        t = URL_RE.sub(" ", t).lower()
        out["lowercase"] = t
        for rx, rep in CONTRACTIONS:
            t = rx.sub(rep, t.replace("’", "'"))
        out["contractions"] = t
        t = WS_RE.sub(" ", NON_ALNUM_RE.sub(" ", t)).strip()
        out["no_punctuation"] = t
        toks = [w for w in t.split() if w not in self.stopwords]
        out["no_stopwords"] = " ".join(toks)
        out["stemmed_tokens"] = " ".join(self._stem_word(w) for w in toks) if self._stemmer else out["no_stopwords"]
        return out


# --------------------------------------------------------------------------- raw IO
def iter_raw(path: Path):
    """Yield (row_index, text, label) from the shared JSONL file or the classic CSV release
    (label 1 = negative, 2 = positive, no header). Malformed rows yield text/label = None."""
    if path.suffix == ".csv":
        with open(path, encoding="utf-8", newline="") as f:
            for i, row in enumerate(csv.reader(f)):
                try:
                    lab = {"1": 0, "2": 1}.get(row[0].strip())
                    yield i, (row[1] if len(row) > 1 else None), lab
                except IndexError:
                    yield i, None, None
        return
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                obj = json.loads(line)
                lab = obj.get("label")
                yield i, obj.get("text"), (int(lab) if lab in (0, 1, "0", "1") else None)
            except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
                yield i, None, None


def _ensure_raw(cfg: dict, key: str, split: str) -> Path:
    p = repo_path(cfg[key])
    if p.exists():
        return p
    if not cfg.get("download_if_missing", False):
        raise FileNotFoundError(f"{rel(p)} not found; run task2_sentiment/data/download_yelp.py")
    import importlib.util
    spec = importlib.util.spec_from_file_location("dl", repo_path("task2_sentiment/data/download_yelp.py"))
    dl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dl)
    return dl.download(split, cfg.get("download_max_rows"), p)


def load_split(path: Path):
    """Read + validate one raw split. Returns kept rows and a malformed-entry report."""
    rows, report = [], Counter()
    for i, text, label in iter_raw(path):
        report["rows"] += 1
        norm = normalize_raw(text)
        if norm is None:
            report["missing_or_non_string_text"] += 1
            continue
        if label is None:
            report["invalid_label"] += 1
            continue
        if not norm:
            report["empty_after_normalisation"] += 1
            continue
        rows.append((i, norm, label))
    report["kept"] = len(rows)
    return rows, dict(report)


# --------------------------------------------------------------------------- EDA
def eda_stats(rows) -> dict:
    labels = np.array([r[2] for r in rows])
    words = np.array([len(r[1].split()) for r in rows])
    chars = np.array([len(r[1]) for r in rows])
    out = {"n": int(labels.size), "class_counts": {"negative": int((labels == 0).sum()),
                                                   "positive": int((labels == 1).sum())}}
    out["positive_rate"] = float(labels.mean())
    out["balance_ratio_minority_over_majority"] = float(min(labels.mean(), 1 - labels.mean())
                                                        / max(labels.mean(), 1 - labels.mean()))
    for name, arr in (("words", words), ("chars", chars)):
        out[f"length_{name}"] = {k: float(v) for k, v in {
            "mean": arr.mean(), "std": arr.std(), "min": arr.min(), "p5": np.percentile(arr, 5),
            "median": np.median(arr), "p95": np.percentile(arr, 95), "p99": np.percentile(arr, 99),
            "max": arr.max()}.items()}
    for c, name in ((0, "negative"), (1, "positive")):
        w = words[labels == c]
        out[f"length_words_{name}"] = {"mean": float(w.mean()), "median": float(np.median(w)),
                                       "p95": float(np.percentile(w, 95))} if w.size else {}
    return out


def plot_eda(train_rows, test_rows, token_lens, max_len, out_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (name, rows) in zip(axes, (("train", train_rows), ("test", test_rows))):
        lab = np.array([r[2] for r in rows])
        counts = [(lab == 0).sum(), (lab == 1).sum()]
        ax.bar(["negative (0)", "positive (1)"], counts, color=["#c0392b", "#2a8f4a"])
        for i, c in enumerate(counts):
            ax.text(i, c, f"{c:,}\n({c / len(lab):.1%})", ha="center", va="bottom", fontsize=9)
        ax.set(title=f"Class distribution - {name}", ylabel="reviews")
        ax.set_ylim(0, max(counts) * 1.25)
    fig.tight_layout(); fig.savefig(out_dir / "class_distribution.png", dpi=150); plt.close(fig)
    files.append("class_distribution.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    words = np.array([len(r[1].split()) for r in train_rows])
    lab = np.array([r[2] for r in train_rows])
    bins = np.linspace(0, np.percentile(words, 99), 80)
    ax.hist(words[lab == 0], bins=bins, alpha=0.6, color="#c0392b", label="negative")
    ax.hist(words[lab == 1], bins=bins, alpha=0.6, color="#2a8f4a", label="positive")
    ax.set(xlabel="review length (raw words, clipped at p99)", ylabel="reviews",
           title="Review length distribution by class (train)")
    ax.legend(); fig.tight_layout(); fig.savefig(out_dir / "length_distribution.png", dpi=150); plt.close(fig)
    files.append("length_distribution.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(token_lens, bins=np.linspace(0, np.percentile(token_lens, 99.5), 80), color="#1f5aa6")
    ax.axvline(max_len, color="#c0392b", ls="--", label=f"max_len = {max_len}")
    ax.set(xlabel="tokens after preprocessing", ylabel="reviews", title="Processed length vs. max_len (train)")
    ax.legend(); fig.tight_layout(); fig.savefig(out_dir / "token_length_vs_maxlen.png", dpi=150); plt.close(fig)
    files.append("token_length_vs_maxlen.png")
    return files


# --------------------------------------------------------------------------- vocab / encoding
def build_vocab(token_lists, min_freq: int, max_vocab: int):
    counts = Counter()
    for toks in token_lists:
        counts.update(toks)
    kept = [w for w, c in counts.most_common() if c >= min_freq][: max_vocab - 2]
    itos = [PAD, UNK] + kept
    return {w: i for i, w in enumerate(itos)}, itos, counts


def encode(tokens: list[str], stoi: dict, max_len: int, head: int, truncation: str = "head_tail"):
    ids = [stoi.get(w, UNK_ID) for w in tokens] or [UNK_ID]          # never empty (RNN packing)
    if len(ids) > max_len:
        ids = ids[:max_len] if truncation == "head" else ids[:head] + ids[-(max_len - head):]
    return ids


def encode_all(token_lists, stoi, max_len, head, truncation):
    X = np.zeros((len(token_lists), max_len), dtype=np.int32)
    L = np.zeros(len(token_lists), dtype=np.int32)
    unk = total = 0
    for i, toks in enumerate(token_lists):
        ids = encode(toks, stoi, max_len, head, truncation)
        X[i, :len(ids)] = ids
        L[i] = len(ids)
        unk += sum(1 for w in toks if w not in stoi)
        total += len(toks)
    return X, L, (unk / total if total else 0.0)


# --------------------------------------------------------------------------- main step
DATA_KEYS = ["raw_train", "raw_test", "val_size", "split_seed", "max_train_rows", "lowercase",
             "expand_contractions", "remove_punctuation", "remove_stopwords", "keep_negations", "stemming",
             "min_freq", "max_vocab", "max_len", "truncation", "head_tokens", "dedupe_train",
             "remove_train_test_overlap"]


def prepare_data(dcfg: dict, force: bool = False, log=print) -> Path:
    out_dir = repo_path(dcfg["processed_dir"])
    key = {k: dcfg.get(k) for k in DATA_KEYS}
    if not force and (out_dir / "meta.json").exists() and (out_dir / "X_train.npy").exists():
        if read_json(out_dir / "meta.json").get("data_config") == key:
            log(f"[data] reusing processed data in {rel(out_dir)} (same data config)")
            return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    raw_train, raw_test = _ensure_raw(dcfg, "raw_train", "train"), _ensure_raw(dcfg, "raw_test", "test")

    train_rows, rep_train = load_split(raw_train)
    test_rows, rep_test = load_split(raw_test)
    log(f"[data] raw train: {rep_train}")
    log(f"[data] raw test : {rep_test}")
    if dcfg.get("max_train_rows"):
        rng = np.random.default_rng(dcfg["split_seed"])
        keep = np.sort(rng.choice(len(train_rows), size=min(dcfg["max_train_rows"], len(train_rows)), replace=False))
        train_rows = [train_rows[i] for i in keep]

    # duplicates inside train and train/test overlap (exact normalised text, case-insensitive)
    if dcfg.get("dedupe_train", True):
        seen, uniq = set(), []
        for r in train_rows:
            k = r[1].lower()
            if k not in seen:
                seen.add(k)
                uniq.append(r)
        rep_train["duplicates_removed"] = len(train_rows) - len(uniq)
        train_rows = uniq
    if dcfg.get("remove_train_test_overlap", True):
        test_keys = {r[1].lower() for r in test_rows}
        before = len(train_rows)
        train_rows = [r for r in train_rows if r[1].lower() not in test_keys]
        rep_train["train_test_overlap_removed"] = before - len(train_rows)
    log(f"[data] after cleaning: train={len(train_rows):,} test={len(test_rows):,} "
        f"(dupes removed {rep_train.get('duplicates_removed', 0)}, overlap removed "
        f"{rep_train.get('train_test_overlap_removed', 0)})")

    eda = {"train_full": eda_stats(train_rows), "test": eda_stats(test_rows)}

    # my own stratified validation split
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(train_rows))
    labels = np.array([r[2] for r in train_rows])
    tr_idx, va_idx = train_test_split(idx, test_size=dcfg["val_size"], random_state=dcfg["split_seed"],
                                      stratify=labels)
    splits = {"train": [train_rows[i] for i in tr_idx], "val": [train_rows[i] for i in va_idx], "test": test_rows}
    eda["train"], eda["val"] = eda_stats(splits["train"]), eda_stats(splits["val"])

    # preprocessing + tokenisation
    pre = Preprocessor(dcfg)
    toks = {}
    for name, rows in splits.items():
        t1 = time.time()
        toks[name] = [pre.tokens(r[1]) for r in rows]
        log(f"[data] preprocessed {name}: {len(rows):,} reviews in {time.time() - t1:.0f}s")
    stoi, itos, counts = build_vocab(toks["train"], dcfg["min_freq"], dcfg["max_vocab"])

    stats = {}
    for name in splits:
        X, L, oov = encode_all(toks[name], stoi, dcfg["max_len"], dcfg["head_tokens"], dcfg["truncation"])
        y = np.array([r[2] for r in splits[name]], dtype=np.int8)
        np.save(out_dir / f"X_{name}.npy", X)
        np.save(out_dir / f"len_{name}.npy", L)
        np.save(out_dir / f"y_{name}.npy", y)
        raw_len = np.array([len(t) for t in toks[name]])
        stats[name] = {"n": int(len(y)), "oov_token_rate": oov,
                       "truncated_fraction": float((raw_len > dcfg["max_len"]).mean()),
                       "empty_after_preprocessing": int((raw_len == 0).sum()),
                       "tokens_mean": float(raw_len.mean()), "tokens_median": float(np.median(raw_len)),
                       "tokens_p95": float(np.percentile(raw_len, 95))}

    # raw row indices (reproducible split) + slice masks for robustness evaluation
    from shared_eval.metrics import define_slices
    np.savez_compressed(out_dir / "split_indices.npz",
                        train_rows=np.array([r[0] for r in splits["train"]], dtype=np.int32),
                        val_rows=np.array([r[0] for r in splits["val"]], dtype=np.int32),
                        test_rows=np.array([r[0] for r in splits["test"]], dtype=np.int32))
    for name in ("val", "test"):
        sl = define_slices(r[1] for r in splits[name])
        np.savez_compressed(out_dir / f"slices_{name}.npz", **{k: v for k, v in sl.items()})

    write_json(out_dir / "vocab.json", {"itos": itos, "pad_id": PAD_ID, "unk_id": UNK_ID,
                                        "vocab_size": len(itos), "min_freq": dcfg["min_freq"],
                                        "top_tokens": counts.most_common(50)})
    # most class-indicative tokens (log-odds with add-1 smoothing) -- sanity check of the pipeline
    pos_c, neg_c = Counter(), Counter()
    for t, r in zip(toks["train"], splits["train"]):
        (pos_c if r[2] == 1 else neg_c).update(set(t))
    common = [w for w in itos[2:5000]]
    n_pos, n_neg = int((labels[tr_idx] == 1).sum()), int((labels[tr_idx] == 0).sum())
    lo = {w: np.log((pos_c[w] + 1) / (n_pos + 2)) - np.log((neg_c[w] + 1) / (n_neg + 2)) for w in common}
    ranked = sorted(lo, key=lo.get)
    eda["most_negative_tokens"] = ranked[:25]
    eda["most_positive_tokens"] = ranked[::-1][:25]

    eda_files = plot_eda(splits["train"], splits["test"], np.array([len(t) for t in toks["train"]]),
                         dcfg["max_len"], MEMBER_DIR / "outputs" / "eda" / Path(dcfg["processed_dir"]).name)
    examples = [pre.steps(r[1]) for r in splits["train"][:3]]
    meta = {"data_config": key, "raw_train": rel(raw_train), "raw_test": rel(raw_test),
            "raw_train_sha256": sha256_file(raw_train), "raw_test_sha256": sha256_file(raw_test),
            "malformed_report": {"train": rep_train, "test": rep_test}, "eda": eda, "splits": stats,
            "vocab_size": len(itos), "stopwords_removed": len(pre.stopwords),
            "kept_from_stopword_list": sorted(KEEP_WORDS), "eda_plots": eda_files,
            "preprocessing_examples": examples, "elapsed_sec": time.time() - t0}
    write_json(out_dir / "meta.json", meta)
    log(f"[data] vocab={len(itos):,} train={stats['train']['n']:,} val={stats['val']['n']:,} "
        f"test={stats['test']['n']:,} oov(test)={stats['test']['oov_token_rate']:.3%} "
        f"truncated(train)={stats['train']['truncated_fraction']:.2%} -> {rel(out_dir)} ({time.time() - t0:.0f}s)")
    return out_dir


def load_vocab(processed_dir: Path) -> list[str]:
    return read_json(processed_dir / "vocab.json")["itos"]


def raw_texts(processed_dir: Path, split: str) -> list[str]:
    """Normalised raw review texts of a split, in the same order as X_<split>.npy."""
    meta = read_json(processed_dir / "meta.json")
    path = repo_path(meta["raw_test" if split == "test" else "raw_train"])
    want = np.load(processed_dir / "split_indices.npz")[f"{split}_rows"]
    pos = {int(r): i for i, r in enumerate(want)}
    out: list[str | None] = [None] * len(want)
    for i, text, _ in iter_raw(path):
        if i in pos:
            out[pos[i]] = normalize_raw(text)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Task 2 data preprocessing")
    p.add_argument("--data-config", default="task2_sentiment/shibin_thomas/configs/data.yaml")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    prepare_data(load_config(a.data_config, a.set), force=a.force)


if __name__ == "__main__":
    main()
