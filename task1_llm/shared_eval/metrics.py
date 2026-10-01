"""Team-agreed evaluation metrics for Task 1 (character-level GPT on TinyStories).

Every member imports these functions so that the numbers in the team comparison table
are computed identically. Pure Python / math only -- no framework dependency.

Definitions
-----------
* Cross-entropy (CE) is the mean negative log-likelihood per *character*, in nats.
* Perplexity        = exp(CE)                    (per-character perplexity)
* Bits-per-character = CE / ln(2)
* Generalization gap = val CE - train CE         (train CE measured in eval mode, no dropout)
* Distinct-n  (corpus level, Li et al. 2016) = #unique word n-grams / #word n-grams over
  ALL generated continuations (prompt excluded). Higher = more diverse.
* Repeated 4-gram rate (per sample, then averaged) = 1 - #unique 4-grams / #4-grams inside
  one continuation. 0 = no repeated phrase, -> 1 = degenerate looping.
* Words are lowercase alphabetic tokens (apostrophes kept): "Lily's" -> "lily's".
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable, Sequence

WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")


def perplexity(ce_nats: float) -> float:
    return math.exp(ce_nats)


def bits_per_char(ce_nats: float) -> float:
    return ce_nats / math.log(2)


def generalization_gap(train_ce: float, val_ce: float) -> float:
    return val_ce - train_ce


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def ngrams(tokens: Sequence[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def distinct_n(texts: Iterable[str], n: int) -> float:
    total, unique = 0, set()
    for t in texts:
        grams = ngrams(words(t), n)
        total += len(grams)
        unique.update(grams)
    return len(unique) / total if total else 0.0


def repeated_ngram_rate(text: str, n: int = 4) -> float | None:
    grams = ngrams(words(text), n)
    if not grams:
        return None
    return 1.0 - len(set(grams)) / len(grams)


def mean_repeated_ngram_rate(texts: Iterable[str], n: int = 4) -> float:
    rates = [r for r in (repeated_ngram_rate(t, n) for t in texts) if r is not None]
    return sum(rates) / len(rates) if rates else 0.0


def most_repeated_ngrams(text: str, n: int = 4, k: int = 3) -> list[tuple[str, int]]:
    c = Counter(ngrams(words(text), n))
    return [(" ".join(g), cnt) for g, cnt in c.most_common(k) if cnt > 1]


def generation_metrics(continuations: Sequence[str]) -> dict[str, float]:
    """All diversity / repetition metrics for a list of generated continuations."""
    return {
        "distinct_1": distinct_n(continuations, 1),
        "distinct_2": distinct_n(continuations, 2),
        "distinct_3": distinct_n(continuations, 3),
        "repeated_4gram_rate": mean_repeated_ngram_rate(continuations, 4),
    }


def loss_metrics(train_ce: float, val_ce: float) -> dict[str, float]:
    return {
        "train_cross_entropy": train_ce,
        "val_cross_entropy": val_ce,
        "val_perplexity": perplexity(val_ce),
        "val_bits_per_char": bits_per_char(val_ce),
        "train_perplexity": perplexity(train_ce),
        "train_bits_per_char": bits_per_char(train_ce),
        "generalization_gap": generalization_gap(train_ce, val_ce),
    }
