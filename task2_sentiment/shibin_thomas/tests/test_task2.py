"""Sanity tests for Task 2 (no data or GPU needed). Run:  python task2_sentiment/shibin_thomas/tests/test_task2.py"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from data_prep import UNK_ID, Preprocessor, build_vocab, encode, normalize_raw  # noqa: E402
from models import build_model  # noqa: E402
from shared_eval.metrics import (bootstrap_ci, classification_metrics, define_slices,  # noqa: E402
                                 expected_calibration_error, mcnemar)

PRE = Preprocessor({"stemming": "snowball"})


def test_normalize_handles_dump_artifacts_and_missing():
    assert normalize_raw('Great food!\\nWe &amp; friends said \\"wow\\"') == 'Great food! We & friends said "wow"'
    assert normalize_raw(None) is None and normalize_raw(3.0) is None


def test_preprocessing_steps():
    toks = PRE.tokens("The pizza wasn't GOOD, but the staff were AMAZING!!! 5 stars")
    assert "not" in toks and "but" in toks                 # negation + contrast survive stopword removal
    assert "the" not in toks and "were" not in toks         # ordinary stopwords removed
    assert "5" in toks and "star" in toks                   # digits kept, plural stemmed
    assert all(t == t.lower() and t.isalnum() for t in toks)  # lowercase, no punctuation
    assert PRE.tokens("loved loving loves") == ["love", "love", "love"]


def test_vocab_and_head_tail_truncation():
    stoi, itos, _ = build_vocab([["a", "b", "a"], ["a", "c"]], min_freq=2, max_vocab=10)
    assert itos[:3] == ["<pad>", "<unk>", "a"] and stoi.get("b") is None
    stoi = {w: i + 2 for i, w in enumerate("t0 t1 t2 t3 t4 t5 t6 t7 t8 t9".split())}
    ids = encode([f"t{i}" for i in range(10)], stoi, max_len=6, head=4)
    assert ids == [2, 3, 4, 5, 10, 11]                      # first 4 + last 2 tokens
    assert encode([], stoi, 6, 4) == [UNK_ID]               # never empty


CFGS = [{"type": "meanpool_mlp", "embed_dim": 16, "hidden_dim": 8, "dropout": 0.0},
        {"type": "textcnn", "embed_dim": 16, "kernel_sizes": [3, 4, 5], "num_filters": 8, "dropout": 0.0},
        {"type": "bigru_attn", "embed_dim": 16, "hidden_dim": 8, "num_layers": 2, "attn_dim": 8, "dropout": 0.0}]


def test_models_shapes_and_padding_invariance():
    torch.manual_seed(0)
    x = torch.randint(2, 50, (3, 12))
    L = torch.tensor([12, 7, 3])
    for i in range(3):
        x[i, L[i]:] = 0
    for cfg in CFGS:
        m = build_model(cfg, vocab_size=50).eval()
        out = m(x, L)
        assert out.shape == (3,), cfg["type"]
        padded = torch.cat([x, torch.zeros(3, 9, dtype=torch.long)], dim=1)   # extra padding must not matter
        assert torch.allclose(out, m(padded, L), atol=1e-5), cfg["type"]


def test_models_learn_a_toy_task():
    torch.manual_seed(0)
    x = torch.randint(4, 30, (64, 10))
    y = (torch.rand(64) > 0.5).float()
    x[:, 3] = torch.where(y > 0, torch.tensor(2), torch.tensor(3))          # token 2 = positive cue
    L = torch.full((64,), 10)
    for cfg in CFGS:
        m = build_model(cfg, vocab_size=30)
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        for _ in range(60):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(m(x, L), y)
            opt.zero_grad(); loss.backward(); opt.step()
        assert loss.item() < 0.2, (cfg["type"], loss.item())


def test_metrics_against_sklearn_and_known_values():
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.6 + rng.random(500) * 0.5, 0, 1)
    m = classification_metrics(y, p)
    yhat = (p >= 0.5).astype(int)
    assert np.isclose(m["accuracy"], accuracy_score(y, yhat))
    assert np.isclose(m["f1_macro"], f1_score(y, yhat, average="macro"))
    assert np.isclose(m["f1_micro"], m["accuracy"])
    assert np.isclose(m["mcc"], matthews_corrcoef(y, yhat))
    assert m["tn"] + m["fp"] + m["fn"] + m["tp"] == 500
    # perfectly calibrated & perfectly confident predictor -> ECE 0, Brier 0
    assert expected_calibration_error(np.array([0, 1]), np.array([0.0, 1.0])) == 0.0
    # always 0.9 confident but right only half the time -> ECE 0.4
    assert np.isclose(expected_calibration_error(np.array([1, 0] * 50), np.full(100, 0.9)), 0.4)
    ci = bootstrap_ci(y, yhat, n_boot=200)
    assert ci["accuracy_ci_low"] <= m["accuracy"] <= ci["accuracy_ci_high"]
    assert ci["mcc_ci_low"] <= m["mcc"] <= ci["mcc_ci_high"]


def test_mcnemar_known_counts():
    y = np.zeros(30, dtype=int)
    a = np.zeros(30, dtype=int)
    b = np.zeros(30, dtype=int)
    a[:10] = 1                 # baseline wrong on 0-9
    b[8:12] = 1                # model wrong on 8-11 -> b=2 (10,11), c=8 (0-7)
    r = mcnemar(y, a, b)
    assert r["b_baseline_only_correct"] == 2 and r["c_model_only_correct"] == 8
    assert np.isclose(r["chi2"], (abs(2 - 8) - 1) ** 2 / 10)


def test_slices():
    s = define_slices(["Not good at all", "Great " * 60, "Good but slow!!!"])
    assert s["has negation"].tolist() == [True, False, False]
    assert s["long (>150 words)"].tolist() == [False, False, False]
    assert s["medium (51-150 words)"].tolist() == [False, True, False]
    assert s["has contrast (but/however/...)"].tolist() == [False, False, True]
    assert s["exclamation-heavy (>=3 '!')"].tolist() == [False, False, True]


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
