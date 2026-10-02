"""Team-agreed evaluation for Task 2 (binary sentiment, Yelp polarity).

Every member computes their numbers with these functions on the official Yelp polarity TEST split,
so the team comparison table is like-for-like.

Conventions
-----------
* Labels: 0 = negative, 1 = positive. Models output p = P(positive). Predicted label = 1 if p >= 0.5.
* Precision / recall / F1 are reported per class and averaged three ways:
  macro (unweighted mean over the 2 classes), micro (pooled counts -- equals accuracy for single-label
  binary classification) and weighted (mean weighted by class support).
* ROC-AUC and PR-AUC (average precision) use p for the positive class.
* Brier score = mean (p - y)^2   (0 = perfect, 0.25 = always predicting 0.5).
* ECE (expected calibration error, Guo et al. 2017): confidence = max(p, 1-p) of the predicted class,
  15 equal-width bins on [0, 1], ECE = sum_b (n_b / N) * |accuracy_b - mean confidence_b|.
* 95% bootstrap CIs: 1,000 resamples of the test set with replacement (seed 0), percentile method.
* Paired McNemar test on the same test examples: b = baseline right & model wrong, c = baseline wrong &
  model right; chi-square with continuity correction (1 dof) and the exact binomial p-value.
* Slices (robustness), defined on the RAW review text: see `define_slices`.
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np
from scipy.stats import binomtest, chi2
from sklearn.metrics import (average_precision_score, confusion_matrix, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score)

THRESHOLD = 0.5
N_BOOT = 1000
ECE_BINS = 15

NEGATION_RE = re.compile(r"\b(?:not|no|never|nothing|nobody|none|neither|nor|cannot|without)\b|n't\b", re.I)
CONTRAST_RE = re.compile(r"\b(?:but|however|although|though|yet|except|despite|whereas)\b", re.I)


# --------------------------------------------------------------------------- core metrics
def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = ECE_BINS) -> float:
    pred = (p >= THRESHOLD).astype(int)
    conf = np.where(pred == 1, p, 1 - p)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def classification_metrics(y: Iterable[int], p: Iterable[float], threshold: float = THRESHOLD) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    yhat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    out = {"n": int(y.size), "accuracy": float((yhat == y).mean()),
           "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    for avg in ("macro", "micro", "weighted"):
        pr, rc, f1, _ = precision_recall_fscore_support(y, yhat, average=avg, labels=[0, 1], zero_division=0)
        out.update({f"precision_{avg}": float(pr), f"recall_{avg}": float(rc), f"f1_{avg}": float(f1)})
    pr, rc, f1, sup = precision_recall_fscore_support(y, yhat, average=None, labels=[0, 1], zero_division=0)
    for i, name in enumerate(["neg", "pos"]):
        out.update({f"precision_{name}": float(pr[i]), f"recall_{name}": float(rc[i]),
                    f"f1_{name}": float(f1[i]), f"support_{name}": int(sup[i])})
    both = len(np.unique(y)) == 2
    out["roc_auc"] = float(roc_auc_score(y, p)) if both else float("nan")
    out["pr_auc"] = float(average_precision_score(y, p)) if both else float("nan")
    out["mcc"] = float(matthews_corrcoef(y, yhat))
    out["brier"] = float(np.mean((p - y) ** 2))
    out["ece"] = expected_calibration_error(y, p)
    out["error_rate"] = 1.0 - out["accuracy"]
    return out


# --------------------------------------------------------------------------- bootstrap
def _acc_f1_mcc_from_counts(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    acc = (tp + tn) / n
    f1_pos = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=(2 * tp + fp + fn) > 0)
    f1_neg = np.divide(2 * tn, 2 * tn + fn + fp, out=np.zeros_like(tn, dtype=float), where=(2 * tn + fn + fp) > 0)
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)
    return acc, (f1_pos + f1_neg) / 2, mcc


def bootstrap_ci(y, yhat, n_boot: int = N_BOOT, seed: int = 0, alpha: float = 0.05, chunk: int = 100) -> dict:
    """Percentile bootstrap CIs for accuracy, macro-F1 and MCC (vectorised over resamples)."""
    y = np.asarray(y).astype(np.int8)
    yhat = np.asarray(yhat).astype(np.int8)
    n = y.size
    rng = np.random.default_rng(seed)
    accs, f1s, mccs = [], [], []
    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        idx = rng.integers(0, n, size=(b, n))
        yy, pp = y[idx], yhat[idx]
        tp = ((yy == 1) & (pp == 1)).sum(1).astype(float)
        tn = ((yy == 0) & (pp == 0)).sum(1).astype(float)
        fp = ((yy == 0) & (pp == 1)).sum(1).astype(float)
        fn = ((yy == 1) & (pp == 0)).sum(1).astype(float)
        a, f, m = _acc_f1_mcc_from_counts(tp, fp, fn, tn)
        accs.append(a), f1s.append(f), mccs.append(m)
    out = {}
    for name, vals in (("accuracy", accs), ("f1_macro", f1s), ("mcc", mccs)):
        v = np.concatenate(vals)
        out[f"{name}_ci_low"] = float(np.percentile(v, 100 * alpha / 2))
        out[f"{name}_ci_high"] = float(np.percentile(v, 100 * (1 - alpha / 2)))
    out["n_boot"] = n_boot
    return out


# --------------------------------------------------------------------------- McNemar
def mcnemar(y, pred_baseline, pred_model) -> dict:
    y, a, m = (np.asarray(v).astype(int) for v in (y, pred_baseline, pred_model))
    a_ok, m_ok = a == y, m == y
    b = int((a_ok & ~m_ok).sum())      # baseline right, model wrong
    c = int((~a_ok & m_ok).sum())      # baseline wrong, model right
    if b + c == 0:
        return {"b_baseline_only_correct": b, "c_model_only_correct": c, "chi2": 0.0, "p_value": 1.0,
                "p_value_exact": 1.0}
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return {"b_baseline_only_correct": b, "c_model_only_correct": c, "chi2": float(stat),
            "p_value": float(chi2.sf(stat, df=1)),
            "p_value_exact": float(binomtest(min(b, c), b + c, 0.5).pvalue)}


# --------------------------------------------------------------------------- slices
def define_slices(raw_texts: Iterable[str]) -> dict[str, np.ndarray]:
    """Boolean masks over the RAW reviews (before any preprocessing)."""
    texts = list(raw_texts)
    n_words = np.array([len(t.split()) for t in texts])
    return {
        "all": np.ones(len(texts), dtype=bool),
        "short (<=50 words)": n_words <= 50,
        "medium (51-150 words)": (n_words > 50) & (n_words <= 150),
        "long (>150 words)": n_words > 150,
        "has negation": np.array([bool(NEGATION_RE.search(t)) for t in texts]),
        "has contrast (but/however/...)": np.array([bool(CONTRAST_RE.search(t)) for t in texts]),
        "no negation & no contrast": np.array([not NEGATION_RE.search(t) and not CONTRAST_RE.search(t)
                                               for t in texts]),
        "exclamation-heavy (>=3 '!')": np.array([t.count("!") >= 3 for t in texts]),
    }


def slice_metrics(y, yhat, slices: dict[str, np.ndarray]) -> list[dict]:
    y, yhat = np.asarray(y).astype(int), np.asarray(yhat).astype(int)
    rows = []
    for name, m in slices.items():
        if m.sum() == 0:
            continue
        _, _, f1, _ = precision_recall_fscore_support(y[m], yhat[m], average="macro", labels=[0, 1],
                                                      zero_division=0)
        rows.append({"slice": name, "n": int(m.sum()), "f1_macro": float(f1),
                     "error_rate": float((y[m] != yhat[m]).mean()), "pos_rate": float(y[m].mean())})
    return rows
