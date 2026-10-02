"""Task 2.2.4 -- select 20 test errors of my best model for manual review.

Best model = highest validation macro-F1 among my three models (chosen on validation, not test).
Selection on the official test split (threshold 0.5):
  * 5 confident false positives   true negative, highest P(positive)
  * 5 confident false negatives   true positive, lowest P(positive)
  * 5 near-threshold errors       misclassified with P(positive) closest to 0.5
  * 5 slice-specific failures     the most confident errors inside the slice with the highest error rate
Each error gets a SUGGESTED error type from simple, transparent heuristics (contrast words, negation,
truncation, explicit star rating, confidence). The draft goes to failure_analysis.md (only while it still
carries the AUTO-DRAFT marker); the review itself -- reading each review and confirming / correcting the
type -- is done by hand.

Run:  python task2_sentiment/shibin_thomas/src/error_analysis.py --run-id <run_id|latest>
"""
from __future__ import annotations

import argparse
import re
from collections import Counter

import numpy as np

from data_prep import Preprocessor, raw_texts
from shared_eval.metrics import CONTRAST_RE, NEGATION_RE
from utils import MEMBER_DIR, latest_run_id, read_json, rel, repo_path, run_dirs

AUTO_MARKER = "<!-- AUTO-DRAFT"
STAR_RE = re.compile(r"\b([1-5]|one|two|three|four|five)\s*(?:-|\s)?stars?\b", re.I)
STAR_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}

FIXES = {
    "Mixed sentiment (contrast)": "weight the clause after 'but/however' (e.g. split on contrast words and add a "
                                  "'post-contrast' feature) - measure macro-F1 on the has-contrast slice",
    "Negation / polarity shift": "negation marking: prefix tokens after a negation with NOT_ until the next "
                                 "punctuation - measure macro-F1 on the has-negation slice",
    "Truncation (verdict outside kept tokens)": "raise max_len or keep more tail tokens - measure the error "
                                                "rate on the long-review slice",
    "Explicit star rating ignored": "keep rating phrases as single tokens ('5_star', 'one_star') before "
                                    "stopword removal - measure on reviews that mention stars",
    "Possible label noise / sarcasm": "manually relabel a sample of confident errors to estimate label noise; "
                                      "sarcasm needs context the model lacks",
    "Weak / ambiguous sentiment": "calibrate and abstain near 0.5 (selective prediction) - measure accuracy "
                                  "vs coverage",
}


def suggest_type(text: str, p: float, y: int, n_tokens: int, max_len: int) -> tuple[str, str]:
    """Heuristic error type + the evidence that triggered it."""
    stars = [STAR_WORDS.get(s.lower(), None) or int(s) for s in STAR_RE.findall(text)
             if s.isdigit() or s.lower() in STAR_WORDS]
    if stars and ((y == 1 and max(stars) >= 4) or (y == 0 and min(stars) <= 2)):
        return "Explicit star rating ignored", f"mentions {stars} star(s)"
    contrast = CONTRAST_RE.findall(text)
    if contrast:
        return "Mixed sentiment (contrast)", f"contrast words: {sorted(set(w.lower() for w in contrast))[:4]}"
    neg = NEGATION_RE.findall(text)
    if neg:
        return "Negation / polarity shift", f"negations: {sorted(set(w.lower() for w in neg))[:4]}"
    if n_tokens > max_len:
        return "Truncation (verdict outside kept tokens)", f"{n_tokens} tokens > max_len {max_len}"
    if abs(p - 0.5) > 0.4:
        return "Possible label noise / sarcasm", f"confident (p={p:.3f}) with no structural cue"
    return "Weak / ambiguous sentiment", f"p={p:.3f}; no contrast, negation, rating or truncation cue"


def select_errors(y, p, slices: dict, k: int = 5):
    err = (p >= 0.5).astype(int) != y
    used: set[int] = set()

    def take(cands):
        out = [int(i) for i in cands if i not in used][:k]
        used.update(out)
        return out

    idx = np.arange(len(y))
    groups = {
        "Confident false positive": take(idx[(y == 0) & err][np.argsort(-p[(y == 0) & err])]),
        "Confident false negative": take(idx[(y == 1) & err][np.argsort(p[(y == 1) & err])]),
        "Near-threshold error": take(idx[err][np.argsort(np.abs(p[err] - 0.5))]),
    }
    rates = {n: err[m].mean() for n, m in slices.items() if n != "all" and m.sum() >= 50}
    hardest = max(rates, key=rates.get)
    m = slices[hardest] & err
    groups[f"Slice-specific failure ({hardest})"] = take(idx[m][np.argsort(-np.abs(p[m] - 0.5))])
    return groups, hardest, rates


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", default="latest")
    ap.add_argument("--prefix", default="yelp3")
    ap.add_argument("--smoke", action="store_true", help="never write failure_analysis.md")
    a = ap.parse_args()
    run_id = latest_run_id(a.prefix) if a.run_id == "latest" else a.run_id
    out = run_dirs(run_id)["outputs"]
    names = [p.parent.name for p in sorted(out.glob("*/train_summary.json"))]
    summaries = {n: read_json(out / n / "train_summary.json") for n in names}
    best = max(names, key=lambda n: summaries[n]["best_val_f1_macro"])
    cfg = summaries[best]["config"]
    processed = repo_path(cfg["data"]["processed_dir"])
    pred = np.load(out / "predictions_test.npz")
    y, p = pred["y"].astype(int), pred[best].astype(float)
    slices = dict(np.load(processed / "slices_test.npz"))
    texts = raw_texts(processed, "test")
    pre = Preprocessor(cfg["data"])
    max_len = cfg["data"]["max_len"]

    groups, hardest, rates = select_errors(y, p, slices)
    rows = []
    for group, ids in groups.items():
        for i in ids:
            n_tok = len(pre.tokens(texts[i]))
            etype, why = suggest_type(texts[i], p[i], y[i], n_tok, max_len)
            rows.append({"group": group, "idx": i, "y": int(y[i]), "p": float(p[i]), "type": etype,
                         "evidence": why, "tokens": n_tok, "text": texts[i]})
    type_counts = Counter(r["type"] for r in rows)
    headline = type_counts.most_common(1)[0][0] if rows else None

    def lab(v):
        return "positive" if v == 1 else "negative"

    def snippet(t, n=400):
        t = t.replace("|", "/")
        return t if len(t) <= n else t[:n] + " ..."

    lines = [f"{AUTO_MARKER}: generated by src/error_analysis.py from run {run_id}. Read every review, confirm or "
             "correct the error type and edit the notes in your own words; once you delete this line the file is "
             "never overwritten again. -->",
             "# Task 2.2.4 - Manual review of 20 errors", "",
             f"**Model reviewed:** `{best}` (highest validation macro-F1 of my three models) - run `{run_id}`.",
             f"**Data:** official Yelp polarity test split (n = {len(y):,}), threshold 0.5, "
             f"{int(((p >= 0.5) != y).sum()):,} errors in total.",
             f"**Hardest slice:** `{hardest}` (error rate {rates[hardest]:.2%}; all slice error rates: "
             + ", ".join(f"{k} {v:.2%}" for k, v in sorted(rates.items(), key=lambda kv: -kv[1])) + ").", "",
             "| # | Group | Test idx | True | P(pos) | Error type | Evidence |", "|---|---|---|---|---|---|---|"]
    for n, r in enumerate(rows, 1):
        lines.append(f"| {n} | {r['group']} | {r['idx']} | {lab(r['y'])} | {r['p']:.3f} | {r['type']} | {r['evidence']} |")
    lines += ["", "## The 20 reviews", ""]
    for n, r in enumerate(rows, 1):
        lines += [f"**{n}. {r['group']}** - true {lab(r['y'])}, P(pos) = {r['p']:.3f}, {r['tokens']} tokens after "
                  f"preprocessing - *{r['type']}*", "", f"> {snippet(r['text'])}", ""]
    lines += ["## Error types found", "", "| Error type | Count | Testable fix (and how to measure it) |",
              "|---|---|---|"]
    for t, c in type_counts.most_common():
        lines.append(f"| {t} | {c} | {FIXES[t]} |")
    if headline:
        lines += ["", "## Proposed testable fix", "",
                  f"The most frequent error type is **{headline}** ({type_counts[headline]} of 20). "
                  f"Fix: {FIXES[headline]}. Test it by retraining the same model with only this change and comparing "
                  "the slice metric and overall macro-F1 against the current model with a paired McNemar test on the "
                  "same test reviews.", ""]
    text = "\n".join(lines)
    (out / "error_candidates.md").write_text(text, encoding="utf-8")
    print(f"wrote {rel(out / 'error_candidates.md')} (model {best})")
    fa = MEMBER_DIR / "failure_analysis.md"
    if a.smoke or cfg.get("smoke"):
        return
    if fa.exists() and AUTO_MARKER not in fa.read_text(encoding="utf-8"):
        print(f"{rel(fa)} has been edited by hand - not overwritten")
        return
    fa.write_text(text, encoding="utf-8")
    print(f"wrote draft {rel(fa)} - review each error and put the notes in your own words")


if __name__ == "__main__":
    main()
