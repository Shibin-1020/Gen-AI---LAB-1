"""Task 2.2 / 2.3 -- evaluate every model of a run on the official Yelp polarity TEST split.

Per model: accuracy; precision / recall / F1 (macro, micro, weighted, per class); confusion matrix;
ROC-AUC; PR-AUC; MCC; Brier score; ECE; 95% bootstrap CIs (accuracy, macro-F1, MCC); macro-F1 and
error rate per data slice; parameter count; training time; train + inference examples/sec; peak memory;
exact hardware. Across models: paired McNemar test baseline vs each experimental model.

Outputs (task2_sentiment/shibin_thomas/outputs/<run_id>/):
    metrics_report.csv      every metric, every model (long format: model, metric, value, notes)
    comparison.csv / .md    side-by-side table of the main metrics + McNemar + slices
    predictions_test.npz    y and P(positive) of every model (used by error_analysis.py)
    <model>/metrics.json    full metric dictionary;  plots: confusion matrices, ROC, PR, reliability,
                            training curves, slice robustness
With promote (default for non-smoke runs) metrics_report.csv and comparison.md are copied to the member
folder and the tables in results.md are regenerated.
"""
from __future__ import annotations

import csv
import re
import shutil
import sys
import textwrap
from pathlib import Path

import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from models import build_model  # noqa: E402
from shared_eval.metrics import bootstrap_ci, classification_metrics, mcnemar, slice_metrics  # noqa: E402
from train import Split, predict  # noqa: E402
from utils import (MEMBER_DIR, get_device, get_logger, peak_memory_mb, read_json, rel, run_dirs,  # noqa: E402
                   update_manifest, write_json)

COLORS = ["#7f8c8d", "#1f5aa6", "#c0392b", "#2a8f4a", "#8e44ad"]
MAIN_ROWS = [  # (key, label) for the comparison table
    ("accuracy", "Accuracy"), ("accuracy_ci", "Accuracy 95% CI"), ("precision_macro", "Precision (macro)"),
    ("recall_macro", "Recall (macro)"), ("f1_macro", "F1 (macro)"), ("f1_macro_ci", "F1 macro 95% CI"),
    ("precision_micro", "Precision (micro)"), ("recall_micro", "Recall (micro)"), ("f1_micro", "F1 (micro)"),
    ("precision_weighted", "Precision (weighted)"), ("recall_weighted", "Recall (weighted)"),
    ("f1_weighted", "F1 (weighted)"), ("confusion", "Confusion matrix [TN FP; FN TP]"),
    ("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("mcc", "MCC"), ("mcc_ci", "MCC 95% CI"),
    ("brier", "Brier score"), ("ece", "ECE (15 bins)"), ("mcnemar", "McNemar vs baseline (b / c, p)"),
    ("params", "Parameters"), ("training_time_min", "Training time (min)"), ("epochs", "Epochs run (best)"),
    ("train_examples_per_sec", "Train examples/sec"), ("inference_examples_per_sec", "Inference examples/sec"),
    ("peak_gpu_allocated_mb", "Peak GPU memory (MB)"), ("peak_cpu_rss_mb", "Peak CPU RSS (MB, process)"),
    ("hardware", "Hardware"),
]


def load_trained(ck_path, device):
    ck = torch.load(ck_path, map_location=device, weights_only=False)
    model = build_model(ck["model_config"], ck["vocab_size"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def _fmt(v, key=""):
    if isinstance(v, float):
        if key.endswith("per_sec") or key.startswith("peak"):
            return f"{v:,.0f}"
        return f"{v:.4f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


# --------------------------------------------------------------------------- plots
def plot_confusion(m, name, path):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    norm = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({norm[i, j]:.1%})", ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["neg", "pos"], yticklabels=["neg", "pos"],
           xlabel="predicted", ylabel="true", title=f"Confusion matrix - {name}")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_curves(y, probs: dict, out):
    from sklearn.metrics import precision_recall_curve, roc_curve
    for kind in ("roc", "pr"):
        fig, ax = plt.subplots(figsize=(5.5, 5))
        for (name, p), c in zip(probs.items(), COLORS):
            if kind == "roc":
                fpr, tpr, _ = roc_curve(y, p)
                ax.plot(fpr, tpr, color=c, label=name)
            else:
                pr, rc, _ = precision_recall_curve(y, p)
                ax.plot(rc, pr, color=c, label=name)
        if kind == "roc":
            ax.plot([0, 1], [0, 1], ls="--", color="#aaa")
            ax.set(xlabel="false positive rate", ylabel="true positive rate", title="ROC curves (test)")
        else:
            ax.set(xlabel="recall", ylabel="precision", title="Precision-recall curves (test)")
        ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout(); fig.savefig(out / f"{kind}_curves.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    bins = np.linspace(0, 1, 11)
    for (name, p), c in zip(probs.items(), COLORS):
        idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
        xs, ys = [], []
        for b in range(10):
            m = idx == b
            if m.sum() >= 20:
                xs.append(p[m].mean()); ys.append(y[m].mean())
        ax.plot(xs, ys, "o-", color=c, label=name)
    ax.plot([0, 1], [0, 1], ls="--", color="#aaa", label="perfect calibration")
    ax.set(xlabel="predicted P(positive)", ylabel="observed fraction positive", title="Reliability diagram (test)")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(out / "reliability.png", dpi=150); plt.close(fig)


def plot_training(run_logs, names, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, c in zip(names, COLORS):
        rows = list(csv.DictReader(open(run_logs / name / "epochs.csv", encoding="utf-8")))
        ep = [int(r["epoch"]) for r in rows]
        axes[0].plot(ep, [float(r["train_loss"]) for r in rows], "o--", color=c, alpha=0.6, label=f"{name} train")
        axes[0].plot(ep, [float(r["val_loss"]) for r in rows], "s-", color=c, label=f"{name} val")
        axes[1].plot(ep, [float(r["val_f1_macro"]) for r in rows], "s-", color=c, label=name)
    axes[0].set(xlabel="epoch", ylabel="BCE loss", title="Training / validation loss")
    axes[1].set(xlabel="epoch", ylabel="macro-F1", title="Validation macro-F1")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "training_curves.png", dpi=150); plt.close(fig)


def plot_slices(slice_rows: dict, out):
    names = list(slice_rows)
    slices = [r["slice"] for r in slice_rows[names[0]]]
    x = np.arange(len(slices))
    w = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(12, 5))
    for k, (name, c) in enumerate(zip(names, COLORS)):
        ax.bar(x + k * w, [r["error_rate"] for r in slice_rows[name]], w, color=c, label=name)
    labels = [textwrap.fill(r["slice"], 16) + f"\nn={r['n']:,}" for r in slice_rows[names[0]]]
    ax.set_xticks(x + w * (len(names) - 1) / 2, labels, fontsize=8)
    ax.set(ylabel="error rate", title="Error rate per data slice (test)")
    ax.grid(alpha=0.3, axis="y"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "slice_error_rates.png", dpi=150); plt.close(fig)


def data_block(processed) -> str:
    """Markdown summary of the EDA / cleaning / preprocessing statistics (from data_processed/<x>/meta.json)."""
    meta = read_json(processed / "meta.json")
    e, sp, mr = meta["eda"], meta["splits"], meta["malformed_report"]
    rows = ["| Split | Reviews | Negative | Positive | Positive rate | Words mean | median | p95 | max |",
            "|---|---|---|---|---|---|---|---|---|"]
    for k, label in (("train_full", "official train (cleaned)"), ("train", "my train"), ("val", "my validation"),
                     ("test", "official test")):
        d = e[k]
        rows.append(f"| {label} | {d['n']:,} | {d['class_counts']['negative']:,} | {d['class_counts']['positive']:,} | "
                    f"{d['positive_rate']:.2%} | {d['length_words']['mean']:.1f} | {d['length_words']['median']:.0f} | "
                    f"{d['length_words']['p95']:.0f} | {d['length_words']['max']:.0f} |")
    tr, te = mr["train"], mr["test"]
    clean = (f"Raw rows: train {tr['rows']:,}, test {te['rows']:,}. Removed from train: "
             f"{tr.get('missing_or_non_string_text', 0)} missing/non-string text, {tr.get('invalid_label', 0)} invalid "
             f"labels, {tr.get('empty_after_normalisation', 0)} empty, {tr.get('duplicates_removed', 0):,} exact "
             f"duplicates, {tr.get('train_test_overlap_removed', 0):,} reviews also present in test. Removed from "
             f"test: {te.get('missing_or_non_string_text', 0)} missing, {te.get('invalid_label', 0)} invalid labels, "
             f"{te.get('empty_after_normalisation', 0)} empty.")
    toks = (f"After preprocessing: mean {sp['train']['tokens_mean']:.1f} tokens per review (median "
            f"{sp['train']['tokens_median']:.0f}, p95 {sp['train']['tokens_p95']:.0f}); "
            f"{sp['train']['truncated_fraction']:.2%} of training reviews exceed max_len and are head+tail truncated; "
            f"vocabulary {meta['vocab_size']:,} tokens; out-of-vocabulary token rate on test "
            f"{sp['test']['oov_token_rate']:.2%}; {sp['train']['empty_after_preprocessing']} training reviews became "
            f"empty after stopword removal (encoded as a single <unk>).")
    bal = e["train_full"]["balance_ratio_minority_over_majority"]
    neg_len, pos_len = e["train"]["length_words_negative"], e["train"]["length_words_positive"]
    verdict = ("essentially balanced - accuracy is meaningful and no re-weighting is needed" if bal >= 0.9 else
               "noticeably imbalanced - macro-F1, MCC and PR-AUC are the metrics to trust over accuracy")
    obs = (f"Class balance (minority / majority) = {bal:.3f}, i.e. {verdict}. Negative reviews are on average {neg_len['mean']:.0f} words long versus "
           f"{pos_len['mean']:.0f} for positive ones (medians {neg_len['median']:.0f} / {pos_len['median']:.0f}).")
    words = (f"Most negative tokens: {', '.join(e['most_negative_tokens'][:12])}. "
             f"Most positive tokens: {', '.join(e['most_positive_tokens'][:12])}.")
    return ("_Auto-generated from `data_processed/" + Path(processed).name + "/meta.json`._\n\n" + "\n".join(rows)
            + "\n\n" + obs + "\n\n**Missing / malformed entries.** " + clean + "\n\n**Tokens.** " + toks
            + "\n\n**Sanity check (log-odds).** " + words)


# --------------------------------------------------------------------------- main
def evaluate_run(cfgs: list[dict], run_id: str, processed, promote: bool | None = None) -> dict:
    dirs = run_dirs(run_id)
    out = dirs["outputs"]
    log = get_logger(f"eval.{run_id}", dirs["raw_logs"] / "eval.log")
    device = get_device(cfgs[0].get("device", "auto"))
    test = Split(processed, "test", device)
    y = test.y.cpu().numpy().astype(int)
    sl = dict(np.load(processed / "slices_test.npz"))
    names = [c["name"] for c in cfgs]
    baseline = next(c["name"] for c in cfgs if c.get("role") == "baseline")

    results, probs, slice_rows = {}, {}, {}
    for cfg in cfgs:
        name = cfg["name"]
        model, ck = load_trained(dirs["checkpoints"] / name / "best_model.pt", device)
        p, inf_eps = predict(model, test, cfg["training"].get("eval_batch_size", 1024))
        summ = read_json(out / name / "train_summary.json")
        m = classification_metrics(y, p)
        m.update(bootstrap_ci(y, (p >= 0.5).astype(int)))
        rows = slice_metrics(y, (p >= 0.5).astype(int), sl)
        m.update({"params": summ["params"], "training_time_sec": summ["training_time_sec"],
                  "training_time_min": summ["training_time_sec"] / 60, "epochs_run": summ["epochs_run"],
                  "best_epoch": summ["best_epoch"], "best_val_f1_macro": summ["best_val_f1_macro"],
                  "train_examples_per_sec": summ["train_examples_per_sec"],
                  "inference_examples_per_sec": inf_eps, "role": cfg.get("role", "experimental"),
                  "hardware": summ["hardware"].get("gpu_name") or summ["hardware"].get("cpu_model"),
                  "cpu": summ["hardware"].get("cpu_model"), "device": summ["device"],
                  **{k: v for k, v in summ.items() if k.startswith("peak_")}})
        results[name], probs[name], slice_rows[name] = m, p, rows
        write_json(out / name / "metrics.json", {"metrics": m, "slices": rows})
        plot_confusion(m, name, out / name / "confusion_matrix.png")
        log.info(f"{name}: acc={m['accuracy']:.4f} [{m['accuracy_ci_low']:.4f}, {m['accuracy_ci_high']:.4f}] "
                 f"f1_macro={m['f1_macro']:.4f} mcc={m['mcc']:.4f} auc={m['roc_auc']:.4f} pr_auc={m['pr_auc']:.4f} "
                 f"brier={m['brier']:.4f} ece={m['ece']:.4f} cm=[{m['tn']} {m['fp']}; {m['fn']} {m['tp']}]")

    tests = {}
    for name in names:
        if name != baseline:
            tests[name] = mcnemar(y, probs[baseline] >= 0.5, probs[name] >= 0.5)
            results[name]["mcnemar"] = tests[name]
            log.info(f"McNemar {baseline} vs {name}: {tests[name]}")
    write_json(out / "mcnemar.json", {"baseline": baseline, "tests": tests})
    np.savez_compressed(out / "predictions_test.npz", y=y, **{n: probs[n].astype(np.float32) for n in names})
    plot_curves(y, probs, out)
    plot_training(dirs["raw_logs"], names, out)
    plot_slices(slice_rows, out)

    # ---- metrics_report.csv (long) -------------------------------------------------------------
    notes = {"f1_micro": "equals accuracy for single-label binary classification",
             "ece": "15 equal-width confidence bins", "brier": "mean (p - y)^2, lower is better",
             "peak_cpu_rss_mb": "whole pipeline process (cumulative peak)",
             "inference_examples_per_sec": "test set, batch inference"}
    skip = {"mcnemar"}
    with open(out / "metrics_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "metric", "value", "notes"])
        for name in names:
            for k, v in results[name].items():
                if k not in skip:
                    w.writerow([name, k, v, notes.get(k, "")])
            for r in slice_rows[name]:
                w.writerow([name, f"slice[{r['slice']}].f1_macro", r["f1_macro"], f"n={r['n']}"])
                w.writerow([name, f"slice[{r['slice']}].error_rate", r["error_rate"], f"n={r['n']}"])
            if name in tests:
                for k, v in tests[name].items():
                    w.writerow([name, f"mcnemar_vs_{baseline}.{k}", v, "paired test on the same test reviews"])
        w.writerow(["all", "test_examples", len(y), "official Yelp polarity test split"])
        w.writerow(["all", "run_id", run_id, "evidence: reproducibility/raw_logs/... and outputs/<run_id>/"])

    # ---- comparison table ------------------------------------------------------------------------
    def cell(name, key):
        m = results[name]
        if key.endswith("_ci"):
            k = key[:-3]
            return f"[{m[k + '_ci_low']:.4f}, {m[k + '_ci_high']:.4f}]"
        if key == "confusion":
            return f"[{m['tn']:,} {m['fp']:,}; {m['fn']:,} {m['tp']:,}]"
        if key == "mcnemar":
            t = m.get("mcnemar")
            return "- (baseline)" if t is None else (f"{t['b_baseline_only_correct']:,} / "
                                                     f"{t['c_model_only_correct']:,}, p={t['p_value']:.2e}")
        if key == "epochs":
            return f"{m['epochs_run']} ({m['best_epoch']})"
        return _fmt(m.get(key, "-"), key)

    header = "| Metric | " + " | ".join(f"{n} ({results[n]['role']})" for n in names) + " |"
    lines = [header, "|---|" + "---|" * len(names)]
    lines += [f"| {label} | " + " | ".join(cell(n, k) for n in names) + " |" for k, label in MAIN_ROWS]
    slice_lines = ["| Slice (n) | " + " | ".join(f"{n} macro-F1 / error" for n in names) + " |",
                   "|---|" + "---|" * len(names)]
    for i, r in enumerate(slice_rows[names[0]]):
        slice_lines.append(f"| {r['slice']} ({r['n']:,}) | " + " | ".join(
            f"{slice_rows[n][i]['f1_macro']:.4f} / {slice_rows[n][i]['error_rate']:.4f}" for n in names) + " |")
    md = (f"# Task 2 model comparison - run `{run_id}`\n\nOfficial Yelp polarity test split, n = {len(y):,}. "
          f"McNemar: b = baseline right & model wrong, c = baseline wrong & model right.\n\n"
          + "\n".join(lines) + "\n\n## Robustness: macro-F1 / error rate per slice\n\n" + "\n".join(slice_lines) + "\n")
    (out / "comparison.md").write_text(md, encoding="utf-8")
    with open(out / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric"] + names)
        for k, label in MAIN_ROWS:
            w.writerow([label] + [cell(n, k) for n in names])

    smoke = any(c.get("smoke") for c in cfgs)
    promote = (not smoke) if promote is None else promote
    if promote:
        shutil.copy(out / "metrics_report.csv", MEMBER_DIR / "metrics_report.csv")
        shutil.copy(out / "comparison.md", MEMBER_DIR / "comparison.md")
        res = MEMBER_DIR / "results.md"
        if res.exists():
            block = (f"_Auto-generated by `src/evaluate.py` from run `{run_id}` (official test split, n = {len(y):,}; "
                     f"source `outputs/{run_id}/metrics_report.csv`)._\n\n" + "\n".join(lines)
                     + "\n\n**Robustness - macro-F1 / error rate per slice**\n\n" + "\n".join(slice_lines))
            text = res.read_text(encoding="utf-8")
            text = re.sub(r"(<!-- DATA:START -->\n).*?(\n<!-- DATA:END -->)",
                          lambda mm: mm.group(1) + data_block(processed) + mm.group(2), text, flags=re.S)
            text = re.sub(r"(<!-- METRICS:START -->\n).*?(\n<!-- METRICS:END -->)",
                          lambda mm: mm.group(1) + block + mm.group(2), text, flags=re.S)
            text = re.sub(r"outputs/(?:RUN_ID|[^/()\s]+)/(?=[a-z_]+\.png)", f"outputs/{run_id}/", text)
            res.write_text(text, encoding="utf-8")
        log.info("promoted metrics_report.csv + comparison.md to the member folder; refreshed results.md")
    update_manifest(run_id, evaluation={
        "metrics_report": rel(out / "metrics_report.csv"), "comparison": rel(out / "comparison.md"),
        "predictions": rel(out / "predictions_test.npz"), "plots": [rel(p) for p in sorted(out.rglob("*.png"))],
        "eval_peak_memory": peak_memory_mb(device)})
    log.info(f"EVAL DONE -> {rel(out)}")
    return results
