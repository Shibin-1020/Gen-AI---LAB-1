"""Task 1 -- Final evaluation of the best checkpoint: every required metric, plots, samples.

Outputs (task1_llm/shibin_thomas/outputs/<run_id>/):
    metrics.json / metrics_report.csv   every Task 1 metric in one place
    samples.json / samples.md           greedy + temperature samples for the shared prompts
    loss_curves.png                     train (per step + per epoch) and validation loss
    grad_norm.png, lr_schedule.png      training-stability evidence
With --promote (default for non-smoke runs) metrics_report.csv is also copied to the member
folder and the metrics table inside results.md is regenerated.

Run:  python task1_llm/shibin_thomas/src/evaluate.py --config ... --run-id <run_id|latest>
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import shutil
import sys
import time

import matplotlib
if "ipykernel" not in sys.modules:      # headless for CLI runs; keep inline plots working in notebooks
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from data_prep import EOS_ID, CharStream, decode, encode, load_vocab  # noqa: E402
from model import GPTCharLM, GPTConfig  # noqa: E402
from shared_eval.metrics import generation_metrics, loss_metrics  # noqa: E402
from train import evaluate_stream, load_step_log  # noqa: E402
from utils import (MEMBER_DIR, autocast_ctx, get_device, get_logger, latest_run_id, load_config,  # noqa: E402
                   peak_memory_mb, read_json, rel, repo_path, resolve_precision, run_dirs, sync,
                   update_manifest, write_json)

# (metric key, display name, unit / note) -- order of metrics_report.csv
REPORT_ROWS = [
    ("train_cross_entropy", "Training cross-entropy loss", "nats/char, eval mode, best ckpt"),
    ("train_loss_running_last_epoch", "Training loss (running avg, last epoch)", "nats/char, with dropout"),
    ("val_cross_entropy", "Validation cross-entropy loss", "nats/char, best ckpt"),
    ("val_perplexity", "Perplexity (validation)", "exp(val CE), per char"),
    ("val_bits_per_char", "Bits-per-character (validation)", "val CE / ln 2"),
    ("train_perplexity", "Perplexity (train)", "exp(train CE)"),
    ("train_bits_per_char", "Bits-per-character (train)", "train CE / ln 2"),
    ("generalization_gap", "Generalization gap", "val CE - train CE (nats)"),
    ("val_top1_accuracy", "Top-1 next-character accuracy (validation)", "fraction"),
    ("train_top1_accuracy", "Top-1 next-character accuracy (train)", "fraction"),
    ("distinct_1", "Distinct-1", "word level, sampled continuations"),
    ("distinct_2", "Distinct-2", "word level, sampled continuations"),
    ("distinct_3", "Distinct-3", "word level, sampled continuations"),
    ("repeated_4gram_rate", "Repeated 4-gram rate", "mean per sample, sampled continuations"),
    ("greedy_distinct_1", "Distinct-1 (greedy)", "word level, greedy continuations"),
    ("greedy_distinct_2", "Distinct-2 (greedy)", "word level, greedy continuations"),
    ("greedy_distinct_3", "Distinct-3 (greedy)", "word level, greedy continuations"),
    ("greedy_repeated_4gram_rate", "Repeated 4-gram rate (greedy)", "mean per sample"),
    ("grad_norm_mean", "Gradient norm (mean, pre-clip)", "L2"),
    ("grad_norm_p99", "Gradient norm (p99, pre-clip)", "L2"),
    ("grad_norm_max", "Gradient norm (max, pre-clip)", "L2"),
    ("frac_steps_clipped", "Fraction of steps clipped", "grad norm > clip"),
    ("loss_spikes", "Loss spikes", "count; loss > 1.5 x EMA after warm-up"),
    ("nonfinite_steps", "NaN / Inf steps", "count"),
    ("params_total", "Parameter count", "trainable, tied weights counted once"),
    ("params_non_embedding", "Parameter count (non-embedding)", ""),
    ("train_tokens_per_sec", "Training tokens/sec", "chars/sec, optimizer steps only"),
    ("generation_tokens_per_sec", "Generation tokens/sec", "chars/sec, batch 1, no KV cache"),
    ("peak_gpu_allocated_mb", "Peak GPU memory (allocated)", "MB, training"),
    ("peak_gpu_reserved_mb", "Peak GPU memory (reserved)", "MB, training"),
    ("peak_cpu_rss_mb", "Peak CPU RSS", "MB, training process"),
    ("total_training_time_sec", "Total training time", "seconds, wall clock incl. per-epoch eval"),
    ("total_training_time_min", "Total training time", "minutes"),
    ("epochs", "Epochs trained", ""),
    ("best_epoch", "Best epoch (lowest val loss)", ""),
    ("hardware", "Hardware", "device used for training"),
]


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GPTCharLM(GPTConfig(**ck["model_config"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def generate_samples(model, cfg, char_to_idx, idx_to_char, device):
    g = cfg["generation"]
    prompts = read_json(repo_path(g["prompts_file"]))["prompts"]
    gen = torch.Generator(device=device).manual_seed(cfg["seed"])
    samples, new_tokens, gen_time = [], 0, 0.0
    modes = ([("greedy", None)] if g.get("include_greedy", True) else []) + \
            [(f"temp{g['temperature']}", i) for i in range(g["num_samples_per_prompt"])]
    for prompt in prompts:
        ids = torch.tensor([encode(prompt, char_to_idx)], device=device)
        for mode, i in modes:
            sync(device)
            t0 = time.time()
            out = model.generate(ids, g["max_new_tokens"], temperature=g["temperature"], top_k=g.get("top_k"),
                                 greedy=(mode == "greedy"), eos_id=EOS_ID, generator=gen)
            sync(device)
            gen_time += time.time() - t0
            cont_ids = out[0, ids.size(1):].tolist()
            new_tokens += len(cont_ids)
            ended = EOS_ID in cont_ids
            samples.append({"prompt": prompt, "mode": mode, "sample_idx": i,
                            "continuation": decode(cont_ids, idx_to_char, stop_at_eos=True),
                            "ended_with_eos": ended, "new_chars": len(cont_ids)})
    return samples, new_tokens / gen_time


def plot_curves(raw_logs, out_dir):
    steps = load_step_log(raw_logs / "train_steps.csv")
    epochs = np.atleast_1d(np.genfromtxt(raw_logs / "epochs.csv", delimiter=",", names=True))
    s, loss = steps["step"], steps["loss"]
    w = max(1, min(200, len(loss) // 20))
    smooth = np.convolve(loss, np.ones(w) / w, mode="valid")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(s, loss, color="#9ab", alpha=0.35, lw=0.6, label="train loss (per step)")
    ax.plot(s[w - 1:], smooth, color="#1f5aa6", lw=1.4, label=f"train loss ({w}-step moving avg)")
    ax.plot(epochs["step"], epochs["train_loss_eval"], "o-", color="#2a8f4a", label="train loss (eval mode, per epoch)")
    ax.plot(epochs["step"], epochs["val_loss"], "s-", color="#c0392b", label="validation loss (per epoch)")
    for e, st in zip(epochs["epoch"], epochs["step"]):
        ax.axvline(st, color="#ddd", lw=0.6, zorder=0)
    ax.set(xlabel="optimizer step", ylabel="cross-entropy (nats / char)", title="Training and validation loss")
    ax.set_ylim(bottom=max(0, min(loss.min(), epochs["val_loss"].min()) - 0.1),
                top=min(loss.max(), float(np.percentile(loss, 99.5)) + 0.3))
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(s, steps["grad_norm"], lw=0.5, color="#555")
    ax.set(xlabel="optimizer step", ylabel="L2 grad norm (pre-clip)", title="Gradient norm", yscale="log")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "grad_norm.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(s, steps["lr"], color="#7d3c98")
    ax.set(xlabel="optimizer step", ylabel="learning rate", title="LR schedule (linear warm-up + cosine decay)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "lr_schedule.png", dpi=150)
    plt.close(fig)


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 1e5 else f"{v:,.0f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def write_samples_md(samples, path, run_id):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Generated samples - {run_id}\n\nPrompt in **bold**, model continuation after it. "
                "`[EOS]` = the model ended the story itself.\n\n")
        for s in samples:
            tail = " `[EOS]`" if s["ended_with_eos"] else ""
            body = s["continuation"].replace("\n", "\n> ")
            f.write(f"### {s['prompt']!r} - {s['mode']}"
                    f"{'' if s['sample_idx'] is None else ' #' + str(s['sample_idx'])}\n\n"
                    f"> **{s['prompt']}**{body}{tail}\n\n")


def update_results_md(metrics, run_id):
    path = MEMBER_DIR / "results.md"
    if not path.exists():
        return
    lines = [f"_Auto-generated by `src/evaluate.py` from run `{run_id}` "
             f"(source: `outputs/{run_id}/metrics_report.csv`)._", "",
             "| Metric | Value | Notes |", "|---|---|---|"]
    lines += [f"| {name} | {_fmt(metrics[k])} | {note} |" for k, name, note in REPORT_ROWS if k in metrics]
    text = path.read_text(encoding="utf-8")
    new = re.sub(r"(<!-- METRICS:START -->\n).*?(\n<!-- METRICS:END -->)",
                 lambda m: m.group(1) + "\n".join(lines) + m.group(2), text, flags=re.S)
    new = re.sub(r"outputs/[^/()\s]+/loss_curves\.png", f"outputs/{run_id}/loss_curves.png", new)
    path.write_text(new, encoding="utf-8")


def evaluate(cfg: dict, run_id: str, promote: bool | None = None) -> dict:
    run_id = latest_run_id(cfg["run_name"]) if run_id == "latest" else run_id
    dirs = run_dirs(run_id)
    log = get_logger(f"eval.{run_id}", dirs["raw_logs"] / "eval.log")
    device = get_device(cfg.get("device", "auto"))
    precision = resolve_precision(cfg["training"].get("precision", "auto"), device)

    model, ck = load_model(dirs["checkpoints"] / "best_model.pt", device)
    processed = repo_path(ck["processed_dir"])
    meta = read_json(processed / "meta.json")
    char_to_idx, idx_to_char = load_vocab(processed)
    dtype = np.uint8 if meta["dtype"] == "uint8" else np.uint16
    T = model.cfg.block_size
    log.info(f"EVAL run {run_id} best_model epoch={ck['epoch']} step={ck['step']} device={device}")

    eb = cfg["training"].get("eval_batch_size", 128)
    val_loss, val_acc = evaluate_stream(model, CharStream(processed / "val.bin", T, device, dtype), eb, device, precision)
    tr_loss, tr_acc = evaluate_stream(model, CharStream(processed / "train.bin", T, device, dtype), eb, device,
                                      precision, max_seqs=cfg.get("eval", {}).get("final_train_max_seqs"))
    log.info(f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f}")

    samples, gen_tps = generate_samples(model, cfg, char_to_idx, idx_to_char, device)
    sampled = [s["continuation"] for s in samples if s["mode"] != "greedy"]
    greedy = [s["continuation"] for s in samples if s["mode"] == "greedy"]
    gm = generation_metrics(sampled)
    gg = {f"greedy_{k}": v for k, v in generation_metrics(greedy).items()} if greedy else {}

    summary = read_json(dirs["outputs"] / "train_summary.json")
    epochs = list(csv.DictReader(open(dirs["raw_logs"] / "epochs.csv", encoding="utf-8")))
    hw = summary["hardware"]
    metrics = {
        **loss_metrics(tr_loss, val_loss),
        "val_top1_accuracy": val_acc, "train_top1_accuracy": tr_acc,
        "train_loss_running_last_epoch": float(epochs[-1]["train_loss_running"]),
        **gm, **gg,
        "generation_tokens_per_sec": gen_tps,
        **{k: summary[k] for k in ("grad_norm_mean", "grad_norm_p99", "grad_norm_max", "frac_steps_clipped",
                                   "loss_spikes", "nonfinite_steps", "params_total", "params_non_embedding",
                                   "train_tokens_per_sec", "total_training_time_sec", "epochs", "best_epoch")},
        "total_training_time_min": summary["total_training_time_sec"] / 60,
        **{k: v for k, v in summary.items() if k.startswith("peak_")},
        "hardware": hw.get("gpu_name") or hw.get("cpu_model"),
        "run_id": run_id, "best_checkpoint": rel(dirs["checkpoints"] / "best_model.pt"),
        "num_generated_samples": len(samples),
    }
    log.info("metrics: " + ", ".join(f"{k}={_fmt(v)}" for k, v in metrics.items()))

    out = dirs["outputs"]
    write_json(out / "metrics.json", metrics)
    write_json(out / "samples.json", samples)
    write_samples_md(samples, out / "samples.md", run_id)
    with open(out / "metrics_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "notes"])
        for k, name, note in REPORT_ROWS:
            if k in metrics:
                w.writerow([name, metrics[k], note])
        w.writerow(["Run id", run_id, "evidence: reproducibility/raw_logs/... and outputs/<run_id>/"])
        w.writerow(["Best checkpoint", metrics["best_checkpoint"], f"epoch {ck['epoch']}"])
    plot_curves(dirs["raw_logs"], out)

    promote = (not cfg.get("smoke", False)) if promote is None else promote
    if promote:
        shutil.copy(out / "metrics_report.csv", MEMBER_DIR / "metrics_report.csv")
        update_results_md(metrics, run_id)
        log.info("promoted metrics_report.csv to member folder and refreshed results.md table")
    update_manifest(run_id, evaluation={"metrics": rel(out / "metrics.json"),
                                        "metrics_report": rel(out / "metrics_report.csv"),
                                        "samples": rel(out / "samples.json"),
                                        "plots": [rel(p) for p in sorted(out.glob("*.png"))],
                                        "eval_peak_memory": peak_memory_mb(device)})
    log.info(f"EVAL DONE -> {rel(out)}")
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Task 1 final evaluation")
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--run-id", default="latest")
    p.add_argument("--promote", action=argparse.BooleanOptionalAction, default=None)
    a = p.parse_args()
    evaluate(load_config(a.config, a.set), a.run_id, a.promote)


if __name__ == "__main__":
    main()
