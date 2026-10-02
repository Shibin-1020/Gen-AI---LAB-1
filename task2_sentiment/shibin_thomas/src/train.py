"""Task 2.2 -- train one sentiment model (called once per model by run_pipeline.py).

Objective: binary cross-entropy on one logit. AdamW, linear warm-up + cosine decay, gradient clipping,
early stopping on validation macro-F1 (patience in epochs); the best-validation epoch is kept.

Per model, everything is logged (append-only) under
reproducibility/raw_logs/task2_sentiment/shibin_thomas/<run_id>/<model_name>/:
    train.log, steps.csv (every optimizer step), epochs.csv (every epoch)
and the weights go to task2_sentiment/shibin_thomas/checkpoints/<run_id>/<model_name>/best_model.pt.
"""
from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from models import build_model, count_params
from utils import (get_device, get_logger, hardware_info, peak_memory_mb, read_json, rel, repo_path, run_dirs,
                   set_seed, sync, write_json)


# --------------------------------------------------------------------------- config
def load_model_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Model YAML + the data YAML it points to (cfg['data']). Overrides: 'a.b=value'
    ('data_config=...' switches the data file; 'data.x=...' edits it)."""
    overrides = list(overrides or [])
    with open(repo_path(path), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for ov in overrides:
        k, _, v = ov.partition("=")
        if k.strip() == "data_config":
            cfg["data_config"] = yaml.safe_load(v)
    with open(repo_path(cfg["data_config"]), encoding="utf-8") as f:
        cfg["data"] = yaml.safe_load(f)
    for ov in overrides:
        k, _, v = ov.partition("=")
        if k.strip() == "data_config":
            continue
        node, keys = cfg, k.strip().split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = yaml.safe_load(v)
    cfg["_config_path"] = Path(repo_path(path)).resolve().relative_to(repo_path(".").resolve()).as_posix()
    cfg["_overrides"] = overrides
    return cfg


# --------------------------------------------------------------------------- data
class Split:
    def __init__(self, processed: Path, name: str, device: torch.device):
        self.X = torch.from_numpy(np.load(processed / f"X_{name}.npy")).to(device)          # int32 ids
        self.L = torch.from_numpy(np.load(processed / f"len_{name}.npy")).long().to(device)
        self.y = torch.from_numpy(np.load(processed / f"y_{name}.npy").astype(np.float32)).to(device)
        self.n = self.y.numel()

    def batches(self, batch_size: int, gen: torch.Generator | None = None, shuffle: bool = False):
        order = torch.randperm(self.n, generator=gen) if shuffle else torch.arange(self.n)
        order = order.to(self.X.device)
        for i in range(0, self.n, batch_size):
            idx = order[i:i + batch_size]
            L = self.L[idx]
            T = int(L.max())                                   # trim padding beyond the longest review
            yield self.X[idx, :T].long(), L, self.y[idx]


@torch.no_grad()
def predict(model, split: Split, batch_size: int) -> tuple[np.ndarray, float]:
    """P(positive) for every example of the split, and inference examples/sec."""
    model.eval()
    sync(split.X.device)
    t0 = time.perf_counter()
    probs = [torch.sigmoid(model(x, L).float()) for x, L, _ in split.batches(batch_size)]
    sync(split.X.device)
    dt = time.perf_counter() - t0
    model.train()
    return torch.cat(probs).cpu().numpy(), split.n / max(dt, 1e-9)


def quick_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import f1_score, roc_auc_score
    yhat = (p >= 0.5).astype(int)
    eps = 1e-7
    loss = float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
    return {"loss": loss, "acc": float((yhat == y).mean()), "f1_macro": float(f1_score(y, yhat, average="macro")),
            "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")}


def _append_csv(path: Path, row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


# --------------------------------------------------------------------------- training
def train_model(cfg: dict, run_id: str, processed: Path) -> dict:
    name, tcfg = cfg["name"], cfg["training"]
    dirs = run_dirs(run_id)
    log_dir, ck_dir, out_dir = dirs["raw_logs"] / name, dirs["checkpoints"] / name, dirs["outputs"] / name
    for d in (log_dir, ck_dir, out_dir):
        d.mkdir(parents=True, exist_ok=True)
    log = get_logger(f"train.{run_id}.{name}", log_dir / "train.log")
    device = get_device(cfg.get("device", "auto"))
    set_seed(cfg["seed"])

    train, val = Split(processed, "train", device), Split(processed, "val", device)
    vocab_size = read_json(processed / "vocab.json")["vocab_size"]
    model = build_model(cfg["model"], vocab_size).to(device)
    n_params = count_params(model)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "emb" not in n]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2 or "emb" in n]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": tcfg["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}], lr=tcfg["lr"])
    B = tcfg["batch_size"]
    steps_per_epoch = math.ceil(train.n / B)
    total = steps_per_epoch * tcfg["epochs"]
    warmup = max(1, int(tcfg.get("warmup_frac", 0.03) * total))
    min_ratio = tcfg.get("min_lr_ratio", 0.05)

    def lr_at(step):
        if step < warmup:
            return tcfg["lr"] * (step + 1) / warmup
        prog = min(1.0, (step - warmup) / max(1, total - warmup))
        return tcfg["lr"] * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))

    hw = hardware_info(device)
    log.info(f"START model {name} run {run_id} config={cfg['_config_path']} overrides={cfg['_overrides']}")
    log.info(f"model={cfg['model']} params={n_params:,} vocab={vocab_size:,}")
    log.info(f"training={tcfg}")
    log.info(f"device={device} hardware={hw}")
    log.info(f"train={train.n:,} val={val.n:,} batch={B} steps/epoch={steps_per_epoch:,} max_steps={total:,} "
             f"warmup={warmup}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    gen = torch.Generator().manual_seed(cfg["seed"])
    best_f1, best_epoch, bad, step = -1.0, 0, 0, 0
    train_time, seen, wall0 = 0.0, 0, time.perf_counter()
    nonfinite = 0
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        ep_loss, ep_n, ep_t0 = 0.0, 0, time.perf_counter()
        for x, L, y in train.batches(B, gen, shuffle=True):
            lr = lr_at(step)
            for g in opt.param_groups:
                g["lr"] = lr
            sync(device)
            t0 = time.perf_counter()
            loss = F.binary_cross_entropy_with_logits(model(x, L).float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"]).item()
            lv = loss.item()
            if math.isfinite(lv) and math.isfinite(gn):
                opt.step()
            else:
                nonfinite += 1
                log.warning(f"non-finite loss/grad at step {step}; update skipped")
            sync(device)
            train_time += time.perf_counter() - t0
            seen += y.numel()
            ep_loss += lv * y.numel()
            ep_n += y.numel()
            _append_csv(log_dir / "steps.csv", {"step": step, "epoch": epoch, "lr": f"{lr:.6e}",
                                                "loss": f"{lv:.5f}", "grad_norm": f"{gn:.4f}"})
            if step % tcfg.get("log_interval", 100) == 0:
                log.info(f"epoch {epoch} step {step}/{total} lr {lr:.2e} loss {lv:.4f} grad_norm {gn:.3f}")
            step += 1

        p_val, _ = predict(model, val, tcfg.get("eval_batch_size", 1024))
        vm = quick_metrics(val.y.cpu().numpy(), p_val)
        ep_time = time.perf_counter() - ep_t0
        _append_csv(log_dir / "epochs.csv", {
            "epoch": epoch, "train_loss": f"{ep_loss / ep_n:.5f}", "val_loss": f"{vm['loss']:.5f}",
            "val_acc": f"{vm['acc']:.5f}", "val_f1_macro": f"{vm['f1_macro']:.5f}",
            "val_roc_auc": f"{vm['roc_auc']:.5f}", "epoch_time_sec": f"{ep_time:.1f}"})
        log.info(f"EPOCH {epoch} | train_loss {ep_loss / ep_n:.4f} | val_loss {vm['loss']:.4f} | "
                 f"val_acc {vm['acc']:.4f} | val_f1_macro {vm['f1_macro']:.4f} | val_auc {vm['roc_auc']:.4f} | "
                 f"{ep_time:.0f}s")
        if vm["f1_macro"] > best_f1:
            best_f1, best_epoch, bad = vm["f1_macro"], epoch, 0
            torch.save({"model": model.state_dict(), "model_config": cfg["model"], "vocab_size": vocab_size,
                        "name": name, "epoch": epoch, "val_f1_macro": best_f1, "run_id": run_id,
                        "processed_dir": rel(processed)}, ck_dir / "best_model.pt")
            log.info(f"new best val macro-F1 {best_f1:.4f} -> {rel(ck_dir / 'best_model.pt')}")
        else:
            bad += 1
            if bad >= tcfg.get("patience", 2):
                log.info(f"early stopping: no val macro-F1 improvement for {bad} epochs")
                break

    summary = {"name": name, "run_id": run_id, "params": n_params, "epochs_run": epoch, "best_epoch": best_epoch,
               "best_val_f1_macro": best_f1, "training_time_sec": time.perf_counter() - wall0,
               "pure_step_time_sec": train_time, "train_examples_per_sec": seen / max(train_time, 1e-9),
               "nonfinite_steps": nonfinite, "device": str(device), "hardware": hw, **peak_memory_mb(device),
               "config": {k: v for k, v in cfg.items() if not k.startswith("_")}}
    write_json(out_dir / "train_summary.json", summary)
    log.info(f"DONE {name} | best val macro-F1 {best_f1:.4f} @ epoch {best_epoch} | "
             f"{summary['training_time_sec']:.0f}s | {summary['train_examples_per_sec']:,.0f} ex/s | "
             f"peak {peak_memory_mb(device)}")
    return summary
