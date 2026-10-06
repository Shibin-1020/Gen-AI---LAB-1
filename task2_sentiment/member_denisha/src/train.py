"""Train three from-scratch Yelp Polarity classifiers and write reproducible metrics.

Expected input is CSV/JSONL with a text column and a binary label column. No
pretrained embeddings or language models are used.
Run from this directory: python src/train.py --config config.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
import yaml


TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
STOPWORDS = {"a", "an", "the", "and", "or", "but", "if", "then", "than", "to", "of", "in", "on", "for", "with", "as", "at", "by", "from", "is", "are", "was", "were", "be", "been", "it", "this", "that", "these", "those", "i", "we", "you", "he", "she", "they", "them", "our", "your", "their"}


def stem_token(token):
    # Lightweight deterministic stemming fallback; sentiment negations are retained.
    for suffix in ("ingly", "edly", "ing", "ed", "ies", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix): return token[:-len(suffix)]
    return token


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def device_from(value):
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else ("cpu" if value == "auto" else value))


def autocast_context(device, cfg):
    if device.type != "cuda" or not cfg.get("amp", True):
        return nullcontext()
    dtype = torch.bfloat16 if cfg.get("precision", "bfloat16") == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def evaluate_loader(model, loader, loss_fn, device, cfg, return_probs=False):
    model.eval(); total_loss = 0.0; total = 0; probs = []; labels = []
    with torch.no_grad():
        for x, y, lengths in loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True); lengths = lengths.to(device, non_blocking=True)
            with autocast_context(device, cfg):
                logits = model(x, lengths); loss = loss_fn(logits, y)
            total_loss += loss.item() * len(y); total += len(y)
            if return_probs:
                probs.extend(torch.sigmoid(logits).float().cpu().numpy()); labels.extend(y.cpu().numpy())
    if return_probs: return total_loss / max(total, 1), np.asarray(labels), np.asarray(probs)
    return total_loss / max(total, 1)


def normalize(text, remove_stopwords=True, stemming=True):
    tokens = TOKEN_RE.findall(str(text).lower())
    if remove_stopwords: tokens = [t for t in tokens if t not in STOPWORDS]
    return [stem_token(t) for t in tokens] if stemming else tokens


class Vocab:
    def __init__(self, texts, max_size, remove_stopwords=True, stemming=True):
        self.remove_stopwords, self.stemming = remove_stopwords, stemming
        counts = Counter(tok for text in texts for tok in normalize(text, remove_stopwords, stemming))
        self.itos = ["<pad>", "<unk>"] + [w for w, _ in counts.most_common(max_size - 2)]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, text, max_length):
        ids = [self.stoi.get(w, 1) for w in normalize(text, self.remove_stopwords, self.stemming)][:max_length]
        return ids + [0] * (max_length - len(ids))


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length):
        self.x = torch.tensor([vocab.encode(x, max_length) for x in texts], dtype=torch.long)
        self.y = torch.tensor(labels, dtype=torch.float32)
        self.lengths = (self.x != 0).sum(1).clamp_min(1)

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x[i], self.y[i], self.lengths[i]


class BoWMLP(nn.Module):
    def __init__(self, vocab_size, dim, hidden, dropout):
        super().__init__(); self.emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x, lengths):
        mask = (x != 0).unsqueeze(-1); pooled = (self.emb(x) * mask).sum(1) / lengths.unsqueeze(-1).float(); return self.net(pooled).squeeze(-1)


class TextCNN(nn.Module):
    def __init__(self, vocab_size, dim, channels, kernels, dropout):
        super().__init__(); self.emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(dim, channels, k) for k in kernels]); self.drop = nn.Dropout(dropout); self.fc = nn.Linear(channels * len(kernels), 1)

    def forward(self, x, lengths):
        h = self.emb(x).transpose(1, 2); pooled = [torch.relu(c(h)).amax(-1) for c in self.convs]; return self.fc(self.drop(torch.cat(pooled, 1))).squeeze(-1)


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, dim, hidden, layers, dropout):
        super().__init__(); self.emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.gru = nn.GRU(dim, hidden, layers, batch_first=True, dropout=dropout if layers > 1 else 0.0, bidirectional=True); self.fc = nn.Linear(hidden * 2, 1)

    def forward(self, x, lengths):
        h, _ = self.gru(self.emb(x)); idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, h.size(-1)); return self.fc(h.gather(1, idx).squeeze(1)).squeeze(-1)


def build_model(name, cfg, vocab_size):
    c = cfg["models"][name]
    if c["type"] == "bow_mlp": return BoWMLP(vocab_size, c["embedding_dim"], c["hidden_dim"], c["dropout"])
    if c["type"] == "text_cnn": return TextCNN(vocab_size, c["embedding_dim"], c["channels"], c["kernel_sizes"], c["dropout"])
    if c["type"] == "gru": return GRUClassifier(vocab_size, c["embedding_dim"], c["hidden_dim"], c["layers"], c["dropout"])
    raise ValueError(f"Unknown model type: {c['type']}")


def expected_calibration_error(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p <= hi if hi == 1 else p < hi)
        if m.any(): total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def bootstrap_ci(y, p, metric, iterations=1000, seed=0):
    rng = np.random.default_rng(seed); vals = []
    for _ in range(iterations):
        idx = rng.integers(0, len(y), len(y)); vals.append(metric(y[idx], p[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def mcnemar(y, pred_a, pred_b):
    b = int(((pred_a == y) & (pred_b != y)).sum()); c = int(((pred_a != y) & (pred_b == y)).sum())
    stat = (abs(b - c) - 1) ** 2 / max(b + c, 1); return {"b": b, "c": c, "chi2": float(stat), "p_value": float(chi2.sf(stat, 1))}


def metrics(y, p, slices):
    pred = (p >= 0.5).astype(int); cm = confusion_matrix(y, pred, labels=[0, 1]).tolist()
    out = {
        "accuracy": float(accuracy_score(y, pred)), "precision_macro": float(precision_score(y, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y, pred, average="macro", zero_division=0)), "f1_macro": float(f1_score(y, pred, average="macro", zero_division=0)),
        "precision_micro": float(precision_score(y, pred, average="micro", zero_division=0)), "recall_micro": float(recall_score(y, pred, average="micro", zero_division=0)),
        "f1_micro": float(f1_score(y, pred, average="micro", zero_division=0)), "precision_weighted": float(precision_score(y, pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y, pred, average="weighted", zero_division=0)), "f1_weighted": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)), "mcc": float(matthews_corrcoef(y, pred)),
        "brier_score": float(brier_score_loss(y, p)), "expected_calibration_error": expected_calibration_error(y, p), "confusion_matrix": cm,
        "accuracy_ci95": bootstrap_ci(y, p, lambda a, q: accuracy_score(a, q >= .5)),
        "macro_f1_ci95": bootstrap_ci(y, p, lambda a, q: f1_score(a, q >= .5, average="macro", zero_division=0)),
        "mcc_ci95": bootstrap_ci(y, p, lambda a, q: matthews_corrcoef(a, q >= .5)),
    }
    out["slice_metrics"] = {}
    for name, mask in slices.items():
        if mask.sum() >= 2: out["slice_metrics"][name] = {"n": int(mask.sum()), "macro_f1": float(f1_score(y[mask], pred[mask], average="macro", zero_division=0)), "error_rate": float((pred[mask] != y[mask]).mean())}
    return out


def load_frame(path):
    if path.endswith(".jsonl"):
        return pd.DataFrame([json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()])
    return pd.read_csv(path)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); ap.add_argument("--checkpoint-dir", default=None); ap.add_argument("--resume-latest", action="store_true"); args = ap.parse_args(); cfg = yaml.safe_load(Path(args.config).read_text()); seed_everything(cfg["seed"])
    root = Path.cwd(); out = root / "outputs"; ckpt = Path(args.checkpoint_dir or cfg.get("checkpoint_dir", "checkpoints")); ckpt = ckpt if ckpt.is_absolute() else root / ckpt; out.mkdir(exist_ok=True); ckpt.mkdir(parents=True, exist_ok=True)
    df = load_frame(cfg["data_path"]); text_col = next((c for c in ["text", "review", "content"] if c in df), None); label_col = next((c for c in ["label", "sentiment", "target"] if c in df), None)
    if text_col is None or label_col is None: raise ValueError("Dataset needs a text/review/content column and label/sentiment/target column")
    df = df[[text_col, label_col]].dropna(); df[text_col] = df[text_col].astype(str); df[label_col] = df[label_col].astype(int)
    # Accommodate Yelp labels encoded as 1/2 as well as 0/1.
    if set(df[label_col].unique()) == {1, 2}: df[label_col] -= 1
    train, test = train_test_split(df, test_size=0.2, random_state=cfg["seed"], stratify=df[label_col]); train, val = train_test_split(train, test_size=cfg["validation_fraction"], random_state=cfg["seed"], stratify=train[label_col])
    vocab = Vocab(train[text_col].tolist(), cfg["max_vocab_size"], cfg.get("remove_stopwords", True), cfg.get("stemming", True)); device = device_from(cfg["device"]); results = {}; all_preds = {}
    slices = {"short": test[text_col].str.split().str.len().to_numpy() < 50, "medium": ((test[text_col].str.split().str.len().to_numpy() >= 50) & (test[text_col].str.split().str.len().to_numpy() < 150)), "long": test[text_col].str.split().str.len().to_numpy() >= 150}
    train_ds = ReviewDataset(train[text_col].tolist(), train[label_col].to_numpy(), vocab, cfg["max_length"]); val_ds = ReviewDataset(val[text_col].tolist(), val[label_col].to_numpy(), vocab, cfg["max_length"]); test_ds = ReviewDataset(test[text_col].tolist(), test[label_col].to_numpy(), vocab, cfg["max_length"])
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    loader_kwargs = {"pin_memory": bool(device.type == "cuda" and cfg.get("pin_memory", True))}
    if cfg.get("num_workers", 0) > 0:
        loader_kwargs.update(num_workers=cfg["num_workers"], persistent_workers=True, prefetch_factor=cfg.get("prefetch_factor", 4))
    eval_loader_kwargs = dict(loader_kwargs)
    eval_loader_kwargs["num_workers"] = cfg.get("eval_workers", max(0, cfg.get("num_workers", 0)))
    if eval_loader_kwargs["num_workers"] == 0:
        eval_loader_kwargs.pop("persistent_workers", None); eval_loader_kwargs.pop("prefetch_factor", None)
    y_test = test[label_col].to_numpy(); base_pred = None
    for name in cfg["models"]:
        model = build_model(name, cfg, len(vocab.itos)).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=cfg.get("lr_factor", 0.5), patience=cfg.get("lr_patience", 1), min_lr=cfg.get("min_learning_rate", 1e-6))
        loss_fn = nn.BCEWithLogitsLoss(); loader = DataLoader(train_ds, cfg["batch_size"], shuffle=True, **loader_kwargs); val_loader = DataLoader(val_ds, cfg["batch_size"], shuffle=False, **eval_loader_kwargs); test_loader = DataLoader(test_ds, cfg["batch_size"], shuffle=False, **eval_loader_kwargs)
        start = time.time(); rows = []; nan_count = 0; best_val = float("inf"); best_state = None; stale_epochs = 0; start_epoch = 1
        use_scaler = device.type == "cuda" and cfg.get("amp", True) and cfg.get("precision", "bfloat16") == "float16"
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
        latest = ckpt / f"{name}_latest.pt"
        if args.resume_latest and latest.exists():
            state = torch.load(latest, map_location=device); model.load_state_dict(state["model"]); opt.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
            rows = state.get("history", []); best_val = float(state.get("best_validation_loss", float("inf"))); best_state = state.get("best_model"); start_epoch = int(state.get("epoch", 0)) + 1
            if best_state is not None: best_state = {k: v.cpu() for k, v in best_state.items()}
            print(json.dumps({"model": name, "resumed_from": str(latest), "start_epoch": start_epoch}), flush=True)
        for epoch in range(start_epoch, cfg["epochs"] + 1):
            model.train(); total = 0.0
            for x, y, lengths in loader:
                x, y, lengths = x.to(device, non_blocking=True), y.to(device, non_blocking=True), lengths.to(device, non_blocking=True); opt.zero_grad(set_to_none=True)
                with autocast_context(device, cfg): loss = loss_fn(model(x, lengths), y)
                nan_count += int(not torch.isfinite(loss).item())
                if use_scaler:
                    scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update()
                else:
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                total += loss.item()
            train_loss = total / max(1, len(loader)); val_loss = evaluate_loader(model, val_loader, loss_fn, device, cfg); scheduler.step(val_loss); current_lr = opt.param_groups[0]["lr"]
            if val_loss < best_val - cfg.get("min_delta", 1e-4): best_val = val_loss; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; stale_epochs = 0
            else: stale_epochs += 1
            rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "learning_rate": current_lr}); print(json.dumps({"model": name, **rows[-1]}), flush=True)
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch, "history": rows, "best_validation_loss": best_val, "best_model": best_state, "config": cfg, "vocab": vocab.stoi}, latest)
            if stale_epochs >= cfg.get("early_stopping_patience", 3): print(json.dumps({"model": name, "early_stopping": True, "epoch": epoch}), flush=True); break
        if best_state is not None: model.load_state_dict(best_state)
        _, _, probs = evaluate_loader(model, test_loader, loss_fn, device, cfg, return_probs=True)
        all_preds[name] = probs; elapsed = time.time() - start; results[name] = metrics(y_test, probs, slices); results[name].update({"parameter_count": sum(p.numel() for p in model.parameters()), "training_time_seconds": elapsed, "examples_per_second": len(train_ds) * len(rows) / max(elapsed, 1e-9), "peak_memory_mb": (torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0), "gradient_nan_count": nan_count, "device": str(device), "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU", "best_validation_loss": best_val, "history": rows})
        torch.save({"model": model.state_dict(), "vocab": vocab.stoi, "config": cfg, "best_validation_loss": best_val}, ckpt / f"{name}.pt")
        (out / f"{name}_history.json").write_text(json.dumps(rows, indent=2))
        if base_pred is None: base_pred = (probs >= .5).astype(int)
    for name, probs in all_preds.items():
        pred = (probs >= .5).astype(int); results[name]["mcnemar_vs_baseline"] = None if name == next(iter(all_preds)) else mcnemar(y_test, base_pred, pred)
    error_rows = []
    for i, (text, y) in enumerate(zip(test[text_col], y_test)):
        p = all_preds[next(iter(all_preds))][i]; pred = int(p >= .5); confidence = abs(p - .5); typ = "false_positive" if pred == 1 and y == 0 else "false_negative" if pred == 0 and y == 1 else "correct"
        if typ != "correct":
            length = len(str(text).split()); bucket = "short" if length < 50 else "medium" if length < 150 else "long"
            error_rows.append({"index": int(test.index[i]), "text": text, "label": int(y), "prediction": pred, "probability": float(p), "error_type": typ, "confidence": float(confidence), "slice": bucket})
    fp = sorted([r for r in error_rows if r["error_type"] == "false_positive"], key=lambda r: -r["confidence"])
    fn = sorted([r for r in error_rows if r["error_type"] == "false_negative"], key=lambda r: -r["confidence"])
    near = sorted(error_rows, key=lambda r: abs(r["probability"] - .5)); slice_rows = []
    for slice_name in ["short", "medium", "long"]:
        slice_rows.extend(sorted([r for r in error_rows if r["slice"] == slice_name], key=lambda r: -r["confidence"])[:5])
    selected = []; used = set()
    def add(rows, bucket, limit):
        for row in rows:
            if len([x for x in selected if x["review_bucket"] == bucket]) >= limit: break
            if row["index"] not in used: row = {**row, "review_bucket": bucket}; selected.append(row); used.add(row["index"])
    add(fp, "confident_false_positive", 5); add(fn, "confident_false_negative", 5); add(slice_rows, "slice_specific", 5); add(near, "near_threshold", 5)
    (out / "error_review.json").write_text(json.dumps(selected, indent=2))
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    scalar_rows = [{"model": k, **{m: v for m, v in val.items() if isinstance(v, (int, float))}} for k, val in results.items()]
    (out / "metrics_summary.csv").write_text(pd.DataFrame(scalar_rows).to_csv(index=False))
    report_rows = []
    for model_name, values in results.items():
        row = {"model": model_name}
        for key, value in values.items():
            if key != "history": row[key] = json.dumps(value) if isinstance(value, (dict, list)) else value
        report_rows.append(row)
    (out / "metrics_report.csv").write_text(pd.DataFrame(report_rows).to_csv(index=False))
    print(json.dumps(results, indent=2))


if __name__ == "__main__": main()
