"""Train and evaluate a character-level GPT with attention implemented from scratch.

Run from this directory with:
    python src/train.py --config config.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
import re
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import yaml


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class CharDataset(Dataset):
    """Fixed-length next-token examples with configurable stride.

    The assignment requires fixed-length autoregressive sequences, not a sample at
    every character offset. The default stride equals sequence_length, so examples
    are non-overlapping and the same character is not used in hundreds of nearly
    duplicate training windows.
    """
    def __init__(self, encoded: torch.Tensor, sequence_length: int, stride: int | None = None):
        self.data, self.sequence_length = encoded, sequence_length
        self.stride = stride or sequence_length
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        last_start = self.data.numel() - self.sequence_length - 1
        self.starts = torch.arange(0, max(0, last_start + 1), self.stride, dtype=torch.long)

    def __len__(self):
        return self.starts.numel()

    def __getitem__(self, index):
        start = int(self.starts[index])
        x = self.data[start:start + self.sequence_length]
        y = self.data[start + 1:start + self.sequence_length + 1]
        return x, y


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float, sequence_length: int):
        super().__init__()
        if dim % heads: raise ValueError("embedding_dim must be divisible by num_heads")
        self.heads, self.head_dim = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(sequence_length, sequence_length)).bool())

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (b, t, self.heads, self.head_dim)
        q = q.view(shape).transpose(1, 2); k = k.view(shape).transpose(1, 2); v = v.view(shape).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~self.mask[:t, :t], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        out = (weights @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.dropout(self.proj(out))


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, ff_dim, dropout, sequence_length):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim); self.attn = CausalSelfAttention(dim, heads, dropout, sequence_length)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, ff_dim), nn.GELU(), nn.Linear(ff_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attn(self.norm1(x)); return x + self.ff(self.norm2(x))


class GPTFromScratch(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__(); dim = cfg["embedding_dim"]; length = cfg["sequence_length"]
        self.token_embedding = nn.Embedding(vocab_size, dim); self.position_embedding = nn.Embedding(length, dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim, cfg["num_heads"], cfg["feedforward_dim"], cfg["dropout"], length) for _ in range(cfg["num_layers"])])
        self.norm = nn.LayerNorm(dim); self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        _, t = idx.shape; pos = torch.arange(t, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)[None, :, :]
        for block in self.blocks: x = block(x)
        logits = self.lm_head(self.norm(x)); loss = None
        if targets is not None: loss = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            context = idx[:, -self.position_embedding.num_embeddings:]
            logits, _ = self(context); logits = logits[:, -1, :] / max(temperature, 1e-5)
            idx = torch.cat((idx, torch.multinomial(torch.softmax(logits, -1), 1)), dim=1)
        return idx


def choose_device(value): return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else ("cpu" if value == "auto" else value))


def autocast_context(device, cfg):
    if device.type != "cuda" or not cfg.get("amp", True): return nullcontext()
    requested = cfg.get("precision", "float16").lower()
    # Colab T4 is a common target for this lab and does not provide native
    # bfloat16 support. Fall back to float16 instead of failing at runtime.
    use_bfloat16 = requested == "bfloat16" and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bfloat16 else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def evaluate(model, loader, device, cfg):
    model.eval(); total = n = correct = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast_context(device, cfg): logits, loss = model(x, y)
            total += loss.item() * y.numel(); n += y.numel(); correct += (logits.argmax(-1) == y).sum().item()
    loss = total / max(n, 1); return loss, correct / max(n, 1)


def generation_metrics(sample):
    tokens = list(sample)
    def distinct(n):
        grams = [tuple(tokens[i:i+n]) for i in range(max(0, len(tokens)-n+1))]
        return len(set(grams)) / max(1, len(grams))
    grams4 = [tuple(tokens[i:i+4]) for i in range(max(0, len(tokens)-3))]
    repeated = sum(1 for i, gram in enumerate(grams4) if gram in grams4[:i]) / max(1, len(grams4))
    return {"distinct_1": distinct(1), "distinct_2": distinct(2), "distinct_3": distinct(3), "repeated_4gram_rate": repeated}


def load_stories(path: Path):
    """Read TinyStories JSONL, JSON, or plain text while preserving story units."""
    if path.suffix == ".jsonl":
        stories = []
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if line.strip():
                    item = json.loads(line)
                    stories.append(str(item.get("text", item) if isinstance(item, dict) else item))
                    if len(stories) % 10000 == 0:
                        print(json.dumps({"loaded_stories": len(stories), "last_line": line_number}), flush=True)
        return stories
    if path.suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [str(x.get("text", x)) if isinstance(x, dict) else str(x) for x in raw]
    return [x for x in path.read_text(encoding="utf-8").split("\n\n") if x.strip()]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); ap.add_argument("--checkpoint-dir", default=None); ap.add_argument("--resume-latest", action="store_true"); args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text()); seed_everything(cfg["seed"])
    root = Path.cwd(); out = root / "outputs"; ckpt = Path(args.checkpoint_dir or cfg.get("checkpoint_dir", "checkpoints")); ckpt = ckpt if ckpt.is_absolute() else root / ckpt; out.mkdir(exist_ok=True); ckpt.mkdir(parents=True, exist_ok=True)
    previous_metrics = {}
    previous_metrics_path = out / "metrics.csv"
    if previous_metrics_path.exists():
        try:
            with previous_metrics_path.open(newline="") as stream:
                previous_metrics = next(csv.DictReader(stream), {})
        except (OSError, StopIteration, csv.Error):
            previous_metrics = {}
    if args.resume_latest and not cfg.get("resume_from"):
        latest = ckpt / "gpt_from_scratch_latest.pt"
        if latest.exists(): cfg["resume_from"] = str(latest)
    stories = load_stories(Path(cfg["data_path"]))
    required_stories = cfg["train_samples"] + cfg["validation_samples"]
    if len(stories) < required_stories:
        raise ValueError(f"Need at least {required_stories} stories, found {len(stories)}")
    # Keep the member's split explicit and reproducible: first shuffle story indices,
    # then take exactly the requested train and validation story counts.
    rng = random.Random(cfg["seed"]); indices = list(range(len(stories))); rng.shuffle(indices)
    train_stories = [stories[i] for i in indices[:cfg["train_samples"]]]
    val_stories = [stories[i] for i in indices[cfg["train_samples"]:required_stories]]
    text = "\n".join(train_stories + val_stories)
    chars = sorted(set(text)); c2i = {c:i for i,c in enumerate(chars)}; i2c = {i:c for c,i in c2i.items()}
    train_data = torch.tensor([c2i[c] for c in "\n".join(train_stories)], dtype=torch.long)
    val_data = torch.tensor([c2i[c] for c in "\n".join(val_stories)], dtype=torch.long)
    stride = cfg.get("stride", cfg["sequence_length"])
    train_dataset = CharDataset(train_data, cfg["sequence_length"], stride)
    val_dataset = CharDataset(val_data, cfg["sequence_length"], stride)
    device = choose_device(cfg["device"])
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    loader_kwargs = {"num_workers": cfg["num_workers"], "pin_memory": bool(cfg.get("pin_memory", True) and device.type == "cuda")}
    if cfg["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    train_loader = DataLoader(train_dataset, cfg["batch_size"], shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, cfg["batch_size"], shuffle=False, **loader_kwargs)
    print(json.dumps({"device": str(choose_device(cfg["device"])), "train_stories": len(train_stories), "validation_stories": len(val_stories), "train_characters": len(train_data), "validation_characters": len(val_data), "stride": stride, "train_sequences": len(train_dataset), "validation_sequences": len(val_dataset), "batches_per_epoch": len(train_loader), "total_training_batches": len(train_loader) * cfg["epochs"]}), flush=True)
    model = GPTFromScratch(len(chars), cfg).to(device)
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"], betas=tuple(cfg.get("betas", [0.9, 0.999])))
    total_steps = cfg["epochs"] * max(1, len(train_loader)); warmup = cfg["warmup_steps"]; train_tokens = 0; nan_count = 0
    def lr_lambda(step): return min((step + 1) / max(1, warmup), 1.0) * max(cfg.get("min_lr_ratio", 0.1), 0.5 * (1 + math.cos(math.pi * step / max(1, total_steps))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda); history = []; start = time.time(); step = 0
    use_scaler = device.type == "cuda" and cfg.get("amp", True) and not (cfg.get("precision", "float16").lower() == "bfloat16" and torch.cuda.is_bf16_supported())
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    start_epoch = 1
    if cfg.get("resume_from"):
        state = torch.load(cfg["resume_from"], map_location=device)
        previous_metrics.update(state.get("metrics", {}))
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        history = state.get("history", []); step = int(state.get("step", 0)); start_epoch = int(state.get("epoch", 0)) + 1
        print(json.dumps({"resumed_from": cfg["resume_from"], "start_epoch": start_epoch}), flush=True)
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        model.train(); train_loss = 0.0; batches = 0; grad_norm = 0.0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True); optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, cfg): _, loss = model(x, y)
            nan_count += int(not torch.isfinite(loss).item())
            if use_scaler:
                scaler.scale(loss).backward(); scaler.unscale_(optimizer); grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"])); scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"])); optimizer.step()
            scheduler.step(); step += 1
            train_loss += loss.item(); batches += 1; train_tokens += int(y.numel())
            checkpoint_every = int(cfg.get("checkpoint_every_batches", 0))
            if checkpoint_every > 0 and batches % checkpoint_every == 0:
                # A batch checkpoint is deliberately marked as the previous
                # completed epoch. On restart the current epoch is repeated
                # from the beginning rather than incorrectly skipping the
                # unfinished remainder of an epoch.
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch - 1, "step": step, "history": history, "config": cfg, "vocab": c2i}, ckpt / "gpt_from_scratch_latest.pt")
                print(json.dumps({"checkpoint": str(ckpt / "gpt_from_scratch_latest.pt"), "epoch": epoch, "batch": batches}), flush=True)
            if batches % cfg.get("log_interval_batches", 250) == 0:
                print(json.dumps({"epoch": epoch, "batch": batches, "batches_per_epoch": len(train_loader), "loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]}), flush=True)
        train_loss /= max(1, batches); val_loss, val_acc = evaluate(model, val_loader, device, cfg)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc, "grad_norm": grad_norm, "lr": optimizer.param_groups[0]["lr"]})
        print(json.dumps(history[-1]), flush=True)
        (out / "history.json").write_text(json.dumps(history, indent=2))
        if cfg.get("save_every_epoch", True):
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch, "step": step, "history": history, "config": cfg, "vocab": c2i}, ckpt / f"gpt_from_scratch_epoch_{epoch:03d}.pt")
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch, "step": step, "history": history, "config": cfg, "vocab": c2i}, ckpt / "gpt_from_scratch_latest.pt")
    elapsed = time.time() - start; val_ppl = math.exp(min(20, history[-1]["val_loss"]))
    prompt_text = train_stories[0][:min(24, len(train_stories[0]))]
    prompt_ids = torch.tensor([[c2i[c] for c in prompt_text]], device=device)
    samples = {}; gen_start = time.time(); generated = None
    for temperature in (0.7, 0.9, 1.1):
        with autocast_context(device, cfg): ids = model.generate(prompt_ids, 300, temperature)[0].cpu()
        if generated is None: generated = ids
        samples[str(temperature)] = "".join(i2c[int(i)] for i in ids)
    generation_time = time.time() - gen_start; sample = samples["0.9"]
    (out / "sample.txt").write_text("\n\n".join(f"temperature={t}\n{s}" for t, s in samples.items()), encoding="utf-8")
    (out / "samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
    (out / "history.json").write_text(json.dumps(history, indent=2)); (out / "vocab.json").write_text(json.dumps({"char_to_idx": c2i, "idx_to_char": i2c}, indent=2))
    torch.save({"model": model.state_dict(), "config": cfg, "vocab": c2i}, ckpt / "gpt_from_scratch.pt")
    # If a completed checkpoint is evaluated again, no training batches run in
    # this invocation. Preserve the original run's performance fields instead
    # of replacing them with a misleading zero-throughput evaluation time.
    ran_training = train_tokens > 0
    tokens_per_second = train_tokens / max(elapsed, 1e-9) if ran_training else float(previous_metrics.get("training_tokens_per_second", 0.0))
    training_time = elapsed if ran_training else float(previous_metrics.get("training_time_seconds", 0.0))
    peak_memory = (torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" and ran_training else float(previous_metrics.get("peak_memory_mb", 0.0)))
    grad_norms = [float(row.get("grad_norm", 0.0)) for row in history]
    metrics = {"train_loss": history[-1]["train_loss"], "validation_loss": history[-1]["val_loss"], "perplexity": val_ppl, "bits_per_character": history[-1]["val_loss"] / math.log(2), "generalization_gap": history[-1]["val_loss"] - history[-1]["train_loss"], "top1_next_character_accuracy": history[-1]["val_accuracy"], "final_grad_norm": grad_norms[-1] if grad_norms else 0.0, "max_grad_norm": max(grad_norms, default=0.0), "gradient_nan_count": nan_count, "parameter_count": sum(p.numel() for p in model.parameters()), "training_tokens_per_second": tokens_per_second, "generation_tokens_per_second": (sum(len(s) for s in samples.values())) / max(generation_time, 1e-9), "peak_memory_mb": peak_memory, "training_time_seconds": training_time, "device": str(device), "vocabulary_size": len(chars), **generation_metrics(sample)}
    with (out / "metrics.csv").open("w", newline="") as f: w = csv.DictWriter(f, fieldnames=metrics); w.writeheader(); w.writerow(metrics)
    # Store final performance fields in the final checkpoint so a later Colab
    # session can evaluate a completed run without losing throughput/time data.
    torch.save({"model": model.state_dict(), "config": cfg, "vocab": c2i, "metrics": metrics, "history": history, "epoch": cfg["epochs"], "step": step}, ckpt / "gpt_from_scratch.pt")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__": main()
