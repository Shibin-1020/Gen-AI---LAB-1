"""Task 1.3 -- Training with cross-entropy, LR warm-up + cosine decay, >= 10 epochs.

Everything that happens during a run goes to an append-only raw log under
reproducibility/raw_logs/task1_llm/shibin_thomas/<run_id>/:
    train.log         human-readable log (config, per-interval losses, per-epoch eval, warnings)
    train_steps.csv   one row per optimizer step: loss, lr, pre-clip grad norm, tokens/sec
    epochs.csv        one row per epoch: train loss (running + eval-mode), val loss, accuracy
Checkpoints go to task1_llm/shibin_thomas/checkpoints/<run_id>/:
    best_model.pt     weights of the epoch with the lowest validation loss (committed)
    last_full.pt      weights + optimizer + scaler + RNG state for --resume (git-ignored)

Run:  python task1_llm/shibin_thomas/src/train.py --config task1_llm/shibin_thomas/configs/gpt_char_v1.yaml
      python task1_llm/shibin_thomas/src/train.py --config ... --resume latest
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from data_prep import EOS_ID, CharStream, decode, encode, load_vocab, prepare_data
from model import GPTCharLM, GPTConfig
from utils import (autocast_ctx, config_hash, deep_copy_config, environment_info, get_device, get_logger,
                   git_info, hardware_info, latest_run_id, load_config, peak_memory_mb, read_json, rel,
                   repo_path, resolve_precision, run_dirs, set_seed, sha256_file, sync, update_manifest,
                   write_json)

STEP_FIELDS = ["step", "epoch", "lr", "loss", "grad_norm", "tokens_per_sec", "elapsed_sec", "nonfinite", "spike"]
EPOCH_FIELDS = ["epoch", "step", "lr", "train_loss_running", "train_loss_eval", "val_loss", "val_ppl",
                "val_bpc", "val_top1_acc", "train_top1_acc", "gen_gap", "epoch_time_sec", "tokens_per_sec"]


def lr_at(step: int, total_steps: int, warmup: int, lr: float, min_lr: float, schedule: str) -> float:
    """Linear warm-up for `warmup` steps, then cosine (or linear) decay to min_lr at total_steps."""
    if step < warmup:
        return lr * (step + 1) / warmup
    if schedule == "constant":
        return lr
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    if schedule == "linear":
        return lr - (lr - min_lr) * progress
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def build_optimizer(model: torch.nn.Module, tcfg: dict) -> torch.optim.Optimizer:
    """AdamW; weight decay on matrices/embeddings only (not biases or LayerNorm gains)."""
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if p.requires_grad:
            (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": tcfg["weight_decay"]},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=tcfg["lr"], betas=tuple(tcfg["betas"]), eps=tcfg.get("eps", 1e-8))


@torch.no_grad()
def evaluate_stream(model, stream: CharStream, batch_size: int, device, precision, max_seqs=None):
    """Mean per-character cross-entropy (nats) and top-1 next-character accuracy."""
    model.eval()
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    for x, y in stream.batches(stream.eval_starts(max_seqs), batch_size, drop_last=False):
        with autocast_ctx(device, precision):
            logits, _ = model(x)
        logits = logits.float()
        total_loss += torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.reshape(-1), reduction="sum").item()
        total_correct += (logits.argmax(-1) == y).sum().item()
        total_tokens += y.numel()
    model.train()
    return total_loss / total_tokens, total_correct / total_tokens


@torch.no_grad()
def quick_sample(model, char_to_idx, idx_to_char, device, prompt="Once upon a time", n=200) -> str:
    model.eval()
    idx = torch.tensor([encode(prompt, char_to_idx)], device=device)
    out = model.generate(idx, n, greedy=True, eos_id=EOS_ID)
    model.train()
    return decode(out[0].tolist(), idx_to_char, stop_at_eos=True)


def _append_csv(path: Path, fields: list[str], row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


def load_step_log(path: Path) -> np.ndarray:
    """Read train_steps.csv. After a --resume the steps between the last checkpoint and the crash
    appear twice in the (append-only) raw log; keep only the latest row for each step."""
    rows = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    _, last = np.unique(rows["step"][::-1], return_index=True)
    return rows[::-1][last]


def _rng_state() -> dict:
    s = {"torch": torch.get_rng_state(), "numpy": np.random.get_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def _set_rng_state(s: dict) -> None:
    torch.set_rng_state(s["torch"])
    np.random.set_state(s["numpy"])
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def train(cfg: dict, resume: str | None = None) -> str:
    tcfg, mcfg = cfg["training"], cfg["model"]
    device = get_device(cfg.get("device", "auto"))
    precision = resolve_precision(tcfg.get("precision", "auto"), device)

    # ---- run id / dirs / logger
    if resume:
        run_id = latest_run_id(cfg["run_name"]) if resume == "latest" else resume
    else:
        run_id = f"{cfg['run_name']}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    dirs = run_dirs(run_id)
    log = get_logger(f"train.{run_id}", dirs["raw_logs"] / "train.log")
    steps_csv, epochs_csv = dirs["raw_logs"] / "train_steps.csv", dirs["raw_logs"] / "epochs.csv"

    # ---- data
    processed = prepare_data(cfg, log=log.info)
    meta = read_json(processed / "meta.json")
    char_to_idx, idx_to_char = load_vocab(processed)
    dtype = np.uint8 if meta["dtype"] == "uint8" else np.uint16
    train_stream = CharStream(processed / "train.bin", mcfg["block_size"], device, dtype)
    val_stream = CharStream(processed / "val.bin", mcfg["block_size"], device, dtype)

    # ---- model / optimizer
    set_seed(cfg["seed"])
    gcfg = GPTConfig(vocab_size=meta["vocab_size"], **mcfg)
    model = GPTCharLM(gcfg).to(device)
    optimizer = build_optimizer(model, tcfg)
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))
    fwd_model = torch.compile(model) if tcfg.get("compile", False) else model

    B, accum, T = tcfg["batch_size"], tcfg.get("grad_accum_steps", 1), mcfg["block_size"]
    steps_per_epoch = train_stream.num_sequences(T - 1) // (B * accum)   # worst-case offset -> constant count
    total_steps = steps_per_epoch * tcfg["epochs"]
    warmup = tcfg.get("warmup_steps") or int(tcfg.get("warmup_frac", 0.02) * total_steps)

    state = {"epoch": 0, "step": 0, "best_val": float("inf"), "best_epoch": None, "train_time": 0.0,
             "train_tokens": 0, "nonfinite_steps": 0, "spikes": [], "grad_norms_sum": 0.0, "grad_norm_max": 0.0,
             "loss_ema": None, "wall_time": 0.0}
    gen = torch.Generator().manual_seed(cfg["seed"])

    if resume:
        ck = torch.load(dirs["checkpoints"] / "last_full.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        state.update(ck["state"])
        gen.set_state(ck["data_gen_state"])
        _set_rng_state(ck["rng"])
        log.info(f"RESUMED run {run_id} from epoch {state['epoch']} step {state['step']}")
    else:
        log.info(f"START run {run_id}")
        update_manifest(
            run_id,
            config=deep_copy_config(cfg), config_file=cfg["_config_path"], config_overrides=cfg["_overrides"],
            config_hash=config_hash(cfg), git=git_info(), hardware=hardware_info(device),
            environment=environment_info(), precision=precision,
            data={"processed_dir": rel(processed), "meta": rel(processed / "meta.json"),
                  "raw_sha256": meta["raw_sha256"], "train_tokens": meta["train"]["tokens"],
                  "val_tokens": meta["val"]["tokens"], "vocab_size": meta["vocab_size"]},
            raw_logs=rel(dirs["raw_logs"]), outputs=rel(dirs["outputs"]), checkpoints=rel(dirs["checkpoints"]),
        )

    n_params, n_params_ne = model.num_params(), model.num_params(non_embedding=True)
    log.info(f"config={cfg['_config_path']} overrides={cfg['_overrides']} hash={config_hash(cfg)}")
    log.info(f"device={device} precision={precision} hardware={hardware_info(device)}")
    log.info(f"model={gcfg.to_dict()}")
    log.info(f"params total={n_params:,} non_embedding={n_params_ne:,}")
    log.info(f"train_tokens={len(train_stream):,} val_tokens={len(val_stream):,} vocab={meta['vocab_size']}")
    log.info(f"batch={B}x{accum} block={T} tokens/step={B * accum * T:,} steps/epoch={steps_per_epoch:,} "
             f"total_steps={total_steps:,} warmup={warmup:,} epochs={tcfg['epochs']}")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    wall_start = time.time() - state["wall_time"]
    spike_factor, log_every = tcfg.get("spike_factor", 1.5), tcfg.get("log_interval", 100)

    for epoch in range(state["epoch"], tcfg["epochs"]):
        epoch_start = time.time()
        epoch_tokens, epoch_loss_sum, epoch_steps = 0, 0.0, 0
        starts = train_stream.epoch_starts(gen)
        batch_iter = train_stream.batches(starts, B)
        interval_t0, interval_tokens = time.time(), 0

        for _ in range(steps_per_epoch):
            step = state["step"]
            lr = lr_at(step, total_steps, warmup, tcfg["lr"], tcfg["min_lr"], tcfg.get("schedule", "cosine"))
            for g in optimizer.param_groups:
                g["lr"] = lr

            sync(device)
            t0 = time.time()
            loss_accum = 0.0
            for _micro in range(accum):
                x, y = next(batch_iter)
                with autocast_ctx(device, precision):
                    _, loss = fwd_model(x, y)
                loss_accum += loss.item() / accum
                scaler.scale(loss / accum).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"]).item()

            nonfinite = not (math.isfinite(loss_accum) and math.isfinite(grad_norm))
            if nonfinite:
                state["nonfinite_steps"] += 1
                log.warning(f"non-finite loss/grad at step {step}: loss={loss_accum} grad_norm={grad_norm}; "
                            "skipping update" + (" (fp16 loss-scale overflow: GradScaler lowers the scale)" if precision == "fp16" else ""))
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
            else:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            sync(device)
            dt = time.time() - t0

            tokens = B * accum * T
            state["train_time"] += dt
            state["train_tokens"] += tokens
            epoch_tokens += tokens
            interval_tokens += tokens

            spike = False
            if not nonfinite:
                ema = state["loss_ema"]
                spike = ema is not None and step > warmup and loss_accum > spike_factor * ema
                if spike:
                    state["spikes"].append({"step": step, "loss": loss_accum, "ema": ema})
                    log.warning(f"loss spike at step {step}: loss={loss_accum:.4f} ema={ema:.4f}")
                state["loss_ema"] = loss_accum if ema is None else 0.99 * ema + 0.01 * loss_accum
                state["grad_norms_sum"] += grad_norm
                state["grad_norm_max"] = max(state["grad_norm_max"], grad_norm)
                epoch_loss_sum += loss_accum
                epoch_steps += 1

            _append_csv(steps_csv, STEP_FIELDS, {
                "step": step, "epoch": epoch, "lr": f"{lr:.6e}", "loss": f"{loss_accum:.5f}",
                "grad_norm": f"{grad_norm:.5f}", "tokens_per_sec": f"{tokens / dt:.1f}",
                "elapsed_sec": f"{time.time() - wall_start:.1f}", "nonfinite": int(nonfinite), "spike": int(spike)})

            if step % log_every == 0:
                elapsed = time.time() - interval_t0
                log.info(f"epoch {epoch + 1}/{tcfg['epochs']} step {step}/{total_steps} lr {lr:.2e} "
                         f"loss {loss_accum:.4f} grad_norm {grad_norm:.3f} "
                         f"tok/s {interval_tokens / max(elapsed, 1e-9):,.0f}")
                interval_t0, interval_tokens = time.time(), 0
            state["step"] += 1

        # ---- end of epoch: evaluation, checkpointing
        train_running = epoch_loss_sum / max(1, epoch_steps)
        eb = tcfg.get("eval_batch_size", B)
        val_loss, val_acc = evaluate_stream(model, val_stream, eb, device, precision)
        tr_eval_loss, tr_acc = evaluate_stream(model, train_stream, eb, device, precision,
                                               max_seqs=tcfg.get("eval_train_max_seqs"))
        epoch_time = time.time() - epoch_start
        row = {"epoch": epoch + 1, "step": state["step"], "lr": f"{lr:.6e}",
               "train_loss_running": f"{train_running:.5f}", "train_loss_eval": f"{tr_eval_loss:.5f}",
               "val_loss": f"{val_loss:.5f}", "val_ppl": f"{math.exp(val_loss):.4f}",
               "val_bpc": f"{val_loss / math.log(2):.4f}", "val_top1_acc": f"{val_acc:.5f}",
               "train_top1_acc": f"{tr_acc:.5f}", "gen_gap": f"{val_loss - tr_eval_loss:.5f}",
               "epoch_time_sec": f"{epoch_time:.1f}", "tokens_per_sec": f"{epoch_tokens / epoch_time:.1f}"}
        _append_csv(epochs_csv, EPOCH_FIELDS, row)
        log.info(f"EPOCH {epoch + 1} | train_running {train_running:.4f} | train_eval {tr_eval_loss:.4f} | "
                 f"val {val_loss:.4f} | val_ppl {math.exp(val_loss):.3f} | val_bpc {val_loss / math.log(2):.3f} | "
                 f"val_acc {val_acc:.4f} | gap {val_loss - tr_eval_loss:+.4f} | time {epoch_time:.0f}s")
        log.info(f"EPOCH {epoch + 1} sample (greedy): {quick_sample(model, char_to_idx, idx_to_char, device)!r}")

        state["epoch"] = epoch + 1
        state["wall_time"] = time.time() - wall_start
        if val_loss < state["best_val"]:
            state["best_val"], state["best_epoch"] = val_loss, epoch + 1
            torch.save({"model": model.state_dict(), "model_config": gcfg.to_dict(), "epoch": epoch + 1,
                        "step": state["step"], "val_loss": val_loss, "run_id": run_id,
                        "processed_dir": rel(processed)}, dirs["checkpoints"] / "best_model.pt")
            log.info(f"new best val_loss {val_loss:.4f} -> {rel(dirs['checkpoints'] / 'best_model.pt')}")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                    "state": state, "data_gen_state": gen.get_state(), "rng": _rng_state(),
                    "model_config": gcfg.to_dict(), "run_id": run_id}, dirs["checkpoints"] / "last_full.pt")

    # ---- summary
    total_wall = time.time() - wall_start
    good_steps = max(1, state["step"] - state["nonfinite_steps"])
    gn = load_step_log(steps_csv)["grad_norm"]
    summary = {
        "run_id": run_id, "epochs": tcfg["epochs"], "total_steps": state["step"],
        "params_total": n_params, "params_non_embedding": n_params_ne,
        "best_epoch": state["best_epoch"], "best_val_loss": state["best_val"],
        "total_training_time_sec": total_wall,
        "pure_step_time_sec": state["train_time"],
        "train_tokens_processed": state["train_tokens"],
        "train_tokens_per_sec": state["train_tokens"] / state["train_time"],
        "grad_norm_mean": state["grad_norms_sum"] / good_steps,
        "grad_norm_max": state["grad_norm_max"],
        "grad_norm_p99": float(np.nanpercentile(gn, 99)),
        "grad_clip": tcfg["grad_clip"],
        "frac_steps_clipped": float(np.mean(gn > tcfg["grad_clip"])),
        "nonfinite_steps": state["nonfinite_steps"],
        "loss_spikes": len(state["spikes"]), "loss_spike_steps": [s["step"] for s in state["spikes"]][:50],
        "spike_rule": f"loss > {spike_factor} x EMA(0.99) after warm-up",
        "precision": precision, "device": str(device), "hardware": hardware_info(device),
        **peak_memory_mb(device),
    }
    write_json(dirs["outputs"] / "train_summary.json", summary)
    ckpts = {p.name: {"path": rel(p), "sha256": sha256_file(p)} for p in sorted(dirs["checkpoints"].glob("*.pt"))}
    update_manifest(run_id, train_summary=rel(dirs["outputs"] / "train_summary.json"), checkpoints_files=ckpts,
                    best_checkpoint=rel(dirs["checkpoints"] / "best_model.pt"))
    log.info(f"DONE run {run_id} | best val {state['best_val']:.4f} @ epoch {state['best_epoch']} | "
             f"wall {total_wall / 60:.1f} min | train tok/s {summary['train_tokens_per_sec']:,.0f} | "
             f"peak mem {peak_memory_mb(device)}")
    return run_id


def main() -> None:
    p = argparse.ArgumentParser(description="Task 1 GPT training")
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. training.epochs=12")
    p.add_argument("--resume", default=None, help="run_id to resume, or 'latest'")
    a = p.parse_args()
    train(load_config(a.config, a.set), a.resume)


if __name__ == "__main__":
    main()
