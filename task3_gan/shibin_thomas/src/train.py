"""Task 3.1 -- CycleGAN training: adversarial (LSGAN) + cycle-consistency + identity losses.

Per optimizer step, with a = real Monet, b = real photo:
    fake_b = G_AB(a), fake_a = G_BA(b), rec_a = G_BA(fake_b), rec_b = G_AB(fake_a)
    L_G  = MSE(D_B(fake_b), 1) + MSE(D_A(fake_a), 1)                       adversarial (least squares)
         + lambda_cyc * (|rec_a - a|_1 + |rec_b - b|_1)                        cycle consistency
         + lambda_id  * (|G_BA(a) - a|_1 + |G_AB(b) - b|_1)                    identity (colour preservation)
    L_DA = 0.5 * (MSE(D_A(a), 1) + MSE(D_A(pool(fake_a)), 0))   and the same for D_B
Adam(2e-4, beta1 0.5); learning rate constant for the first half of the epochs, then linear decay to 0.

Raw, append-only logs (reproducibility/raw_logs/task3_gan/shibin_thomas/<run_id>/):
    train.log, steps.csv (every step: each loss term, gradient norms of G and D, lr, non-finite flag),
    epochs.csv (per epoch: mean losses, fixed-batch cycle L1, periodic FID in both directions).
Checkpoints: checkpoints/<run_id>/last_full.pt (everything, for --resume; git-ignored) and
G_AB.pt / G_BA.pt (final generator weights; committed).
"""
from __future__ import annotations

import csv
import math
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import FolderDataset, UnpairedDataset, list_images, to_pil, train_transform
from models import ImagePool, build_models, count_params
from utils import (config_hash, deep_copy_config, environment_info, get_device, get_logger, git_info,
                   hardware_info, latest_run_id, peak_memory_mb, rel, repo_path, run_dirs, set_seed, sync,
                   update_manifest, write_json)

LOSS_KEYS = ["loss_G", "gan_A2B", "gan_B2A", "cyc_A", "cyc_B", "idt_A", "idt_B", "loss_D_A", "loss_D_B"]


def gpu_snapshot() -> str:
    """nvidia-smi summary at start-up, so a GPU shared with other jobs is documented in the raw log.
    Only GPU memory / load and program NAMES are kept -- no paths, PIDs or user names."""
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
        procs = subprocess.run(["nvidia-smi", "--query-compute-apps=process_name", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=20).stdout.split("\n")
        names = [p.strip().replace("\\", "/").rsplit("/", 1)[-1] for p in procs if p.strip()]
        visible = sorted({n for n in names if not n.startswith("[")})
        hidden = sum(1 for n in names if n.startswith("["))
        python_jobs = sum(1 for n in names if n.lower().startswith("python"))
        return (f"{gpu} | {len(names)} processes on the GPU ({python_jobs} python, {hidden} not visible to this user); "
                f"programs: {', '.join(visible)}")
    except Exception as e:  # nvidia-smi missing (CPU machine)
        return f"unavailable ({type(e).__name__})"


def data_fingerprint(paths: list[str]) -> str:
    """SHA-256 over 'file name:size' of every image: identifies the exact dataset without committing it."""
    import hashlib
    import os
    h = hashlib.sha256()
    for p in paths:
        h.update(f"{os.path.basename(p)}:{os.path.getsize(p)}\n".encode())
    return h.hexdigest()


def lr_factor(epoch: int, n_const: int, n_decay: int) -> float:
    """1.0 for the first n_const epochs, then linear decay to 0 over n_decay epochs (Zhu et al. 2017)."""
    return 1.0 - max(0, epoch + 1 - n_const) / float(n_decay + 1)


def _append_csv(path: Path, row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def _grad_norm(params) -> float:
    return torch.nn.utils.clip_grad_norm_(params, float("inf")).item()


@torch.no_grad()
def translate_batch(G, x: torch.Tensor, device, bs: int = 16) -> torch.Tensor:
    G.eval()
    out = torch.cat([G(x[i:i + bs].to(device)).float().cpu() for i in range(0, len(x), bs)])
    G.train()
    return out


def load_fixed(paths: list[str], size: int) -> torch.Tensor:
    ds = FolderDataset(paths, size)
    return torch.stack([ds[i][0] for i in range(len(ds))]) if len(ds) else torch.zeros(0, 3, size, size)


def save_grid(rows: list[torch.Tensor], path: Path) -> None:
    """rows: list of (N,3,H,W) tensors in [-1,1] -> one PNG, each row one tensor."""
    from PIL import Image
    n, h, w = rows[0].shape[0], rows[0].shape[2], rows[0].shape[3]
    canvas = Image.new("RGB", (n * w, len(rows) * h), "white")
    for r, t in enumerate(rows):
        for c in range(n):
            canvas.paste(to_pil(t[c]), (c * w, r * h))
    canvas.save(path)


def train(cfg: dict, resume: str | None = None) -> str:
    d, mcfg, tcfg = cfg["data"], cfg["model"], cfg["training"]
    device = get_device(cfg.get("device", "auto"))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    run_id = (latest_run_id(cfg["run_name"]) if resume == "latest" else resume) if resume else \
        f"{cfg['run_name']}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    dirs = run_dirs(run_id)
    log = get_logger(f"train.{run_id}", dirs["raw_logs"] / "train.log")
    steps_csv, epochs_csv = dirs["raw_logs"] / "steps.csv", dirs["raw_logs"] / "epochs.csv"

    paths_a, paths_b = list_images(repo_path(d["monet_dir"])), list_images(repo_path(d["photo_dir"]))
    if d.get("max_images_a"):
        paths_a = paths_a[: d["max_images_a"]]
    if d.get("max_images_b"):
        paths_b = paths_b[: d["max_images_b"]]
    set_seed(cfg["seed"])
    ds = UnpairedDataset(paths_a, paths_b, tcfg["samples_per_epoch"],
                         train_transform(d["load_size"], d["crop_size"], d.get("flip", True)), seed=cfg["seed"])

    nets = build_models(mcfg)
    for n in nets.values():
        n.to(device)
    G_AB, G_BA, D_A, D_B = nets["G_AB"], nets["G_BA"], nets["D_A"], nets["D_B"]
    params_G = list(G_AB.parameters()) + list(G_BA.parameters())
    params_D = list(D_A.parameters()) + list(D_B.parameters())
    betas = tuple(tcfg["betas"])
    opt_G = torch.optim.Adam(params_G, lr=tcfg["lr"], betas=betas)
    opt_D = torch.optim.Adam(params_D, lr=tcfg["lr"], betas=betas)
    pool_A, pool_B = ImagePool(tcfg["pool_size"], cfg["seed"]), ImagePool(tcfg["pool_size"], cfg["seed"] + 1)
    lam_cyc, lam_id = tcfg["lambda_cycle"], tcfg["lambda_identity"]
    epochs = tcfg["epochs_constant"] + tcfg["epochs_decay"]

    state = {"epoch": 0, "step": 0, "train_time": 0.0, "pairs_seen": 0, "nonfinite_steps": 0, "wall_time": 0.0}
    if resume:
        ck = torch.load(dirs["checkpoints"] / "last_full.pt", map_location=device, weights_only=False)
        for k in nets:
            nets[k].load_state_dict(ck[k])
        opt_G.load_state_dict(ck["opt_G"])
        opt_D.load_state_dict(ck["opt_D"])
        state.update(ck["state"])
        log.info(f"RESUMED run {run_id} at epoch {state['epoch']} step {state['step']} (image pools restart empty)")
    else:
        log.info(f"START run {run_id}")
        update_manifest(run_id, config=deep_copy_config(cfg), config_file=cfg["_config_path"],
                        config_overrides=cfg["_overrides"], config_hash=config_hash(cfg), git=git_info(),
                        hardware=hardware_info(device), environment=environment_info(), gpu_at_start=gpu_snapshot(),
                        data={"monet_dir": d["monet_dir"], "photo_dir": d["photo_dir"], "n_monet": len(paths_a),
                              "n_photo": len(paths_b), "monet_fingerprint": data_fingerprint(paths_a),
                              "photo_fingerprint": data_fingerprint(paths_b)},
                        raw_logs=rel(dirs["raw_logs"]), outputs=rel(dirs["outputs"]), checkpoints=rel(dirs["checkpoints"]))

    params = {k: count_params(v) for k, v in nets.items()}
    steps_per_epoch = math.ceil(tcfg["samples_per_epoch"] / tcfg["batch_size"])
    log.info(f"config={cfg['_config_path']} overrides={cfg['_overrides']} hash={config_hash(cfg)}")
    log.info(f"device={device} hardware={hardware_info(device)}")
    log.info(f"GPU at start: {gpu_snapshot()}")
    log.info(f"domains: A=Monet {len(paths_a):,} images ({d['monet_dir']}), B=Photo {len(paths_b):,} images ({d['photo_dir']})")
    log.info(f"model={mcfg} params={params} total={sum(params.values()):,}")
    log.info(f"training={tcfg}")
    log.info(f"epochs={epochs} ({tcfg['epochs_constant']} constant + {tcfg['epochs_decay']} decay) "
             f"steps/epoch={steps_per_epoch} total_steps={epochs * steps_per_epoch:,}")

    fixed_a = load_fixed(paths_a[: tcfg.get("n_fixed", 4)], d["crop_size"])
    fixed_b = load_fixed(paths_b[: tcfg.get("n_fixed", 4)], d["crop_size"])
    eval_a = load_fixed(paths_a[: tcfg.get("cycle_eval_n", 64)], d["crop_size"])
    eval_b = load_fixed(paths_b[: tcfg.get("cycle_eval_n", 64)], d["crop_size"])
    fid_every = tcfg.get("fid_every", 0)
    incep, real_feats = None, {}
    if fid_every:
        try:
            from gan_metrics import InceptionFeatures
            incep = InceptionFeatures(device, pretrained=cfg["eval"].get("pretrained", True))
            n_fid = tcfg.get("fid_n", 300)
            real_feats = {"A": incep.from_paths(paths_a[:n_fid]), "B": incep.from_paths(paths_b[:n_fid])}
            fid_inputs = {"A": load_fixed(paths_a[:n_fid], d["crop_size"]), "B": load_fixed(paths_b[:n_fid], d["crop_size"])}
            log.info(f"periodic FID every {fid_every} epochs on {n_fid} images per domain")
        except Exception as e:
            log.warning(f"periodic FID disabled: {type(e).__name__}: {e}")
            incep = None
    samples_dir = dirs["outputs"] / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    wall0 = time.perf_counter() - state["wall_time"]
    gen = torch.Generator().manual_seed(cfg["seed"])
    for epoch in range(state["epoch"], epochs):
        f = lr_factor(epoch, tcfg["epochs_constant"], tcfg["epochs_decay"])
        for opt in (opt_G, opt_D):
            for g in opt.param_groups:
                g["lr"] = tcfg["lr"] * f
        ds.set_epoch(epoch)
        gen.manual_seed(cfg["seed"] * 1000 + epoch)
        loader = DataLoader(ds, batch_size=tcfg["batch_size"], shuffle=False, drop_last=True,
                            num_workers=tcfg.get("num_workers", 0), pin_memory=device.type == "cuda", generator=gen)
        sums = {k: 0.0 for k in LOSS_KEYS}
        n_ok, ep_t0 = 0, time.perf_counter()
        for real_a, real_b in loader:
            real_a, real_b = real_a.to(device, non_blocking=True), real_b.to(device, non_blocking=True)
            sync(device)
            t0 = time.perf_counter()

            # ---- generators
            for p in params_D:
                p.requires_grad_(False)
            fake_b, fake_a = G_AB(real_a), G_BA(real_b)
            rec_a, rec_b = G_BA(fake_b), G_AB(fake_a)
            pred_fb, pred_fa = D_B(fake_b), D_A(fake_a)
            gan_ab = F.mse_loss(pred_fb, torch.ones_like(pred_fb))
            gan_ba = F.mse_loss(pred_fa, torch.ones_like(pred_fa))
            cyc_a, cyc_b = F.l1_loss(rec_a, real_a), F.l1_loss(rec_b, real_b)
            if lam_id > 0:
                idt_a, idt_b = F.l1_loss(G_BA(real_a), real_a), F.l1_loss(G_AB(real_b), real_b)
            else:
                idt_a = idt_b = torch.zeros((), device=device)
            loss_G = gan_ab + gan_ba + lam_cyc * (cyc_a + cyc_b) + lam_id * (idt_a + idt_b)
            opt_G.zero_grad(set_to_none=True)
            loss_G.backward()
            gn_G = _grad_norm(params_G)

            # ---- discriminators (history pool of fakes)
            for p in params_D:
                p.requires_grad_(True)
            pa, pb = pool_A.query(fake_a), pool_B.query(fake_b)
            d_ra, d_fa = D_A(real_a), D_A(pa)
            d_rb, d_fb = D_B(real_b), D_B(pb)
            loss_D_A = 0.5 * (F.mse_loss(d_ra, torch.ones_like(d_ra)) + F.mse_loss(d_fa, torch.zeros_like(d_fa)))
            loss_D_B = 0.5 * (F.mse_loss(d_rb, torch.ones_like(d_rb)) + F.mse_loss(d_fb, torch.zeros_like(d_fb)))
            opt_D.zero_grad(set_to_none=True)
            (loss_D_A + loss_D_B).backward()
            gn_D = _grad_norm(params_D)

            vals = {"loss_G": loss_G.item(), "gan_A2B": gan_ab.item(), "gan_B2A": gan_ba.item(),
                    "cyc_A": cyc_a.item(), "cyc_B": cyc_b.item(), "idt_A": idt_a.item(), "idt_B": idt_b.item(),
                    "loss_D_A": loss_D_A.item(), "loss_D_B": loss_D_B.item()}
            ok = all(math.isfinite(v) for v in vals.values()) and math.isfinite(gn_G) and math.isfinite(gn_D)
            if ok:
                opt_G.step()
                opt_D.step()
                for k in LOSS_KEYS:
                    sums[k] += vals[k]
                n_ok += 1
            else:
                state["nonfinite_steps"] += 1
                opt_G.zero_grad(set_to_none=True)
                opt_D.zero_grad(set_to_none=True)
                log.warning(f"non-finite loss/gradient at step {state['step']}: {vals} gnG={gn_G} gnD={gn_D}; skipped")
            sync(device)
            state["train_time"] += time.perf_counter() - t0
            state["pairs_seen"] += real_a.size(0)
            _append_csv(steps_csv, {"step": state["step"], "epoch": epoch + 1, "lr": f"{tcfg['lr'] * f:.3e}",
                                    **{k: f"{v:.5f}" for k, v in vals.items()},
                                    "grad_norm_G": f"{gn_G:.4f}", "grad_norm_D": f"{gn_D:.4f}", "nonfinite": int(not ok)})
            if state["step"] % tcfg.get("log_interval", 100) == 0:
                log.info(f"epoch {epoch + 1}/{epochs} step {state['step']} lr {tcfg['lr'] * f:.2e} | G {vals['loss_G']:.3f} "
                         f"(gan {vals['gan_A2B']:.3f}/{vals['gan_B2A']:.3f} cyc {vals['cyc_A']:.3f}/{vals['cyc_B']:.3f} "
                         f"idt {vals['idt_A']:.3f}/{vals['idt_B']:.3f}) | D_A {vals['loss_D_A']:.3f} D_B {vals['loss_D_B']:.3f} "
                         f"| gnG {gn_G:.2f} gnD {gn_D:.2f}")
            state["step"] += 1

        # ---- end of epoch: fixed-batch cycle error, sample grid, periodic FID, checkpoint
        ep_time = time.perf_counter() - ep_t0
        fb = translate_batch(G_AB, eval_a, device)
        cyc_l1_a = (translate_batch(G_BA, fb, device) - eval_a).abs().mean().item() / 2   # in [0,1] pixel units
        fa = translate_batch(G_BA, eval_b, device)
        cyc_l1_b = (translate_batch(G_AB, fa, device) - eval_b).abs().mean().item() / 2
        save_grid([fixed_b, translate_batch(G_BA, fixed_b, device), fixed_a, translate_batch(G_AB, fixed_a, device)],
                  samples_dir / f"epoch_{epoch + 1:03d}.png")
        row = {"epoch": epoch + 1, "step": state["step"], "lr": f"{tcfg['lr'] * f:.3e}",
               **{f"mean_{k}": f"{sums[k] / max(1, n_ok):.5f}" for k in LOSS_KEYS},
               "cycle_l1_A": f"{cyc_l1_a:.5f}", "cycle_l1_B": f"{cyc_l1_b:.5f}", "fid_B2A": "", "fid_A2B": "",
               "epoch_time_sec": f"{ep_time:.1f}"}
        if incep is not None and ((epoch + 1) % fid_every == 0 or epoch + 1 == epochs):
            from gan_metrics import fid_from_features
            g_ba = [to_pil(t) for t in translate_batch(G_BA, fid_inputs["B"], device)]
            g_ab = [to_pil(t) for t in translate_batch(G_AB, fid_inputs["A"], device)]
            row["fid_B2A"] = f"{fid_from_features(real_feats['A'], incep.from_pil(g_ba)):.3f}"
            row["fid_A2B"] = f"{fid_from_features(real_feats['B'], incep.from_pil(g_ab)):.3f}"
        _append_csv(epochs_csv, row)
        log.info(f"EPOCH {epoch + 1}/{epochs} | G {row['mean_loss_G']} D_A {row['mean_loss_D_A']} D_B {row['mean_loss_D_B']} "
                 f"| cyc_l1 A {cyc_l1_a:.4f} B {cyc_l1_b:.4f} | FID B2A {row['fid_B2A'] or '-'} A2B {row['fid_A2B'] or '-'} "
                 f"| {ep_time:.0f}s")
        state["epoch"] = epoch + 1
        state["wall_time"] = time.perf_counter() - wall0
        torch.save({**{k: v.state_dict() for k, v in nets.items()}, "opt_G": opt_G.state_dict(),
                    "opt_D": opt_D.state_dict(), "state": state, "model_config": mcfg, "run_id": run_id},
                   dirs["checkpoints"] / "last_full.pt")

    for k in ("G_AB", "G_BA"):
        torch.save({"model": nets[k].state_dict(), "model_config": mcfg, "run_id": run_id, "epoch": state["epoch"]},
                   dirs["checkpoints"] / f"{k}.pt")
    total = time.perf_counter() - wall0
    summary = {"run_id": run_id, "epochs": epochs, "total_steps": state["step"], "params": params,
               "params_total": sum(params.values()), "params_generators": params["G_AB"] + params["G_BA"],
               "params_discriminators": params["D_A"] + params["D_B"], "training_time_sec": total,
               "pure_step_time_sec": state["train_time"], "pairs_seen": state["pairs_seen"],
               "train_pairs_per_sec": state["pairs_seen"] / max(state["train_time"], 1e-9),
               "train_images_per_sec": 2 * state["pairs_seen"] / max(state["train_time"], 1e-9),
               "nonfinite_steps": state["nonfinite_steps"], "device": str(device), "hardware": hardware_info(device),
               "gpu_at_start": gpu_snapshot(), **peak_memory_mb(device)}
    write_json(dirs["outputs"] / "train_summary.json", summary)
    update_manifest(run_id, train_summary=rel(dirs["outputs"] / "train_summary.json"),
                    checkpoints_files={k: rel(dirs["checkpoints"] / f"{k}.pt") for k in ("G_AB", "G_BA")})
    log.info(f"DONE run {run_id} | {total / 60:.1f} min | {summary['train_images_per_sec']:.1f} images/s | "
             f"non-finite steps {state['nonfinite_steps']} | peak {peak_memory_mb(device)}")
    return run_id


def main() -> None:
    import argparse
    from utils import load_config
    p = argparse.ArgumentParser(description="Task 3 CycleGAN training only (run_pipeline.py does train + export + eval)")
    p.add_argument("--config", default="task3_gan/shibin_thomas/configs/cyclegan_v1.yaml")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--resume", default=None, help="run id or 'latest'")
    a = p.parse_args()
    train(load_config(a.config, a.set), a.resume)


if __name__ == "__main__":
    main()
