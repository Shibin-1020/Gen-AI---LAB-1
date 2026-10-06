"""GPU-optimized CycleGAN training and competition export.

Run from this directory:
    python -u src/train.py --config config.yaml

The implementation is from scratch: two generators, two PatchGAN discriminators,
LSGAN loss, cycle/identity losses, replay buffers, augmentation, AMP, linear
learning-rate decay, checkpoints, and a Kaggle-compatible images.zip.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def device_from(value: str) -> torch.device:
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else ("cpu" if value == "auto" else value))


def autocast_context(device: torch.device, cfg: dict):
    if device.type != "cuda" or not cfg.get("amp", True): return nullcontext()
    dtype = torch.float16 if cfg.get("precision", "bfloat16").lower() == "float16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


class ImageDomain(Dataset):
    def __init__(self, root: Path, size: int, train: bool, crop_margin: int, random_flip: bool):
        self.files = sorted(p for p in root.glob("**/*") if not p.name.startswith("._") and p.name != ".DS_Store" and p.suffix.lower() in IMAGE_EXTENSIONS)
        if not self.files: raise FileNotFoundError(f"No images found in {root}")
        if train:
            ops = [transforms.Resize(size + crop_margin, interpolation=transforms.InterpolationMode.BICUBIC), transforms.RandomCrop(size)]
            if random_flip: ops.append(transforms.RandomHorizontalFlip())
        else:
            ops = [transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC), transforms.CenterCrop(size)]
        self.transform = transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize((.5, .5, .5), (.5, .5, .5))])

    def __len__(self): return len(self.files)

    def __getitem__(self, index):
        with Image.open(self.files[index]) as image: image = image.convert("RGB")
        return self.transform(image), self.files[index].name


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__(); self.net = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3), nn.InstanceNorm2d(channels), nn.ReLU(True), nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3), nn.InstanceNorm2d(channels))

    def forward(self, x): return x + self.net(x)


class Generator(nn.Module):
    def __init__(self, blocks: int = 9):
        super().__init__(); layers = [nn.ReflectionPad2d(3), nn.Conv2d(3, 64, 7), nn.InstanceNorm2d(64), nn.ReLU(True)]; channels = 64
        for _ in range(2): layers += [nn.Conv2d(channels, channels * 2, 3, 2, 1), nn.InstanceNorm2d(channels * 2), nn.ReLU(True)]; channels *= 2
        layers += [ResBlock(channels) for _ in range(blocks)]
        for _ in range(2): layers += [nn.ConvTranspose2d(channels, channels // 2, 3, 2, 1, output_padding=1), nn.InstanceNorm2d(channels // 2), nn.ReLU(True)]; channels //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(channels, 3, 7), nn.Tanh()]; self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)


class PatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__(); layers = [nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(.2, True)]; channels = 64
        for out_channels in [128, 256, 512]: layers += [nn.Conv2d(channels, out_channels, 4, 2, 1), nn.InstanceNorm2d(out_channels), nn.LeakyReLU(.2, True)]; channels = out_channels
        layers += [nn.Conv2d(channels, 1, 4, 1, 1)]; self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)


class ReplayBuffer:
    def __init__(self, max_size: int): self.max_size, self.images = max_size, []

    def push_and_pop(self, batch: torch.Tensor) -> torch.Tensor:
        returned = []
        for image in batch.detach():
            image = image.unsqueeze(0)
            if len(self.images) < self.max_size:
                self.images.append(image); returned.append(image)
            elif random.random() < .5:
                index = random.randrange(self.max_size); returned.append(self.images[index].clone()); self.images[index] = image
            else: returned.append(image)
        return torch.cat(returned, 0)


def save_tensor_image(tensor: torch.Tensor, path: Path) -> None:
    array = ((tensor.detach().cpu().clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).numpy()
    Image.fromarray(array).save(path, quality=95)


def save_checkpoint(state: dict, path: Path) -> None:
    """Write checkpoints atomically so an interruption cannot corrupt the latest file."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def content_cosine(a, b):
    va, vb = a.mean((2, 3)).flatten(1), b.mean((2, 3)).flatten(1)
    return float(nn.functional.cosine_similarity(va, vb).mean())


def linear_decay(epoch: int, total_epochs: int, decay_start: int) -> float:
    if epoch < decay_start: return 1.0
    return max(0.0, 1.0 - (epoch - decay_start) / max(1, total_epochs - decay_start))


def make_loader(root, cfg, device, train):
    dataset = ImageDomain(Path(root), cfg["image_size"], train, cfg.get("crop_margin", 30), cfg.get("random_flip", True))
    kwargs = {"num_workers": cfg["num_workers"], "pin_memory": bool(cfg.get("pin_memory", True) and device.type == "cuda")}
    if cfg["num_workers"] > 0: kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(dataset, cfg["batch_size"], shuffle=train, drop_last=train, **kwargs)


def export_predictions(generator_ab, generator_ba, root_a, root_b, cfg, device, out, audit_limit=30):
    eval_a = DataLoader(ImageDomain(root_a, cfg["image_size"], False, 0, False), 1, shuffle=False, num_workers=cfg.get("eval_workers", 1))
    eval_b = DataLoader(ImageDomain(root_b, cfg["image_size"], False, 0, False), 1, shuffle=False, num_workers=cfg.get("eval_workers", 1))
    pred_ab, pred_ba, recon_a, recon_b = out / "pred_A2B", out / "pred_B2A", out / "recon_A", out / "recon_B"
    for folder in [pred_ab, pred_ba, recon_a, recon_b]: folder.mkdir(exist_ok=True)
    l1_a, l1_b, cosine_a, cosine_b, audit = [], [], [], [], []
    generator_ab.eval(); generator_ba.eval()
    with torch.inference_mode():
        for index, (real_a, _) in enumerate(eval_a):
            if index >= cfg.get("export_limit", 10000): break
            real_a = real_a.to(device, non_blocking=True)
            with autocast_context(device, cfg):
                fake_b = generator_ab(real_a); rec_a = generator_ba(fake_b)
            name = f"{index:06d}.jpg"; save_tensor_image(fake_b[0], pred_ab / name); save_tensor_image(rec_a[0], recon_a / name); l1_a.append(float(nn.functional.l1_loss(rec_a, real_a))); cosine_a.append(content_cosine(real_a, fake_b))
        for index, (real_b, _) in enumerate(eval_b):
            if index >= cfg.get("export_limit", 10000): break
            real_b = real_b.to(device, non_blocking=True)
            with autocast_context(device, cfg):
                fake_a = generator_ba(real_b); rec_b = generator_ab(fake_a)
            name = f"{index:06d}.jpg"; save_tensor_image(fake_a[0], pred_ba / name); save_tensor_image(rec_b[0], recon_b / name); l1_b.append(float(nn.functional.l1_loss(rec_b, real_b))); cosine_b.append(content_cosine(real_b, fake_a))
            if index < audit_limit: audit.append({"sample_id": name, "direction": "B2A_photo_to_monet", "style_score_1": "", "style_score_2": "", "content_score_1": "", "content_score_2": "", "artifact_score_1": "", "artifact_score_2": ""})
    if cfg.get("make_kaggle_zip", True):
        with ZipFile(out / cfg.get("submission_zip", "images.zip"), "w", ZIP_DEFLATED) as archive:
            for path in sorted(pred_ba.glob("*.jpg")): archive.write(path, arcname=path.name)
    return {"cycle_reconstruction_l1_A": float(np.mean(l1_a)), "cycle_reconstruction_l1_B": float(np.mean(l1_b)), "content_preservation_cosine_A2B": float(np.mean(cosine_a)), "content_preservation_cosine_B2A": float(np.mean(cosine_b)), "exported_A2B": len(l1_a), "exported_B2A": len(l1_b)}, audit


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); ap.add_argument("--checkpoint-dir", default=None); ap.add_argument("--resume-latest", action="store_true"); args = ap.parse_args(); cfg = yaml.safe_load(Path(args.config).read_text()); seed_everything(cfg["seed"])
    root = Path.cwd(); out = root / "outputs"; ckpt = Path(args.checkpoint_dir or cfg.get("checkpoint_dir", "checkpoints")); ckpt = ckpt if ckpt.is_absolute() else root / ckpt; out.mkdir(exist_ok=True); ckpt.mkdir(parents=True, exist_ok=True)
    data_root = Path(cfg["data_root"]); root_a = next((p for p in [data_root / "monet_jpg", data_root / "monet", data_root / "A"] if p.exists()), None); root_b = next((p for p in [data_root / "photo_jpg", data_root / "photo", data_root / "B"] if p.exists()), None)
    if root_a is None or root_b is None: raise FileNotFoundError("Expected Monet and photo folders under data_root")
    device = device_from(cfg["device"]); torch.set_float32_matmul_precision("high")
    if device.type == "cuda": torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.benchmark = True; torch.cuda.reset_peak_memory_stats(device)
    loader_a, loader_b = make_loader(root_a, cfg, device, True), make_loader(root_b, cfg, device, True)
    iter_a, iter_b = iter(loader_a), iter(loader_b)
    G_AB, G_BA, D_A, D_B = Generator(cfg.get("residual_blocks", 9)).to(device), Generator(cfg.get("residual_blocks", 9)).to(device), PatchDiscriminator().to(device), PatchDiscriminator().to(device)
    opt_g = torch.optim.Adam(list(G_AB.parameters()) + list(G_BA.parameters()), lr=cfg["learning_rate"], betas=(cfg["beta1"], cfg["beta2"])); opt_da = torch.optim.Adam(D_A.parameters(), lr=cfg["learning_rate"], betas=(cfg["beta1"], cfg["beta2"])); opt_db = torch.optim.Adam(D_B.parameters(), lr=cfg["learning_rate"], betas=(cfg["beta1"], cfg["beta2"]))
    schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, lambda epoch: linear_decay(epoch, cfg["epochs"], cfg["decay_start_epoch"])) for opt in [opt_g, opt_da, opt_db]]; scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and cfg.get("amp", True) and cfg.get("precision", "bfloat16").lower() == "float16")); adv, cycle, identity = nn.MSELoss(), nn.L1Loss(), nn.L1Loss(); buffer_a, buffer_b = ReplayBuffer(cfg.get("replay_buffer_size", 50)), ReplayBuffer(cfg.get("replay_buffer_size", 50)); history, grad_norms, nan_count, images_seen, start = [], [], 0, 0, time.time(); start_epoch = 1
    latest = ckpt / "cyclegan_latest.pt"
    if args.resume_latest and latest.exists():
        state = torch.load(latest, map_location=device); G_AB.load_state_dict(state["G_AB"]); G_BA.load_state_dict(state["G_BA"]); D_A.load_state_dict(state["D_A"]); D_B.load_state_dict(state["D_B"]); opt_g.load_state_dict(state["opt_g"]); opt_da.load_state_dict(state["opt_da"]); opt_db.load_state_dict(state["opt_db"])
        for scheduler, saved in zip(schedulers, state.get("schedulers", [])): scheduler.load_state_dict(saved)
        history = state.get("history", []); start_epoch = int(state.get("epoch", 0)) + 1; print(json.dumps({"resumed_from": str(latest), "start_epoch": start_epoch}), flush=True)
    samples_per_epoch = int(cfg.get("samples_per_epoch", len(loader_a) * cfg["batch_size"]))
    steps_per_epoch = max(1, math.ceil(samples_per_epoch / cfg["batch_size"]))
    checkpoint_every_batches = int(cfg.get("checkpoint_every_batches", 0))
    print(json.dumps({"device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU", "domain_A": str(root_a), "domain_B": str(root_b), "image_size": cfg["image_size"], "epochs": cfg["epochs"], "batches_per_epoch": steps_per_epoch, "samples_per_epoch": samples_per_epoch, "amp": cfg.get("amp", True), "precision": cfg.get("precision", "bfloat16")}), flush=True)
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        sums, batches = {"g": 0., "d_a": 0., "d_b": 0., "cycle": 0., "identity": 0.}, 0
        for batch_index in range(1, steps_per_epoch + 1):
            try: real_a, _ = next(iter_a)
            except StopIteration: iter_a = iter(loader_a); real_a, _ = next(iter_a)
            try: real_b, _ = next(iter_b)
            except StopIteration: iter_b = iter(loader_b); real_b, _ = next(iter_b)
            real_a, real_b = real_a.to(device, non_blocking=True), real_b.to(device, non_blocking=True)
            opt_g.zero_grad(set_to_none=True)
            with autocast_context(device, cfg):
                fake_b, fake_a = G_AB(real_a), G_BA(real_b); rec_a, rec_b = G_BA(fake_b), G_AB(fake_a); id_a, id_b = G_BA(real_a), G_AB(real_b)
                pred_fake_b, pred_fake_a = D_B(fake_b), D_A(fake_a)
                loss_g = adv(pred_fake_b, torch.ones_like(pred_fake_b)) + adv(pred_fake_a, torch.ones_like(pred_fake_a)) + cfg["lambda_cycle"] * (cycle(rec_a, real_a) + cycle(rec_b, real_b)) + cfg["lambda_identity"] * (identity(id_a, real_a) + identity(id_b, real_b))
            nan_count += int(not torch.isfinite(loss_g).item()); scaler.scale(loss_g).backward(); scaler.unscale_(opt_g); grad_norms.append(float(torch.nn.utils.clip_grad_norm_(list(G_AB.parameters()) + list(G_BA.parameters()), cfg.get("gradient_clip_norm", 5.0)))); scaler.step(opt_g); scaler.update()
            opt_da.zero_grad(set_to_none=True); fake_a_replay = buffer_a.push_and_pop(fake_a)
            with autocast_context(device, cfg):
                pred_real_a, pred_old_a = D_A(real_a), D_A(fake_a_replay); loss_da = .5 * (adv(pred_real_a, torch.ones_like(pred_real_a)) + adv(pred_old_a, torch.zeros_like(pred_old_a)))
            scaler.scale(loss_da).backward(); scaler.step(opt_da); scaler.update()
            opt_db.zero_grad(set_to_none=True); fake_b_replay = buffer_b.push_and_pop(fake_b)
            with autocast_context(device, cfg):
                pred_real_b, pred_old_b = D_B(real_b), D_B(fake_b_replay); loss_db = .5 * (adv(pred_real_b, torch.ones_like(pred_real_b)) + adv(pred_old_b, torch.zeros_like(pred_old_b)))
            scaler.scale(loss_db).backward(); scaler.step(opt_db); scaler.update()
            sums["g"] += loss_g.item(); sums["d_a"] += loss_da.item(); sums["d_b"] += loss_db.item(); sums["cycle"] += (cycle(rec_a, real_a) + cycle(rec_b, real_b)).item(); sums["identity"] += (identity(id_a, real_a) + identity(id_b, real_b)).item(); batches += 1; images_seen += int(real_a.size(0) + real_b.size(0))
            if checkpoint_every_batches > 0 and batch_index % checkpoint_every_batches == 0:
                mid_epoch_state = {"G_AB": G_AB.state_dict(), "G_BA": G_BA.state_dict(), "D_A": D_A.state_dict(), "D_B": D_B.state_dict(), "opt_g": opt_g.state_dict(), "opt_da": opt_da.state_dict(), "opt_db": opt_db.state_dict(), "schedulers": [scheduler.state_dict() for scheduler in schedulers], "epoch": epoch - 1, "config": cfg, "history": history}
                save_checkpoint(mid_epoch_state, latest)
                print(json.dumps({"checkpoint": str(latest), "epoch": epoch, "batch": batch_index}), flush=True)
            if batch_index % cfg.get("log_interval_batches", 250) == 0: print(json.dumps({"epoch": epoch, "batch": batch_index, "batches_per_epoch": steps_per_epoch, "g_loss": loss_g.item(), "lr": opt_g.param_groups[0]["lr"]}), flush=True)
        for scheduler in schedulers: scheduler.step()
        row = {"epoch": epoch, **{key: value / max(1, batches) for key, value in sums.items()}, "lr": opt_g.param_groups[0]["lr"], "elapsed_seconds": time.time() - start}; history.append(row); (out / "history.json").write_text(json.dumps(history, indent=2)); print(json.dumps(row), flush=True)
        checkpoint = {"G_AB": G_AB.state_dict(), "G_BA": G_BA.state_dict(), "D_A": D_A.state_dict(), "D_B": D_B.state_dict(), "opt_g": opt_g.state_dict(), "opt_da": opt_da.state_dict(), "opt_db": opt_db.state_dict(), "schedulers": [scheduler.state_dict() for scheduler in schedulers], "epoch": epoch, "config": cfg, "history": history}
        save_checkpoint(checkpoint, latest)
        if epoch == 1 or epoch % cfg["save_every"] == 0 or epoch == cfg["epochs"]: save_checkpoint(checkpoint, ckpt / f"cyclegan_epoch_{epoch:03d}.pt")
    cycle_metrics, audit_rows = export_predictions(G_AB, G_BA, root_a, root_b, cfg, device, out); elapsed = time.time() - start; metrics = {**cycle_metrics, "generator_gradient_norm_mean": float(np.mean(grad_norms)), "generator_gradient_norm_max": float(np.max(grad_norms)), "gradient_nan_count": nan_count, "parameter_count_generators": sum(p.numel() for p in list(G_AB.parameters()) + list(G_BA.parameters())), "parameter_count_discriminators": sum(p.numel() for p in list(D_A.parameters()) + list(D_B.parameters())), "training_time_seconds": elapsed, "images_per_second": images_seen / max(elapsed, 1e-9), "peak_memory_mb": (torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0), "device": str(device), "fid_kid_precision_recall": "run evaluate_metrics.py for both directions", "lpips": "run evaluate_lpips.py for both reconstruction folders", "kaggle_submission": str(out / cfg.get("submission_zip", "images.zip")), "human_audit": "complete human_audit.csv"}; (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with (out / "human_audit.csv").open("w", newline="") as stream: writer = csv.DictWriter(stream, fieldnames=audit_rows[0].keys() if audit_rows else ["sample_id"]); writer.writeheader(); writer.writerows(audit_rows)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__": main()
