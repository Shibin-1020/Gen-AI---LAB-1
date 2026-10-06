"""Fast Task 3 preflight: data, model shapes, finite losses, and one backward pass."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from train import Generator, ImageDomain, PatchDiscriminator, autocast_context, device_from


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    device = device_from(cfg.get("device", "auto"))
    data_root = Path(cfg["data_root"])
    root_a = next((p for p in [data_root / "monet_jpg", data_root / "monet", data_root / "A"] if p.exists()), None)
    root_b = next((p for p in [data_root / "photo_jpg", data_root / "photo", data_root / "B"] if p.exists()), None)
    assert root_a and root_b, "Monet and photo folders must exist before the full run."
    ds_a = ImageDomain(root_a, 64, False, 0, False)
    ds_b = ImageDomain(root_b, 64, False, 0, False)
    assert len(ds_a) > 0 and len(ds_b) > 0, "Both image domains must be non-empty."
    real_a, _ = next(iter(DataLoader(ds_a, batch_size=1, num_workers=0)))
    real_b, _ = next(iter(DataLoader(ds_b, batch_size=1, num_workers=0)))
    real_a, real_b = real_a.to(device), real_b.to(device)
    generator_ab = Generator(blocks=1).to(device)
    generator_ba = Generator(blocks=1).to(device)
    discriminator = PatchDiscriminator().to(device)
    with autocast_context(device, cfg):
        fake_b = generator_ab(real_a)
        recon_a = generator_ba(fake_b)
        score = discriminator(fake_b)
        loss = nn.functional.l1_loss(recon_a, real_a) + score.square().mean()
    assert fake_b.shape == real_a.shape == recon_a.shape, "Generator shape check failed."
    assert torch.isfinite(loss).item(), "Smoke loss is not finite."
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in generator_ab.parameters()), "Backward pass failed."
    print(f"PASS: Task 3 smoke test; device={device}, domain_A={len(ds_a)}, domain_B={len(ds_b)}, image_shape={tuple(real_a.shape)}")


if __name__ == "__main__":
    main()
