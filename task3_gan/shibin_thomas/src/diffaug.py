"""Differentiable Augmentation for the discriminators (Zhao, Liu, Lin, Zhu & Han, NeurIPS 2020), written
from scratch.

Why: domain A has only 300 Monet paintings, so D_A sees every painting hundreds of times and starts to
memorise them. A memorising discriminator gives the generator poor gradients (in v1, D_A's loss kept
falling, far below the LSGAN balance point of 0.25). DiffAugment applies the SAME random, differentiable
augmentation to every image the discriminator sees, real and fake, in both the D step and the G step:
* the discriminator can no longer memorise exact paintings;
* the generator still receives gradients through the augmentation;
* augmented images are never shown to the generator as targets, so the augmentation does not "leak" into
  the generated images.

Policies (comma-separated):
* color: random brightness, saturation and contrast;
* translation: random shift of up to 1/8 of the image size, zero padding;
* cutout: one random square of half the image size set to 0.
Images are (N, 3, H, W) in [-1, 1]. Training-time only: generation and export never use this module.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _brightness(x):
    return x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)


def _saturation(x):
    mean = x.mean(dim=1, keepdim=True)
    return (x - mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + mean


def _contrast(x):
    mean = x.mean(dim=[1, 2, 3], keepdim=True)
    return (x - mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + mean


def _translation(x, ratio: float = 0.125):
    n, _, h, w = x.shape
    sh, sw = int(h * ratio + 0.5), int(w * ratio + 0.5)
    tx = torch.randint(-sh, sh + 1, size=[n, 1, 1], device=x.device)
    ty = torch.randint(-sw, sw + 1, size=[n, 1, 1], device=x.device)
    gb, gx, gy = torch.meshgrid(torch.arange(n, device=x.device), torch.arange(h, device=x.device),
                                torch.arange(w, device=x.device), indexing="ij")
    gx = torch.clamp(gx + tx + 1, 0, h + 1)
    gy = torch.clamp(gy + ty + 1, 0, w + 1)
    x_pad = F.pad(x, [1, 1, 1, 1])                       # the 1-pixel zero border fills the shifted-in area
    return x_pad.permute(0, 2, 3, 1).contiguous()[gb, gx, gy].permute(0, 3, 1, 2)


def _cutout(x, ratio: float = 0.5):
    n, _, h, w = x.shape
    ch, cw = int(h * ratio + 0.5), int(w * ratio + 0.5)
    ox = torch.randint(0, h + (1 - ch % 2), size=[n, 1, 1], device=x.device)
    oy = torch.randint(0, w + (1 - cw % 2), size=[n, 1, 1], device=x.device)
    gb, gx, gy = torch.meshgrid(torch.arange(n, device=x.device), torch.arange(ch, device=x.device),
                                torch.arange(cw, device=x.device), indexing="ij")
    gx = torch.clamp(gx + ox - ch // 2, min=0, max=h - 1)
    gy = torch.clamp(gy + oy - cw // 2, min=0, max=w - 1)
    mask = torch.ones(n, h, w, dtype=x.dtype, device=x.device)
    mask[gb, gx, gy] = 0
    return x * mask.unsqueeze(1)


OPS = {"color": [_brightness, _saturation, _contrast], "translation": [_translation], "cutout": [_cutout]}


def diff_augment(x: torch.Tensor, policy: str = "") -> torch.Tensor:
    """Apply the policy's augmentations in order; an empty policy returns x unchanged."""
    for name in [p.strip() for p in (policy or "").split(",") if p.strip()]:
        if name not in OPS:
            raise ValueError(f"unknown DiffAugment op '{name}' (choose from {sorted(OPS)})")
        for f in OPS[name]:
            x = f(x)
    return x
