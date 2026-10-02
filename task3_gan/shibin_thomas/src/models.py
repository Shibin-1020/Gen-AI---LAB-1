"""Task 3.1 -- CycleGAN implemented from scratch: two generators, two discriminators.

Domains (as in the instructor's evaluation script): A = Monet paintings, B = photos.
    G_AB : Monet -> Photo  (outputs go to pred_A2B/)      D_A : real vs fake Monet
    G_BA : Photo -> Monet  (outputs go to pred_B2A/)      D_B : real vs fake Photo

Generator (Johnson et al. 2016 / Zhu et al. 2017 "resnet_9blocks"):
    c7s1-64 -> d128 -> d256 -> 9 x residual block(256) -> u128 -> u64 -> c7s1-3 -> tanh
    InstanceNorm, ReLU, reflection padding. Upsampling is configurable: "resize_conv"
    (nearest-neighbour x2 + 3x3 conv, Odena et al. 2016 -- avoids checkerboard artifacts; my default)
    or "deconv" (transposed convolution, as in the original paper).
Discriminator: 70x70 PatchGAN (C64 - C128 - C256 - C512 - 1), LeakyReLU 0.2, InstanceNorm
    (none on the first layer); outputs a grid of real/fake scores, one per overlapping 70x70 patch.
All weights are initialised from N(0, 0.02) -- no pretrained networks are used anywhere in training
or generation.
"""
from __future__ import annotations

import random

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3), nn.InstanceNorm2d(ch), nn.ReLU(True)]
        if dropout:
            layers.append(nn.Dropout(dropout))
        layers += [nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3), nn.InstanceNorm2d(ch)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)                       # residual connection


def _up(in_ch: int, out_ch: int, mode: str) -> list[nn.Module]:
    if mode == "deconv":
        conv = nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=1)
        return [conv, nn.InstanceNorm2d(out_ch), nn.ReLU(True)]
    return [nn.Upsample(scale_factor=2, mode="nearest"), nn.ReflectionPad2d(1), nn.Conv2d(in_ch, out_ch, 3),
            nn.InstanceNorm2d(out_ch), nn.ReLU(True)]


class ResnetGenerator(nn.Module):
    def __init__(self, ngf: int = 64, n_res: int = 9, n_down: int = 2, upsample: str = "resize_conv",
                 dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = [nn.ReflectionPad2d(3), nn.Conv2d(3, ngf, 7), nn.InstanceNorm2d(ngf), nn.ReLU(True)]
        ch = ngf
        for _ in range(n_down):                        # downsampling: 256 -> 128 -> 64
            layers += [nn.Conv2d(ch, ch * 2, 3, stride=2, padding=1), nn.InstanceNorm2d(ch * 2), nn.ReLU(True)]
            ch *= 2
        layers += [ResidualBlock(ch, dropout) for _ in range(n_res)]
        for _ in range(n_down):                        # upsampling: 64 -> 128 -> 256
            layers += _up(ch, ch // 2, upsample)
            ch //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ch, 3, 7), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(3, ndf, 4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        ch = ndf
        for i in range(1, n_layers):
            nxt = ndf * min(2 ** i, 8)
            layers += [nn.Conv2d(ch, nxt, 4, stride=2, padding=1), nn.InstanceNorm2d(nxt), nn.LeakyReLU(0.2, True)]
            ch = nxt
        nxt = ndf * min(2 ** n_layers, 8)
        layers += [nn.Conv2d(ch, nxt, 4, stride=1, padding=1), nn.InstanceNorm2d(nxt), nn.LeakyReLU(0.2, True),
                   nn.Conv2d(nxt, 1, 4, stride=1, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def init_weights(m: nn.Module, std: float = 0.02) -> None:
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, std)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def build_models(mcfg: dict) -> dict[str, nn.Module]:
    g = dict(ngf=mcfg["ngf"], n_res=mcfg["n_res_blocks"], upsample=mcfg.get("upsample", "resize_conv"),
             dropout=mcfg.get("dropout", 0.0))
    nets = {"G_AB": ResnetGenerator(**g), "G_BA": ResnetGenerator(**g),
            "D_A": PatchDiscriminator(mcfg["ndf"], mcfg.get("d_layers", 3)),
            "D_B": PatchDiscriminator(mcfg["ndf"], mcfg.get("d_layers", 3))}
    for n in nets.values():
        n.apply(init_weights)
    return nets


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


class ImagePool:
    """History buffer of generated images (Shrivastava et al. 2017): the discriminator sees a mix of the
    current fakes and older ones, which damps generator/discriminator oscillation."""

    def __init__(self, size: int = 50, seed: int = 0):
        self.size, self.images = size, []
        self.rng = random.Random(seed)

    def query(self, images: torch.Tensor) -> torch.Tensor:
        if self.size == 0:
            return images
        out = []
        for img in images.detach():
            img = img.unsqueeze(0)
            if len(self.images) < self.size:
                self.images.append(img)
                out.append(img)
            elif self.rng.random() < 0.5:
                i = self.rng.randrange(self.size)
                out.append(self.images[i].clone())
                self.images[i] = img
            else:
                out.append(img)
        return torch.cat(out, 0)
