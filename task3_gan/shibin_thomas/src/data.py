"""Task 3.1.1 -- the two unpaired image domains.

A = Monet paintings (task3_gan/data/monet_jpg, 300 images), B = photos (task3_gan/data/photo_jpg, 7,038).
Each training sample is an independent random (Monet, photo) pair -- there is no correspondence
between the two images, which is exactly the unpaired setting CycleGAN is designed for.

Training augmentation (CycleGAN "resize_and_crop"): resize to load_size (286) with bicubic interpolation,
random 256x256 crop, random horizontal flip; pixels scaled to [-1, 1] (tanh range).
One epoch = `samples_per_epoch` pairs. Monet images are drawn by cycling through shuffled permutations
(every painting is used equally often); photos are drawn from shuffled permutations of the 7,038 photos.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

EXTS = (".jpg", ".jpeg", ".png")


def list_images(folder: str | Path) -> list[str]:
    """Same rule as the instructor's script: *.jpg/*.jpeg/*.png (any case) directly in the folder, sorted."""
    paths = []
    for ext in EXTS:
        paths += glob.glob(os.path.join(str(folder), f"*{ext}")) + glob.glob(os.path.join(str(folder), f"*{ext.upper()}"))
    return sorted(set(paths))


def train_transform(load_size: int, crop_size: int, flip: bool = True):
    t = [T.Resize(load_size, interpolation=T.InterpolationMode.BICUBIC), T.RandomCrop(crop_size)]
    if flip:
        t.append(T.RandomHorizontalFlip())
    return T.Compose(t + [T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])


def eval_transform(size: int):
    return T.Compose([T.Resize(size, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(size),
                      T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])


def load_image(path: str, tf) -> torch.Tensor:
    with Image.open(path) as im:
        return tf(im.convert("RGB"))


def to_pil(t: torch.Tensor) -> Image.Image:
    """[-1, 1] CHW tensor -> PIL image."""
    arr = ((t.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()
    return Image.fromarray(arr)


class UnpairedDataset(Dataset):
    def __init__(self, paths_a: list[str], paths_b: list[str], samples_per_epoch: int, tf, seed: int = 0):
        assert paths_a and paths_b, "both domains need at least one image"
        self.paths_a, self.paths_b, self.n, self.tf, self.seed = paths_a, paths_b, samples_per_epoch, tf, seed
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        rng = np.random.default_rng(self.seed * 100_003 + epoch)
        reps_a = -(-self.n // len(self.paths_a))
        reps_b = -(-self.n // len(self.paths_b))
        self.idx_a = np.concatenate([rng.permutation(len(self.paths_a)) for _ in range(reps_a)])[: self.n]
        self.idx_b = np.concatenate([rng.permutation(len(self.paths_b)) for _ in range(reps_b)])[: self.n]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        return load_image(self.paths_a[self.idx_a[i]], self.tf), load_image(self.paths_b[self.idx_b[i]], self.tf)


class FolderDataset(Dataset):
    """Deterministic, sorted images of one folder (for translation / evaluation)."""

    def __init__(self, paths: list[str], size: int):
        self.paths, self.tf = paths, eval_transform(size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return load_image(self.paths[i], self.tf), i
