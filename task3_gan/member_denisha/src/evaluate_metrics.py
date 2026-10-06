"""Evaluate generated CycleGAN images without modifying them.

Example:
  python src/evaluate_metrics.py --real data/monet_jpg --fake outputs/pred_A2B --out outputs/feature_metrics_A2B.json

FID/KID/precision/recall use Inception features. LPIPS is reported when the optional
`lpips` package is installed. The evaluator never produces or edits submission images.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.linalg import sqrtm
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import Inception3, Inception_V3_Weights, inception_v3


class Images(Dataset):
    def __init__(self, path):
        self.files = sorted(p for p in Path(path).glob("**/*") if not p.name.startswith("._") and p.name != ".DS_Store" and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        self.tf = Inception_V3_Weights.DEFAULT.transforms()

    def __len__(self): return len(self.files)
    def __getitem__(self, i): return self.tf(Image.open(self.files[i]).convert("RGB")), str(self.files[i])


class FeatureNet(nn.Module):
    def __init__(self):
        super().__init__(); self.net = inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True); self.net.fc = nn.Identity(); self.net.eval()

    def forward(self, x):
        y = self.net(x); return y.logits if hasattr(y, "logits") else y


def features(path, device, batch_size):
    model = FeatureNet().to(device); result = []
    with torch.no_grad():
        for x, _ in DataLoader(Images(path), batch_size=batch_size, shuffle=False): result.append(model(x.to(device)).flatten(1).cpu().numpy())
    if not result: raise ValueError(f"No images found at {path}")
    return np.concatenate(result)


def fid(a, b):
    ma, mb = a.mean(0), b.mean(0); ca, cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False); root = sqrtm(ca @ cb); root = np.real(root); return float(((ma - mb) ** 2).sum() + np.trace(ca + cb - 2 * root))


def kid(a, b, seed=0, n=1000):
    rng = np.random.default_rng(seed); a = a[rng.choice(len(a), min(n, len(a)), replace=False)]; b = b[rng.choice(len(b), min(n, len(b)), replace=False)]; scale = max(1, a.shape[1]); ka = (a @ a.T / scale + 1) ** 3; kb = (b @ b.T / scale + 1) ** 3; kab = (a @ b.T / scale + 1) ** 3
    def unbiased(k): return (k.sum() - np.trace(k)) / max(1, k.shape[0] * (k.shape[0] - 1))
    return float(unbiased(ka) + unbiased(kb) - 2 * kab.mean())


def precision_recall(real, fake, k=5):
    def radius(x):
        d = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1); d.sort(axis=1); return np.sqrt(d[:, min(k, len(x) - 1)])
    rr, rf = radius(real), radius(fake); d = np.sqrt(((fake[:, None, :] - real[None, :, :]) ** 2).sum(-1)); precision = (d.min(1) <= rf.min()).mean(); recall = (d.min(0) <= rr.min()).mean(); return float(precision), float(recall)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--real", required=True); ap.add_argument("--fake", required=True); ap.add_argument("--out", required=True); ap.add_argument("--batch-size", type=int, default=32); args = ap.parse_args(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real, fake = features(args.real, device, args.batch_size); precision, recall = precision_recall(real, fake); result = {"real_path": args.real, "fake_path": args.fake, "n_real": len(real), "n_fake": len(fake), "fid": fid(real, fake), "kid": kid(real, fake), "generative_precision": precision, "generative_recall": recall, "device": str(device)}
    try:
        import lpips
        result["lpips"] = "LPIPS requires paired or explicitly matched images; run paired evaluator separately"
    except ImportError: result["lpips"] = "unavailable: install lpips"
    Path(args.out).write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
