"""Reproduce the official Part 3 FID/MiFID submission calculation locally."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import scipy.linalg
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from scipy.spatial.distance import cosine
from torch import nn
from tqdm import tqdm

EXTENSIONS = {".jpg", ".jpeg", ".png"}


def image_paths(folder: Path, limit: int) -> list[Path]:
    paths = sorted(p for p in folder.glob("**/*") if not p.name.startswith("._") and p.name != ".DS_Store" and p.suffix.lower() in EXTENSIONS)
    return paths[: min(limit, len(paths))]


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-monet", type=Path, required=True)
    ap.add_argument("--real-photo", type=Path, required=True)
    ap.add_argument("--gen-a2b", type=Path, required=True)
    ap.add_argument("--gen-b2a", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--json-out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    groups = {name: image_paths(path, args.limit) for name, path in {
        "real_monet": args.real_monet, "real_photo": args.real_photo,
        "gen_a2b": args.gen_a2b, "gen_b2a": args.gen_b2a,
    }.items()}
    for name, paths in groups.items():
        if not paths:
            raise SystemExit(f"No images found for {name}")
        print(f"{name}: {len(paths)}")

    model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
    model.fc = nn.Identity()
    model.to(device).eval()
    transform = T.Compose([
        T.Resize(299), T.CenterCrop(299), T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    @torch.no_grad()
    def activations(paths: list[Path]) -> np.ndarray:
        result = []
        for start in tqdm(range(0, len(paths), args.batch_size), desc="Inception activations"):
            batch = torch.stack([transform(Image.open(p).convert("RGB")) for p in paths[start:start + args.batch_size]]).to(device)
            result.append(model(batch).detach().cpu().numpy())
        return np.concatenate(result, axis=0)

    def calculate(real: list[Path], generated: list[Path]) -> tuple[float, float]:
        n = min(len(real), len(generated))
        real_features, generated_features = activations(real[:n]), activations(generated[:n])
        mu_r, sig_r = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu_g, sig_g = generated_features.mean(axis=0), np.cov(generated_features, rowvar=False)
        fid = frechet_distance(mu_r, sig_r, mu_g, sig_g)
        mifid = float(np.mean([cosine(real_features[i], generated_features[i]) for i in range(n)]))
        return fid, mifid

    fid_b2a, mifid_b2a = calculate(groups["real_monet"], groups["gen_b2a"])
    fid_a2b, mifid_a2b = calculate(groups["real_photo"], groups["gen_a2b"])
    metrics = {
        "protocol": "Part3_Evaluation_Script.ipynb", "limit_per_set": args.limit,
        "device": str(device), "FID_B2A": fid_b2a, "MiFID_B2A": mifid_b2a,
        "FID_A2B": fid_a2b, "MiFID_A2B": mifid_a2b,
        "FID": (fid_a2b + fid_b2a) / 2, "MiFID": (mifid_a2b + mifid_b2a) / 2,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(metrics, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ID", "FID", "MiFID"])
        writer.writeheader()
        writer.writerow({"ID": 1, "FID": metrics["FID"], "MiFID": metrics["MiFID"]})
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
