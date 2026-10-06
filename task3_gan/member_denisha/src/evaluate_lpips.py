"""Compute LPIPS between inputs and CycleGAN reconstructions when lpips is installed."""
import argparse, json
from pathlib import Path
import torch
from PIL import Image
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument('--real', required=True); ap.add_argument('--recon', required=True); ap.add_argument('--out', required=True); args = ap.parse_args()
try:
    import lpips
except ImportError as exc:
    raise SystemExit('Install the optional lpips package first') from exc
net = lpips.LPIPS(net='alex').eval(); files = sorted(p for p in Path(args.real).glob('**/*') if p.suffix.lower() in {'.jpg','.jpeg','.png'}); scores = []
for path in files:
    other = Path(args.recon) / path.name
    if not other.exists(): continue
    def load(p): return torch.from_numpy(np.asarray(Image.open(p).convert('RGB').resize((256,256)), dtype=np.float32) / 127.5 - 1).permute(2,0,1).unsqueeze(0)
    with torch.no_grad(): scores.append(float(net(load(path), load(other)).item()))
result = {'n_pairs': len(scores), 'lpips_mean': float(np.mean(scores)) if scores else None, 'real': args.real, 'recon': args.recon}; Path(args.out).write_text(json.dumps(result, indent=2)); print(json.dumps(result, indent=2))
