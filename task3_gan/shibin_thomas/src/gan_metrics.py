"""Task 3 evaluation metrics (all computed on Inception-v3 2048-d pool features unless noted).

* Feature extractor and FID / "MiFID" are an exact re-implementation of the instructor's
  Part3_Evaluation_Script.ipynb: torchvision inception_v3 (IMAGENET1K_V1, fc = Identity, eval mode),
  Resize(299) -> CenterCrop(299) -> ImageNet normalisation, first N sorted images of each folder,
  counts matched, FID via scipy sqrtm, "MiFID" = mean cosine distance between real[i] and gen[i].
* KID (Binkowski et al. 2018): unbiased MMD^2 with the cubic polynomial kernel, averaged over random
  subsets (lower is better; unbiased, so usable with only 300 images, unlike FID).
* Generative precision / recall (Kynkaanniemi et al. 2019, k = 3) and density / coverage (Naeem et al.
  2020, k = 5): precision/density = how realistic the generated images are, recall/coverage = how much of
  the real distribution they cover.
* LPIPS (Zhang et al. 2018, AlexNet): perceptual distance between two images (0 = identical).
* Content-preservation cosine similarity: cosine between the Inception features of an input image and
  of its translation (higher = more of the input's content survives the style change).
These pretrained networks are used ONLY to measure images; they never generate or modify the
submitted images (Kaggle integrity rule).
"""
from __future__ import annotations

import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from scipy.spatial.distance import cosine

INCEPTION_TF = T.Compose([T.Resize(299), T.CenterCrop(299), T.ToTensor(),
                          T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


class InceptionFeatures:
    def __init__(self, device: torch.device, pretrained: bool = True, batch_size: int = 32):
        weights = models.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.inception_v3(weights=weights, transform_input=False, aux_logits=True,
                                  init_weights=not pretrained)
        net.fc = nn.Identity()
        self.net = net.to(device).eval()
        self.device, self.bs, self.pretrained = device, batch_size, pretrained

    @torch.no_grad()
    def _run(self, pil_images) -> np.ndarray:
        x = torch.stack([INCEPTION_TF(im) for im in pil_images]).to(self.device)
        return self.net(x).float().cpu().numpy()

    def from_paths(self, paths: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(paths), self.bs):
            ims = []
            for p in paths[i:i + self.bs]:
                with Image.open(p) as im:
                    ims.append(im.convert("RGB"))
            out.append(self._run(ims))
        return np.concatenate(out) if out else np.zeros((0, 2048), np.float32)

    def from_pil(self, images: list[Image.Image]) -> np.ndarray:
        return np.concatenate([self._run(images[i:i + self.bs]) for i in range(0, len(images), self.bs)])


# --------------------------------------------------------------------------- instructor protocol
def _sqrtm(m: np.ndarray) -> np.ndarray:
    """Matrix square root on every SciPy version: before 1.18 sqrtm(..., disp=False) returns (sqrt, error);
    SciPy >= 1.18 removed the `disp` argument and returns only the matrix. The numbers are identical."""
    try:
        out = scipy.linalg.sqrtm(m, disp=False)
    except TypeError:
        out = scipy.linalg.sqrtm(m)
    return out[0] if isinstance(out, tuple) else out


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6) -> float:
    covmean = _sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = _sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean))


def fid_from_features(real: np.ndarray, gen: np.ndarray) -> float:
    return frechet_distance(real.mean(0), np.cov(real, rowvar=False), gen.mean(0), np.cov(gen, rowvar=False))


def instructor_fid_mifid(real: np.ndarray, gen: np.ndarray) -> tuple[float, float]:
    """FID and the script's 'MiFID' (mean paired cosine distance) on index-matched feature sets."""
    n = min(len(real), len(gen))
    real, gen = real[:n], gen[:n]
    return fid_from_features(real, gen), float(np.mean([cosine(real[i], gen[i]) for i in range(n)]))


# --------------------------------------------------------------------------- additional metrics
def kid(real: np.ndarray, gen: np.ndarray, n_subsets: int = 100, subset_size: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)
    m = min(subset_size, len(real), len(gen))
    d = real.shape[1]
    vals = []
    for _ in range(n_subsets):
        x = real[rng.choice(len(real), m, replace=False)].astype(np.float64)
        y = gen[rng.choice(len(gen), m, replace=False)].astype(np.float64)
        kxx, kyy, kxy = (x @ x.T / d + 1) ** 3, (y @ y.T / d + 1) ** 3, (x @ y.T / d + 1) ** 3
        mmd = ((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
               - 2 * kxy.mean())
        vals.append(mmd)
    return float(np.mean(vals)), float(np.std(vals))


def _knn_radius(feats: torch.Tensor, k: int) -> torch.Tensor:
    d = torch.cdist(feats, feats)
    return d.kthvalue(k + 1, dim=1).values                 # +1: distance to itself is 0


def precision_recall_density_coverage(real: np.ndarray, gen: np.ndarray, k_pr: int = 3, k_dc: int = 5) -> dict:
    r, g = torch.from_numpy(real).double(), torch.from_numpy(gen).double()
    d_rg = torch.cdist(r, g)                               # (n_real, n_gen)
    rad_r3, rad_g3 = _knn_radius(r, k_pr), _knn_radius(g, k_pr)
    precision = (d_rg <= rad_r3[:, None]).any(0).double().mean().item()
    recall = (d_rg.T <= rad_g3[:, None]).any(0).double().mean().item()
    rad_r5 = _knn_radius(r, k_dc)
    inside = d_rg <= rad_r5[:, None]
    density = (inside.sum(0).double() / k_dc).mean().item()
    coverage = (d_rg.min(1).values <= rad_r5).double().mean().item()
    return {"precision": precision, "recall": recall, "density": density, "coverage": coverage}


def paired_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return float((a * b).sum(1).mean())


class LPIPSMetric:
    def __init__(self, device: torch.device, pretrained: bool = True):
        import warnings
        import lpips
        with warnings.catch_warnings():                 # lpips calls torchvision's deprecated 'pretrained' API
            warnings.simplefilter("ignore", UserWarning)
            self.fn = lpips.LPIPS(net="alex", pretrained=pretrained, pnet_rand=not pretrained,
                                  verbose=False).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> np.ndarray:
        """x, y: (N, 3, H, W) in [-1, 1]; returns per-image distances."""
        return self.fn(x.to(self.device), y.to(self.device)).flatten().float().cpu().numpy()
