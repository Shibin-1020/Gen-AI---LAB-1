"""Sanity tests for Task 3 (no dataset, GPU or pretrained weights needed).
Run:  python task3_gan/shibin_thomas/tests/test_task3.py"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from data import UnpairedDataset, list_images, train_transform  # noqa: E402
from diffaug import diff_augment  # noqa: E402
from gan_metrics import (fid_from_features, instructor_fid_mifid, kid,  # noqa: E402
                         paired_cosine_similarity, precision_recall_density_coverage)
from models import ImagePool, PatchDiscriminator, ResnetGenerator, build_models, count_params  # noqa: E402
from train import ema_decay_at, ema_update, lr_factor  # noqa: E402


def test_generator_shapes_and_range():
    for up in ("resize_conv", "deconv"):
        G = ResnetGenerator(ngf=8, n_res=2, upsample=up)
        y = G(torch.randn(2, 3, 64, 64))
        assert y.shape == (2, 3, 64, 64) and y.min() >= -1 and y.max() <= 1, up


def test_patchgan_is_70x70_patch_grid():
    D = PatchDiscriminator(ndf=8, n_layers=3)
    assert D(torch.randn(1, 3, 256, 256)).shape == (1, 1, 30, 30)        # 30x30 overlapping 70x70 patches


def test_full_size_parameter_counts():
    nets = build_models({"ngf": 64, "n_res_blocks": 9, "upsample": "resize_conv", "ndf": 64, "d_layers": 3})
    g, d = count_params(nets["G_AB"]), count_params(nets["D_A"])
    assert 11.0e6 < g < 11.6e6, g                                       # ~11.4M (resnet_9blocks)
    assert 2.7e6 < d < 2.8e6, d                                         # ~2.77M (70x70 PatchGAN)


def test_cycle_and_identity_losses_are_wired_correctly():
    """Identity generators reconstruct perfectly: cycle and identity losses must be exactly 0."""
    a, b = torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)
    G_AB = G_BA = torch.nn.Identity()
    assert F.l1_loss(G_BA(G_AB(a)), a) == 0 and F.l1_loss(G_AB(G_BA(b)), b) == 0
    assert F.l1_loss(G_BA(a), a) == 0


def test_cycle_training_reduces_reconstruction_error():
    torch.manual_seed(0)
    G_AB, G_BA = ResnetGenerator(8, 1), ResnetGenerator(8, 1)
    opt = torch.optim.Adam(list(G_AB.parameters()) + list(G_BA.parameters()), lr=2e-3)
    a = torch.rand(2, 3, 32, 32) * 2 - 1
    first = None
    for _ in range(100):
        loss = F.l1_loss(G_BA(G_AB(a)), a)
        first = first if first is not None else loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < 0.5 * first, (first, loss.item())


def test_image_pool():
    pool = ImagePool(size=4, seed=0)
    for _ in range(10):
        out = pool.query(torch.randn(3, 3, 8, 8))
        assert out.shape == (3, 3, 8, 8)
    assert len(pool.images) == 4
    assert ImagePool(0).query(torch.ones(2, 3, 4, 4)).sum() == 2 * 3 * 16


def test_lr_schedule():
    f = [lr_factor(e, 20, 20) for e in range(40)]
    assert all(v == 1.0 for v in f[:20]) and all(x > y for x, y in zip(f[19:], f[20:])) and 0 < f[-1] < 0.1


def test_fid_kid_prdc_known_values():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 8))
    y = rng.normal(size=(2000, 8))
    assert fid_from_features(x, y) < 0.05                                # same distribution -> ~0
    shifted = y + np.array([3.0] + [0.0] * 7)
    assert abs(fid_from_features(x, shifted) - 9.0) < 0.3                # mean shift 3 -> FID ~ 3^2
    k_same, _ = kid(x, y, 20, 100)
    k_shift, _ = kid(x, shifted, 20, 100)
    assert abs(k_same) < 0.05 and k_shift > 0.5
    pr = precision_recall_density_coverage(x[:300], x[:300])
    assert pr["precision"] == 1 and pr["recall"] == 1 and pr["coverage"] == 1
    far = precision_recall_density_coverage(x[:300], x[:300] + 100)
    assert far["precision"] == 0 and far["recall"] == 0 and far["coverage"] == 0
    assert np.isclose(paired_cosine_similarity(x[:10], 3 * x[:10]), 1.0)


def test_fid_works_with_scipy_1_18_sqrtm_api():
    """SciPy >= 1.18 removed sqrtm's `disp` argument and returns a bare array; FID must give the same value."""
    import scipy.linalg
    import gan_metrics
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=(500, 6)), rng.normal(size=(500, 6)) + 1.0
    ref = fid_from_features(x, y)
    orig = scipy.linalg.sqrtm

    def sqrtm_v118(a, **kw):
        if "disp" in kw:
            raise TypeError("sqrtm() got an unexpected keyword argument 'disp'")
        return orig(a, disp=False)[0] if "disp" in orig.__code__.co_varnames else orig(a)
    gan_metrics.scipy.linalg.sqrtm = sqrtm_v118
    try:
        assert np.isclose(fid_from_features(x, y), ref)
    finally:
        gan_metrics.scipy.linalg.sqrtm = orig


def test_matches_instructor_script_protocol():
    """Re-run the instructor's calculate_fid_mifid logic on the same features and compare."""
    from scipy.spatial.distance import cosine
    rng = np.random.default_rng(1)
    real, gen = rng.normal(size=(120, 16)), rng.normal(size=(100, 16)) + 0.5
    n = min(len(real), len(gen))
    r, g = real[:n], gen[:n]
    ref_mifid = float(np.mean([cosine(r[i], g[i]) for i in range(n)]))
    fid, mifid = instructor_fid_mifid(real, gen)
    assert np.isclose(mifid, ref_mifid) and np.isclose(fid, fid_from_features(r, g))


def test_data_listing_and_unpaired_sampling():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for i in range(5):
            Image.new("RGB", (40, 40), (i * 40, 0, 0)).save(tmp / f"{i}.jpg")
        (tmp / "notes.txt").write_text("x")
        (tmp / ".DS_Store").write_text("x")
        paths = list_images(tmp)
        assert [Path(p).name for p in paths] == [f"{i}.jpg" for i in range(5)]
        ds = UnpairedDataset(paths, paths, samples_per_epoch=10, tf=train_transform(36, 32))
        a, b = ds[0]
        assert a.shape == (3, 32, 32) and a.min() >= -1 and a.max() <= 1
        assert sorted(np.bincount(ds.idx_a, minlength=5).tolist()) == [2, 2, 2, 2, 2]   # every painting equally often
        before = ds.idx_a.copy()
        ds.set_epoch(1)
        assert not np.array_equal(before, ds.idx_a)


def test_diffaugment_shapes_gradients_and_policies():
    torch.manual_seed(0)
    x = (torch.rand(4, 3, 32, 32) * 2 - 1).requires_grad_(True)
    assert torch.equal(diff_augment(x, ""), x)                                # empty policy = no-op
    y = diff_augment(x, "color,translation,cutout")
    assert y.shape == x.shape
    y.sum().backward()                                                     # differentiable: G gets gradients
    assert x.grad is not None and x.grad.abs().sum() > 0
    ones = torch.ones(8, 3, 32, 32)
    cut = diff_augment(ones, "cutout")
    frac = (cut == 0).float().mean().item()                                # a 16x16 square of 32x32 = 25%
    assert 0.05 < frac <= 0.25 + 1e-6, frac
    t = diff_augment(ones, "translation")                                  # shift: zeros only at the border
    assert t.max() == 1 and t[:, :, 8:24, 8:24].min() == 1
    try:
        diff_augment(x, "rotate")
        raise AssertionError("unknown op accepted")
    except ValueError:
        pass


def test_generator_ema():
    torch.manual_seed(0)
    G, E = ResnetGenerator(4, 1), ResnetGenerator(4, 1)
    E.load_state_dict(G.state_dict())
    assert ema_decay_at(0, 0.999) == 0.1 and ema_decay_at(10 ** 6, 0.999) == 0.999
    with torch.no_grad():
        for p in G.parameters():
            p.add_(1.0)
    before = [p.clone() for p in E.parameters()]
    ema_update(E, G, 0.9)                                                  # E <- 0.9 E + 0.1 G
    for b, e, g in zip(before, E.parameters(), G.parameters()):
        assert torch.allclose(e, 0.9 * b + 0.1 * g, atol=1e-6)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
