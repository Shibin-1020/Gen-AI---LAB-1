"""Task 3 live demo: translate images with Shibin's trained CycleGAN generators.

    python demo/demo_task3_translate.py                       # the sample images in demo/sample_inputs
    python demo/demo_task3_translate.py --images my_photo.jpg  # any photo -> Monet style

G_BA turns photos into Monet-style paintings, G_AB turns Monet paintings into photos. Each output grid shows
input | translation | reconstruction (the image translated back), and is saved to demo/outputs/.
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MEMBER = ROOT / "task3_gan/shibin_thomas"
sys.path.insert(0, str(MEMBER / "src"))
from data import eval_transform, to_pil  # noqa: E402
from translate import load_generator  # noqa: E402

CK = MEMBER / "checkpoints/cyclegan_v1_20261001-201801"
SAMPLES = Path(__file__).resolve().parent / "sample_inputs"
OUT = Path(__file__).resolve().parent / "outputs"


def grid(rows):
    w = 256
    g = Image.new("RGB", (3 * w + 16, len(rows) * (w + 8)), "white")
    for r, ims in enumerate(rows):
        for c, im in enumerate(ims):
            g.paste(im.resize((w, w)), (c * (w + 8), r * (w + 8)))
    return g


def run(G, G_back, paths, tf):
    rows = []
    with torch.no_grad():
        for p in paths:
            x = tf(Image.open(p).convert("RGB")).unsqueeze(0)
            y = G(x)
            rec = G_back(y)
            rows.append([to_pil(x[0]), to_pil(y[0]), to_pil(rec[0])])
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--images", nargs="*", help="your own photos (translated photo -> Monet)")
    a = p.parse_args()
    t0 = time.time()
    G_AB = load_generator(CK / "G_AB.pt", torch.device("cpu"))
    G_BA = load_generator(CK / "G_BA.pt", torch.device("cpu"))
    print(f"Loaded G_AB (Monet->photo) and G_BA (photo->Monet) from {CK.relative_to(ROOT)} in {time.time() - t0:.1f}s")
    tf = eval_transform(256)
    OUT.mkdir(exist_ok=True)
    jobs = [("photo_to_monet", G_BA, G_AB, a.images)] if a.images else [
        ("photo_to_monet", G_BA, G_AB, sorted(SAMPLES.glob("photo_*.jpg"))),
        ("monet_to_photo", G_AB, G_BA, sorted(SAMPLES.glob("monet_*.jpg")))]
    for name, G, G_back, paths in jobs:
        t1 = time.time()
        g = grid(run(G, G_back, paths, tf))
        path = OUT / f"{name}.png"
        g.save(path)
        print(f"{name}: {len(paths)} images in {time.time() - t1:.1f}s -> {path.relative_to(ROOT)} "
              f"(columns: input | translation | reconstruction)")
        try:
            g.show()
        except Exception:
            pass


if __name__ == "__main__":
    main()
