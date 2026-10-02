"""Task 3.1.4 -- translate images between the two domains with the trained generators.

    pred_A2B/  G_AB(Monet)  -> generated photos   (one per Monet painting, same file name)
    pred_B2A/  G_BA(photo)  -> generated Monets   (the first `b2a_count` photos in sorted order)
The instructor's evaluation script scores the first N_EVAL = 300 sorted images of each folder, so the
default export (all 300 Monets, first 300 photos) is exactly what is scored; set export.b2a_count to
null to translate all 7,038 photos.

Images are written as JPEG (quality 95) straight from the generator output -- no editing, selection or
post-processing (Kaggle integrity rule). The canonical folders live in outputs/pred_A2B and
outputs/pred_B2A (as in the brief's repo layout) and record which run produced them.
"""
from __future__ import annotations

import argparse
import shutil

import torch
from torch.utils.data import DataLoader

from data import FolderDataset, list_images, to_pil
from models import ResnetGenerator
from utils import MEMBER_DIR, get_device, latest_run_id, load_config, rel, repo_path, run_dirs, write_json


def load_generator(path, device) -> torch.nn.Module:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = ck["model_config"]
    G = ResnetGenerator(m["ngf"], m["n_res_blocks"], upsample=m.get("upsample", "resize_conv"),
                        dropout=m.get("dropout", 0.0)).to(device)
    G.load_state_dict(ck["model"])
    return G.eval()


@torch.no_grad()
def translate_folder(G, paths: list[str], out_dir, size: int, device, bs: int = 16) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(FolderDataset(paths, size), batch_size=bs, shuffle=False)
    n = 0
    for x, idx in loader:
        y = G(x.to(device)).float().cpu()
        for t, i in zip(y, idx.tolist()):
            name = paths[i].replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0] + ".jpg"
            to_pil(t).save(out_dir / name, quality=95)
            n += 1
    return n


def export(cfg: dict, run_id: str) -> dict:
    device = get_device(cfg.get("device", "auto"))
    d, e = cfg["data"], cfg["export"]
    ck_dir = run_dirs(run_id)["checkpoints"]
    G_AB, G_BA = load_generator(ck_dir / "G_AB.pt", device), load_generator(ck_dir / "G_BA.pt", device)
    paths_a, paths_b = list_images(repo_path(d["monet_dir"])), list_images(repo_path(d["photo_dir"]))
    if e.get("a2b_count"):
        paths_a = paths_a[: e["a2b_count"]]
    if e.get("b2a_count"):
        paths_b = paths_b[: e["b2a_count"]]
    out_root = repo_path(e["out_dir"])
    ck_meta = torch.load(ck_dir / "G_AB.pt", map_location="cpu", weights_only=False)
    info = {"run_id": run_id, "generator_checkpoints": [rel(ck_dir / "G_AB.pt"), rel(ck_dir / "G_BA.pt")],
            "weights": ck_meta.get("weights", f"raw generators, final epoch {ck_meta.get('epoch')}")}
    for name, G, paths in (("pred_A2B", G_AB, paths_a), ("pred_B2A", G_BA, paths_b)):
        tgt = out_root / name
        if tgt.exists():
            shutil.rmtree(tgt)                         # never mix outputs of two runs
        info[name] = {"count": translate_folder(G, paths, tgt, d["crop_size"], device), "dir": rel(tgt)}
        print(f"[translate] {name}: {info[name]['count']} images -> {rel(tgt)}")
    write_json(out_root / "predictions_info.json", info)
    return info


def main() -> None:
    p = argparse.ArgumentParser(description="Task 3 translation / export")
    p.add_argument("--config", default="task3_gan/shibin_thomas/configs/cyclegan_v1.yaml")
    p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--run-id", default="latest")
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    export(cfg, latest_run_id(cfg["run_name"]) if a.run_id == "latest" else a.run_id)


if __name__ == "__main__":
    main()
