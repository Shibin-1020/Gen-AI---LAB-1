"""Task 3.2.6 -- blinded human audit of 30 fixed samples with 2 raters + inter-rater agreement.

    python task3_gan/shibin_thomas/src/human_audit.py make     # build the audit pack (once, after export)
    python task3_gan/shibin_thomas/src/human_audit.py score    # after both raters filled in their sheets

make : picks 30 FIXED samples with a fixed seed (15 photo->Monet, 15 Monet->photo) from the exported
       predictions, renders each as [input | output] side by side, shuffles them and names them S01..S30.
       The key (which file / direction) goes to audit_key.csv -- raters must not open it (blinded).
       It also writes rater1.csv and rater2.csv to fill in, plus a half-size overview (contact_sheet.jpg) of all 30 panels.
       Scores, 1 (worst) - 5 (best):
         style     does the output look like the TARGET domain (a real Monet painting / a real photo)?
         content   is the scene of the input preserved (layout, objects, shapes)?
         artifacts 5 = clean, 1 = severe artifacts (checkerboard, colour blotches, smearing, noise)
score: mean score per criterion (overall and per direction) and inter-rater agreement per criterion:
       Cohen's kappa with quadratic weights (ordinal scale), exact % agreement and % within +-1.
"""
from __future__ import annotations

import argparse
import csv
import random

import numpy as np
from PIL import Image, ImageDraw

from data import list_images
from utils import MEMBER_DIR, read_json, rel, repo_path, write_json

CRITERIA = ["style", "content", "artifacts"]


def make(n_per_dir: int = 15, seed: int = 266, pred_root=None, monet_dir="task3_gan/data/monet_jpg",
         photo_dir="task3_gan/data/photo_jpg") -> None:
    pred_root = pred_root or MEMBER_DIR / "outputs"
    AUDIT_DIR = pred_root / "human_audit"
    info = read_json(pred_root / "predictions_info.json")
    rng = random.Random(seed)
    items = []
    for direction, pred, src in (("photo->monet", "pred_B2A", photo_dir), ("monet->photo", "pred_A2B", monet_dir)):
        gen = sorted(p.name for p in (pred_root / pred).glob("*.jpg"))
        src_map = {p.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]: p for p in list_images(repo_path(src))}
        for name in rng.sample(gen, min(n_per_dir, len(gen))):
            items.append((direction, src_map[name.rsplit(".", 1)[0]], pred_root / pred / name))
    rng.shuffle(items)
    (AUDIT_DIR / "images").mkdir(parents=True, exist_ok=True)
    panels = []
    with open(AUDIT_DIR / "audit_key.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "direction", "input_file", "output_file"])
        for k, (direction, src, out) in enumerate(items, 1):
            sid = f"S{k:02d}"
            a, b = Image.open(src).convert("RGB").resize((256, 256)), Image.open(out).convert("RGB").resize((256, 256))
            panel = Image.new("RGB", (520, 280), "white")
            panel.paste(a, (0, 24)), panel.paste(b, (264, 24))
            ImageDraw.Draw(panel).text((4, 4), f"{sid}   input  |  output", fill="black")
            panel.save(AUDIT_DIR / "images" / f"{sid}.png")
            panels.append(panel)
            w.writerow([sid, direction, rel(src), rel(out)])
    for r in ("rater1", "rater2"):
        p = AUDIT_DIR / f"{r}.csv"
        if not p.exists():                              # never overwrite ratings already entered
            with open(p, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["sample_id", "style", "content", "artifacts", "comment"])
                for k in range(1, len(items) + 1):
                    w.writerow([f"S{k:02d}", "", "", "", ""])
    cols = 5
    sheet = Image.new("RGB", (cols * 520, -(-len(panels) // cols) * 280), "white")
    for k, pnl in enumerate(panels):
        sheet.paste(pnl, ((k % cols) * 520, (k // cols) * 280))
    sheet = sheet.resize((sheet.width // 2, sheet.height // 2), Image.LANCZOS)   # overview only; full-size panels in images/
    sheet.save(AUDIT_DIR / "contact_sheet.jpg", quality=85)
    (AUDIT_DIR / "INSTRUCTIONS.md").write_text(
        "# Blinded human audit (30 fixed samples, 2 raters)\n\n"
        f"Predictions from run `{info['run_id']}`. Open `images/S01.png` ... `S30.png` (or the overview `contact_sheet.jpg`). "
        "Each panel shows the INPUT (left) and the model OUTPUT (right). Do **not** open `audit_key.csv`.\n\n"
        "Fill in your own sheet (`rater1.csv` or `rater2.csv`) independently -- do not discuss scores before both are "
        "done. Integers 1-5:\n\n"
        "| Criterion | 1 | 5 |\n|---|---|---|\n"
        "| style | output does not look like the target domain | indistinguishable from a real Monet / real photo |\n"
        "| content | scene of the input lost | layout, objects and shapes fully preserved |\n"
        "| artifacts | severe checkerboard / blotches / smearing / noise | clean |\n\n"
        "Then run `python task3_gan/shibin_thomas/src/human_audit.py score`.\n", encoding="utf-8")
    print(f"[audit] {len(items)} samples -> {rel(AUDIT_DIR)} (fill rater1.csv / rater2.csv, then run 'score')")


def _read(path) -> dict[str, dict[str, int]]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if all((row.get(c) or "").strip() for c in CRITERIA):
                out[row["sample_id"]] = {c: int(row[c]) for c in CRITERIA}
    return out


def score(pred_root=None) -> dict:
    from sklearn.metrics import cohen_kappa_score
    AUDIT_DIR = (pred_root or MEMBER_DIR / "outputs") / "human_audit"
    r1, r2 = _read(AUDIT_DIR / "rater1.csv"), _read(AUDIT_DIR / "rater2.csv")
    key = {}
    with open(AUDIT_DIR / "audit_key.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key[row["sample_id"]] = row["direction"]
    ids = sorted(set(r1) & set(r2))
    if not ids:
        raise SystemExit("no sample has been scored by both raters yet")
    res = {"n_samples_scored_by_both": len(ids), "criteria": {}}
    for c in CRITERIA:
        a, b = np.array([r1[i][c] for i in ids]), np.array([r2[i][c] for i in ids])
        both = (a + b) / 2
        res["criteria"][c] = {
            "mean_rater1": float(a.mean()), "mean_rater2": float(b.mean()), "mean_both": float(both.mean()),
            "mean_photo->monet": float(np.mean([both[k] for k, i in enumerate(ids) if key[i] == "photo->monet"] or [np.nan])),
            "mean_monet->photo": float(np.mean([both[k] for k, i in enumerate(ids) if key[i] == "monet->photo"] or [np.nan])),
            "cohen_kappa_quadratic": float(cohen_kappa_score(a, b, weights="quadratic", labels=[1, 2, 3, 4, 5])),
            "agreement_exact": float((a == b).mean()), "agreement_within_1": float((np.abs(a - b) <= 1).mean())}
    res["overall_mean_score"] = float(np.mean([res["criteria"][c]["mean_both"] for c in CRITERIA]))
    write_json(AUDIT_DIR / "human_audit_results.json", res)
    for c, v in res["criteria"].items():
        print(f"{c:9s} mean {v['mean_both']:.2f} (P->M {v['mean_photo->monet']:.2f}, M->P {v['mean_monet->photo']:.2f}) "
              f"kappa_w {v['cohen_kappa_quadratic']:.3f} exact {v['agreement_exact']:.0%} within1 {v['agreement_within_1']:.0%}")
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["make", "score"])
    a = p.parse_args()
    make() if a.action == "make" else score()


if __name__ == "__main__":
    main()
