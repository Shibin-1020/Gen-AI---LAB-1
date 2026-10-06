"""Build side-by-side panels (input photo | Denisha's generated Monet) for the 30 samples in
task3_gan/member_denisha/outputs/human_audit.csv, so the second rater can score them.

    git fetch origin denisha-lab1
    python report/make_denisha_audit_panels.py

Needs the course photos in task3_gan/data/photo_jpg (git-ignored). Denisha's exporter translated the photos
in sorted order and named the outputs 000000.jpg, 000001.jpg, ..., so output i belongs to the i-th sorted
photo. Panels are written to denisha_audit_panels/ (git-ignored). Read-only with respect to her branch.
"""
import io
import subprocess
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "denisha_audit_panels"
EXT = {".jpg", ".jpeg", ".png"}


def main() -> None:
    data = subprocess.run(["git", "show", "origin/denisha-lab1:task3_gan/member_denisha/outputs/images.zip"],
                          cwd=ROOT, capture_output=True, check=True).stdout
    photos = sorted(p for p in (ROOT / "task3_gan/data/photo_jpg").glob("**/*")
                    if p.suffix.lower() in EXT and not p.name.startswith("._"))
    if not photos:
        raise SystemExit("No photos found in task3_gan/data/photo_jpg")
    OUT.mkdir(exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for i in range(30):
            name = f"{i:06d}.jpg"
            gen = Image.open(io.BytesIO(z.read(name))).convert("RGB").resize((256, 256))
            src = Image.open(photos[i]).convert("RGB").resize((256, 256))
            panel = Image.new("RGB", (532, 280), "white")
            panel.paste(src, (6, 18))
            panel.paste(gen, (270, 18))
            d = ImageDraw.Draw(panel)
            d.text((6, 3), f"{name}  input: {photos[i].name}", fill="black")
            d.text((270, 3), "generated Monet", fill="black")
            panel.save(OUT / f"{name[:-4]}.png")
    print(f"30 panels written to {OUT.relative_to(ROOT).as_posix()}/ (left = input photo, right = output)")


if __name__ == "__main__":
    main()
