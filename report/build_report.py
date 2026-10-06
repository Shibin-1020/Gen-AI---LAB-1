"""Build the combined team report: report/DATA266_Lab1_Report_Team_<NN>.pdf

    python report/build_report.py

Numbers for Shibin Thomas's models are read from the committed result files at build time
(metrics_report.csv, comparison.csv, full_metrics_report.csv, epochs.csv, manifests), so the report
always matches the evidence in the repository. Teammates' rows come from report/team_info.yaml.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from pathlib import Path

import yaml
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
INFO = yaml.safe_load((REPORT / "team_info.yaml").read_text(encoding="utf-8"))
BLOB = f"{INFO['repo_url']}/blob/{INFO['repo_branch']}/"
TREE = f"{INFO['repo_url']}/tree/{INFO['repo_branch']}/"
BRANCH_URL = f"{INFO['repo_url']}/tree/{INFO['repo_branch']}"

TM = next((m for m in INFO["members"] if m.get("branch")), None)
TM_BRANCH = TM["branch"] if TM else None
TM_BLOB = f"{INFO['repo_url']}/blob/{TM_BRANCH}/" if TM else ""
TM_TREE = f"{INFO['repo_url']}/tree/{TM_BRANCH}/" if TM else ""
D1, D2, D3 = "task1_llm/member_denisha", "task2_sentiment/member_denisha", "task3_gan/member_denisha"


def tm_bytes(path: str) -> bytes:
    """Read a teammate file from their branch (read-only; nothing is copied into this branch)."""
    import subprocess
    return subprocess.run(["git", "show", f"origin/{TM_BRANCH}:{path}"], cwd=ROOT, capture_output=True,
                          check=True).stdout


T1 = "task1_llm/shibin_thomas"
T2 = "task2_sentiment/shibin_thomas"
T3 = "task3_gan/shibin_thomas"
RUN1, RUN2, RUN3 = "gpt_char_v1_20261001-153007", "yelp3_20261001-182849", "cyclegan_v1_20261001-201801"

# ----------------------------------------------------------------------------------------- fonts / styles
FONTS = REPORT / "fonts"
for name, file in (("Serif", "LiberationSerif-Regular.ttf"), ("Serif-B", "LiberationSerif-Bold.ttf"),
                   ("Serif-I", "LiberationSerif-Italic.ttf"), ("Serif-BI", "LiberationSerif-BoldItalic.ttf"),
                   ("Mono", "DejaVuSansMono.ttf")):
    pdfmetrics.registerFont(TTFont(name, str(FONTS / file)))
pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-B", italic="Serif-I", boldItalic="Serif-BI")

BLACK = colors.black
BODY = ParagraphStyle("body", fontName="Serif", fontSize=10.5, leading=13.6, textColor=BLACK,
                      alignment=TA_JUSTIFY, spaceAfter=6)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=8.6, leading=10.6, alignment=0, spaceAfter=4)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8.4, leading=10.0, alignment=0, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="Serif-B")
CAP = ParagraphStyle("cap", parent=BODY, fontSize=9, leading=11, alignment=TA_CENTER, spaceBefore=3, spaceAfter=10)
H1 = ParagraphStyle("h1", parent=BODY, fontName="Serif-B", fontSize=14, leading=17, alignment=0,
                    spaceBefore=6, spaceAfter=8, keepWithNext=1)
H2 = ParagraphStyle("h2", parent=BODY, fontName="Serif-B", fontSize=11.5, leading=14, alignment=0,
                    spaceBefore=10, spaceAfter=5, keepWithNext=1)
H3 = ParagraphStyle("h3", parent=BODY, fontName="Serif-BI", fontSize=10.5, leading=13, alignment=0,
                    spaceBefore=6, spaceAfter=3, keepWithNext=1)
QUOTE = ParagraphStyle("quote", fontName="Mono", fontSize=7.6, leading=9.4, leftIndent=18, rightIndent=10,
                       textColor=BLACK, spaceBefore=2, spaceAfter=6)
REF = ParagraphStyle("ref", parent=BODY, fontSize=9.6, leading=12, alignment=0, leftIndent=20,
                     firstLineIndent=-20, spaceAfter=3)
PEND = "Pending"
WIDTH = letter[0] - 2.0 * inch

_fig = [0]
TAB = {}
_tab = [0]


def esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def md(s: str) -> str:
    """`code` -> monospace, *text* -> italic."""
    s = esc(s)
    s = re.sub(r"`([^`]+)`", r'<font name="Mono" size="8">\1</font>', s)
    return re.sub(r"(?<![\w*])\*([^*]+)\*(?![\w*])", r"<i>\1</i>", s)


def P(s, style=BODY):
    return Paragraph(md(s), style)


def path_ref(path: str, tree=False, teammate=False) -> str:
    url = ((TM_TREE if tree else TM_BLOB) if teammate else (TREE if tree else BLOB)) + path
    return f'<link href="{url}"><font name="Mono" size="7.4">{esc(path)}</font></link>'


def table(rows, widths, header=True, first_col_bold=False, font=8.4):
    data = []
    for r_i, r in enumerate(rows):
        row = []
        for c_i, c in enumerate(r):
            if isinstance(c, (Paragraph, Image, Table)):
                row.append(c)
                continue
            st = CELLB if (header and r_i == 0) or (first_col_bold and c_i == 0) else CELL
            if font != 8.4:
                st = ParagraphStyle("x", parent=st, fontSize=font, leading=font * 1.2)
            row.append(Paragraph(c if "</" in str(c) else md(str(c)), st))
        data.append(row)
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 3),
             ("RIGHTPADDING", (0, 0), (-1, -1), 3), ("TOPPADDING", (0, 0), (-1, -1), 1.8),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 1.8),
             ("LINEABOVE", (0, 0), (-1, 0), 1.0, BLACK), ("LINEBELOW", (0, -1), (-1, -1), 1.0, BLACK),
             ("LINEBELOW", (0, 1 if header else 0), (-1, -2), 0.25, colors.Color(0.7, 0.7, 0.7))]
    if header:
        style.append(("LINEBELOW", (0, 0), (-1, 0), 0.6, BLACK))
    t.setStyle(TableStyle(style))
    return t


def tcap(text: str):
    _tab[0] += 1
    return Paragraph(f"<b>Table {_tab[0]}.</b> {md(text)}", ParagraphStyle("tc", parent=CAP, spaceBefore=8,
                                                                         spaceAfter=4, keepWithNext=1))


def source(*paths, tree=False, teammate=()):
    refs = [path_ref(p, tree) for p in paths] + [path_ref(p, teammate=True) + f" (branch {TM_BRANCH})" for p in teammate]
    return Paragraph("Source: " + ", ".join(refs), SMALL)


def figure(path: str, caption: str, width=WIDTH, max_h=4.0 * inch, teammate=False, image=None, ref=True):
    if image is not None:
        im = image.convert("RGB")
    else:
        src = io.BytesIO(tm_bytes(path)) if teammate else ROOT / path
        im = PILImage.open(src).convert("RGB")
    w, h = im.size
    scale = min(width / w, max_h / h)
    target_px = int(width / inch * 200)
    if w > target_px:
        im = im.resize((target_px, int(h * target_px / w)), PILImage.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88, optimize=True)
    buf.seek(0)
    _fig[0] += 1
    return KeepTogether([Image(buf, width=w * scale, height=h * scale),
                         Paragraph(f"<b>Figure {_fig[0]}.</b> {md(caption)}" +
                                   (f" ({path_ref(path, teammate=teammate)})" if ref else ""), CAP)])


def evidence(items, tm_items=()):
    rows = [["Evidence", "Location in the repository"]]
    rows += [[Paragraph(md("Shibin: " + d), CELL), Paragraph(path_ref(p, tree=t), CELL)] for d, p, t in items]
    rows += [[Paragraph(md("Denisha: " + d), CELL),
              Paragraph(path_ref(p, tree=t, teammate=True) + f" (branch {TM_BRANCH})", CELL)] for d, p, t in tm_items]
    return table(rows, [2.0 * inch, WIDTH - 2.0 * inch])


# ----------------------------------------------------------------------------------------- data loaders
def read_csv(path):
    with open(ROOT / path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def jload(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(ROOT / path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fnum(v, nd=4):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return esc(v) if v not in (None, "") else PEND
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if x == int(x) and "." not in str(v):
        return str(int(x))
    return f"{x:.{nd}f}"


def teammate_models(task: str):
    return INFO.get("teammate", {}).get(task) or []


def tm_val(m, key):
    v = (m.get("metrics") or {}).get(key)
    return fnum(v) if v is not None else PEND


t1 = {r["metric"] + ("" if r["metric"] != "Total training time" else f" ({r['notes']})"): r
      for r in read_csv(f"{T1}/metrics_report.csv")}
t1_epochs = read_csv(f"reproducibility/raw_logs/task1_llm/shibin_thomas/{RUN1}/epochs.csv")
t1_cfg = yaml.safe_load((ROOT / f"{T1}/configs/gpt_char_v1.yaml").read_text(encoding="utf-8"))
t1_man = jload(f"reproducibility/manifests/task1_llm/shibin_thomas/{RUN1}.json")
t2 = {r["metric"]: r for r in read_csv(f"{T2}/outputs/{RUN2}/comparison.csv")}
t2_man = jload(f"reproducibility/manifests/task2_sentiment/shibin_thomas/{RUN2}.json")
T2_MODELS = ["baseline_meanpool", "exp1_textcnn", "exp2_bigru_attn"]
t3 = {(r["section"], r["metric"]): r["value"] for r in read_csv(f"{T3}/full_metrics_report.csv")}
t3_epochs = read_csv(f"reproducibility/raw_logs/task3_gan/shibin_thomas/{RUN3}/epochs.csv")
kaggle = jload(f"{T3}/kaggle_leaderboard.json")
B2A, A2B = "B2A (photo->monet)", "A2B (monet->photo)"
fid_avg = float(t3[("Kaggle submission (avg of both directions)", "FID")])
mifid_avg = float(t3[("Kaggle submission (avg of both directions)", "MiFID")])
kaggle_score = (fid_avg + mifid_avg) / 2


def load_rater(path):
    rows = read_csv(path)
    ok = [r for r in rows if all(str(r.get(c, "")).strip() for c in ("style", "content", "artifacts"))]
    return ok if len(ok) == len(rows) and rows else None


AUD = f"{T3}/outputs/human_audit"
aud_key = {r["sample_id"]: r["direction"] for r in read_csv(f"{AUD}/audit_key.csv")}
aud_r1, aud_r2 = load_rater(f"{AUD}/rater1.csv"), load_rater(f"{AUD}/rater2.csv")
den_aud = jload("report/human_audit/denisha_model_human_audit_agreement.json") \
    if (ROOT / "report/human_audit/denisha_model_human_audit_agreement.json").exists() else None


def aud_means(rows, direction=None):
    sub = [r for r in rows if direction is None or aud_key[r["sample_id"]] == direction]
    return [sum(int(r[c]) for r in sub) / len(sub) for c in ("style", "content", "artifacts")]


def shibin_audit_cell():
    if not aud_r1:
        return PEND
    m = aud_means(aud_r1)
    txt = f"rater 1: style {m[0]:.2f}, content {m[1]:.2f}, artifacts {m[2]:.2f}"
    return txt + ("" if aud_r2 else "; rater 2 pending")


# ======================================================================================================
def build() -> Path:
    out = REPORT / f"DATA266_Lab1_Report_Team_{INFO['team_number']}.pdf"
    members = INFO["members"]
    story = []

    # ------------------------------------------------------------------------------------- title page
    def c(size):
        return ParagraphStyle("c", parent=BODY, fontSize=size, leading=size * 1.3, alignment=TA_CENTER)

    story += [Spacer(1, 1.6 * inch),
              Paragraph("DATA 266: Generative AI, Fall 2026", c(12)),
              Spacer(1, 18),
              Paragraph("<b>Lab 1 Report</b>", c(22)),
              Spacer(1, 8),
              Paragraph("Character-Level Language Model Pretraining, Sentiment Classification,<br/>"
                        "and CycleGAN Image Style Transfer", c(13)),
              Spacer(1, 36),
              Paragraph(f"Team {INFO['team_number']} ({esc(INFO['team_name'])})", c(12)),
              Spacer(1, 6),
              Paragraph("<br/>".join(esc(m["name"]) for m in members), c(12)),
              Spacer(1, 36),
              Paragraph(f'Repository: <link href="{INFO["repo_url"]}">{esc(INFO["repo_url"])}</link>', c(10.5)),
              Paragraph(f'Branch: <link href="{BRANCH_URL}">{esc(INFO["repo_branch"])}</link>', c(10.5)),
              Spacer(1, 24),
              Paragraph(esc(INFO.get("report_date", "October 2026")), c(11)),
              PageBreak()]

    # ------------------------------------------------------------------------------------- 1 ownership
    story.append(Paragraph("1 Team Ownership Statement", H1))
    story.append(P(
        " ".join(f"{m['name']} built {m['built'].rstrip('.')}." for m in members) +
        " Each member trained their own models from scratch in a personal folder for every task "
        "(`task1_llm/<member>`, `task2_sentiment/<member>`, `task3_gan/<member>`). To keep the results "
        "comparable we shared the raw TinyStories file and the generation prompts for Task 1, the official Yelp "
        "Polarity test split and a common metric module for Task 2, and the instructor's evaluation script for "
        "Task 3. The comparison tables and the analysis sections of this report were written jointly."))
    story.append(Paragraph("1.1 Repository Organisation and Reproducibility", H2))
    story.append(P(
        "Every member folder follows the same layout: source code and the task notebook in `src/`, configuration "
        "files in `configs/`, model weights in `checkpoints/<run_id>/`, plots and samples in `outputs/<run_id>/`, "
        "and the files `metrics_report.csv`, `results.md` and `failure_analysis.md`. Unedited training logs are kept "
        "in `reproducibility/raw_logs/` and a manifest for each run (configuration, git commit, library versions, "
        "hardware and checkpoint hashes) in `reproducibility/manifests/`. Each figure and table below names the file "
        "it was taken from, so every number can be traced back to its log or checkpoint."))

    story.append(Paragraph("1.2 Contribution Overview", H2))
    audit_sh = ("Built the blinded audit pack (30 panels, hidden direction key) and the rating form; rated his own "
                "model as rater 1 and Denisha's 30 images as rater 2.")
    audit_dn = ("Rated her own 30 photo-to-Monet images as rater 1" +
                ("; rated Shibin's 30 panels as rater 2." if aud_r2 else "; her rating of Shibin's panels (rater 2) is pending."))
    contrib = [
        ["Area", "Shibin Thomas", "Denisha Ketan Tank"],
        ["Task 1: character-level GPT",
         "Cleaned TinyStories and drew his own 100K/10K split; hand-wrote attention, LayerNorm and the Transformer "
         "blocks with unit tests; trained gpt_char_v1 (6 layers, 7.5M parameters, context 256) for 10 epochs on an "
         "RTX 5090; computed every required metric, generated 40 samples and wrote the failure analysis.",
         "Drew her own 100K/10K split; implemented and trained a 4-layer character GPT (3.25M parameters, context "
         "128) for 10 epochs on a Tesla T4 with checkpointing and a smoke test; produced metrics, samples, loss "
         "curves and the failure analysis."],
        ["Task 2: Yelp sentiment",
         "Built the preprocessing pipeline (deduplication, train-test overlap removal, negation-preserving "
         "stopwords, stemming); trained the mean-pooling baseline, TextCNN and BiGRU with attention; evaluated on "
         "the official test split with bootstrap intervals, slices, calibration and McNemar tests; reviewed 20 "
         "errors by hand.",
         "Built her own preprocessing and train/validation/test split; trained a mean-pooling baseline, TextCNN and "
         "bidirectional GRU; produced the full metric set with confidence intervals, slices and McNemar tests; "
         "reviewed 20 errors by hand."],
        ["Task 3: CycleGAN",
         "Implemented and trained CycleGAN v1 from scratch (ResNet-9 with resize-convolution, 70x70 PatchGAN); "
         "re-implemented and unit-tested the instructor's FID/MiFID protocol; added KID, precision/recall, density/"
         "coverage and LPIPS; wrote the visual failure analysis; prepared the improved v2 recipe.",
         "Implemented and trained an independent CycleGAN (60 epochs, identity weight 2.5, fp16) on an RTX 4090; "
         "ran the official evaluation notebook (FID 100.79); exported translations, reconstructions and the "
         "Kaggle image archive."],
        ["Kaggle", f"Submitted his model's file (score -{kaggle_score:.2f}).",
         f"Submitted the team's best entry (score {kaggle.get('public_score')}, rank {kaggle.get('public_rank')})."],
        ["Human audit", audit_sh, audit_dn],
        ["Shared infrastructure",
         "Set up the repository layout, the shared TinyStories download, the shared metric modules for Tasks 1 "
         "and 2, run manifests, smoke tests and the report build script.",
         "Maintained her member_denisha folders with configurations, notebooks, checkpoints, raw logs, manifests, "
         "results and failure analyses on branch " + str(TM_BRANCH) + "."],
        ["Report", "Assembled the combined report and wrote his sections; joint analysis written together.",
         "Contributed her results and analysis sections and reviewed the report; joint analysis written together."],
    ]
    story.append(tcap("Who built what, by area."))
    story.append(table(contrib, [1.25 * inch, (WIDTH - 1.25 * inch) / 2, (WIDTH - 1.25 * inch) / 2], first_col_bold=True))

    # ===================================================================================== 2 TASK 1
    story += [PageBreak(), Paragraph("2 Task 1: GPT-Style Character-Level Language Model", H1)]
    m = t1_cfg["model"]
    tr = t1_cfg["training"]
    story.append(P(
        f"The model is a decoder-only Transformer trained on characters from TinyStories [2]. All components "
        f"(token and positional embeddings, layer normalisation, multi-head causal self-attention, the feed-forward "
        f"blocks and the residual connections) were written by hand following Vaswani et al. [1], with the "
        f"pre-LayerNorm arrangement of Xiong et al. [5] and the GPT-2 initialisation scheme [4]; no built-in "
        f"Transformer or attention module of PyTorch is used, and a unit test (`tests/test_model.py`) checks this. "
        f"Each member drew their own split from the cleaned story pool. Shibin's split contains 100,000 training "
        f"and 10,000 validation stories (seed 266), and the character vocabulary of 90 symbols was built from the "
        f"training split only. The network has {m['n_layer']} layers, a model width of {m['d_model']}, "
        f"{m['n_head']} attention heads and a context of {m['block_size']} characters, about 7.5 million parameters "
        f"in total. Denisha built a smaller and shallower model (4 layers, width 256, 4 heads, context 128, "
        f"3.25 million parameters) on her own 100,000/10,000 split (seed 20260929) and trained it on a Tesla T4 "
        f"with fp16 mixed precision, so the two models differ in depth, width, context length and optimiser "
        f"settings."))

    story.append(Paragraph("2.1 Model Comparison", H2))
    tms = teammate_models("task1")
    arch = (f"{m['n_layer']} layers, d_model {m['d_model']}, {m['n_head']} heads (head dim "
            f"{m['d_model'] // m['n_head']}), FFN 4x with GELU, pre-LayerNorm, learned positions, tied "
            f"input/output embeddings, context {m['block_size']}")
    hp = (f"AdamW (0.9, 0.95), weight decay {tr['weight_decay']}, peak LR {tr['lr']:g} with {tr['warmup_frac']:.0%} "
          f"warm-up and cosine decay to {tr['min_lr']:g}, batch {tr['batch_size']} x {m['block_size']}, gradient clip "
          f"{tr['grad_clip']}, dropout {m['dropout']}, {tr['epochs']} epochs, bf16")
    rows = [["", "Shibin Thomas (gpt_char_v1)"] + [esc(x["model"]) for x in tms],
            ["Architecture", arch] + [esc(x.get("arch") or PEND) for x in tms],
            ["Hyperparameters", hp] + [esc(x.get("hparams") or PEND) for x in tms]]

    def v(k, nd=4):
        return fnum(t1[k]["value"], nd)

    t1_rows = [
        ("Training cross-entropy (nats/char)", v("Training cross-entropy loss")),
        ("Validation cross-entropy (nats/char)", v("Validation cross-entropy loss")),
        ("Validation perplexity", v("Perplexity (validation)")),
        ("Validation bits per character", v("Bits-per-character (validation)")),
        ("Training perplexity / BPC", f"{v('Perplexity (train)')} / {v('Bits-per-character (train)')}"),
        ("Generalisation gap (val - train CE)", v("Generalization gap")),
        ("Next-character accuracy (validation)", v("Top-1 next-character accuracy (validation)")),
        ("Distinct-1 / 2 / 3 (sampled, T = 0.8)", f"{v('Distinct-1')} / {v('Distinct-2')} / {v('Distinct-3')}"),
        ("Repeated 4-gram rate (sampled / greedy)", f"{v('Repeated 4-gram rate')} / {v('Repeated 4-gram rate (greedy)')}"),
        ("Gradient norm mean / p99 / max", f"{v('Gradient norm (mean, pre-clip)', 3)} / "
                                           f"{v('Gradient norm (p99, pre-clip)', 3)} / {v('Gradient norm (max, pre-clip)', 2)}"),
        ("Clipped steps / loss spikes / NaN steps", f"{float(t1['Fraction of steps clipped']['value']):.2%} / "
                                                    f"{v('Loss spikes')} / {v('NaN / Inf steps')}"),
        ("Parameters (total / non-embedding)", f"{v('Parameter count')} / {v('Parameter count (non-embedding)')}"),
        ("Training throughput (characters/s)", v("Training tokens/sec")),
        ("Generation speed (characters/s)", f"{float(t1['Generation tokens/sec']['value']):.1f}"),
        ("Peak GPU memory (MB)", f"{float(t1['Peak GPU memory (allocated)']['value']):,.0f}"),
        ("Training time (min) / epochs / best epoch", f"{float(t1['Total training time (minutes)']['value']):.1f} / "
                                                      f"{v('Epochs trained')} / {v('Best epoch (lowest val loss)')}"),
        ("Hardware", v("Hardware")),
        ("Run and checkpoint", f"`{RUN1}`, `best_model.pt` (SHA-256 "
                               f"{t1_man['checkpoints_files']['best_model.pt']['sha256'][:12]})"),
    ]
    for label, val in t1_rows:
        rows.append([label, val] + [tm_val(x, label) for x in tms])
    ncol = 1 + len(tms)
    story.append(tcap("Task 1 comparison of architecture, hyperparameters and metrics. Losses, perplexity, BPC and "
                      "accuracy are computed on the best checkpoint; metric definitions follow the shared module "
                      "`task1_llm/shared_eval/metrics.py`."))
    TAB["t1"] = _tab[0]
    story.append(table(rows, [2.0 * inch] + [(WIDTH - 2.0 * inch) / ncol] * ncol, first_col_bold=True))
    story.append(source(f"{T1}/metrics_report.csv", teammate=[f"{D1}/outputs/metrics.csv", f"{D1}/config.yaml"]))

    story.append(Paragraph("2.2 Training Behaviour", H2))
    story.append(figure(f"{T1}/outputs/{RUN1}/loss_curves.png",
                        "Task 1 training and validation loss (Shibin Thomas)", max_h=2.9 * inch))
    story.append(figure(f"{D1}/outputs/loss_curves.png", "Task 1 training and validation loss per epoch "
                        "(Denisha Ketan Tank)", max_h=2.6 * inch, teammate=True))
    ep_rows = [["Epoch"] + [r["epoch"] for r in t1_epochs],
               ["Val. CE"] + [f"{float(r['val_loss']):.3f}" for r in t1_epochs],
               ["Val. acc."] + [f"{float(r['val_top1_acc']) * 100:.1f}" for r in t1_epochs]]
    n = len(t1_epochs)
    story.append(tcap("Validation cross-entropy (nats/char) and next-character accuracy (%) after each epoch."))
    story.append(table(ep_rows, [0.8 * inch] + [(WIDTH - 0.8 * inch) / n] * n, first_col_bold=True, font=8))
    story.append(source(f"reproducibility/raw_logs/task1_llm/shibin_thomas/{RUN1}/epochs.csv"))
    story.append(P(
        "The training loss starts at 4.52, which is the loss of a uniform guess over 90 characters, and drops below "
        "1.0 within the first 1,400 steps. After that the validation loss keeps falling every epoch, from 0.715 to "
        "0.571, so the last epoch is also the best one. The gap between validation and training loss stays very "
        "small (0.012 nats at the end), which is expected with roughly 120 training characters per parameter. "
        "Training was stable: there were no NaN steps or loss spikes, and all of the 738 clipped steps happened "
        "during the learning-rate warm-up. Denisha's model followed the same pattern with a slightly negative "
        "generalisation gap (validation loss 0.682 against a training loss of 0.713 measured with dropout active), "
        "no NaN gradients and a final gradient norm of 0.23."))

    story.append(Paragraph("2.3 Discussion", H2))
    story.append(P(
        "*Strengths.* Both small models, trained from scratch on a single GPU, write fluent TinyStories-style text "
        "with named characters, dialogue and simple cause and effect. Shibin's 7.5M-parameter model reaches a "
        "validation BPC of 0.824 and 81.7% next-character accuracy; Denisha's 3.25M-parameter model reaches 0.984 "
        "and 78.2%. The comparison shows the expected effect of capacity and context: the deeper, wider model "
        "with twice the context predicts better and also produces more varied text (Distinct-3 of 0.93 against "
        "0.74 and a repeated 4-gram rate of 0.007 against 0.153), while the smaller model needs less than a "
        "tenth of the GPU memory (408 MB against 3,347 MB)."))
    story.append(P(
        "*Weaknesses.* Both models show the same three failure types: repetition, broken or invented words, and "
        "loss of coherence. Greedy decoding falls into loops (repeated 4-gram rate 0.137 for Shibin's model). "
        "Because words are produced one character at a time, rare words are misspelled and sometimes spelled "
        "differently each time they appear. Stories lose their thread once they grow beyond the context window, "
        "which happens sooner for Denisha's 128-character context, where a new story often starts in the middle "
        "of a continuation."))
    story.append(P(
        "*Limitations.* Each model was trained once with a single seed, on different hardware (RTX 5090 and "
        "Tesla T4) and on different random splits, so small differences should not be over-interpreted. The "
        "diversity metrics were computed on different prompts and temperatures (10 shared prompts at T = 0.8 "
        "for Shibin, one prompt at T = 0.7 to 1.1 for Denisha). Neither validation loss had fully flattened by "
        "epoch 10, Distinct-n measures diversity rather than story quality, and generation speed was measured "
        "without a key-value cache."))
    story.append(P(
        "*Next steps.* We would train for more epochs or use a slightly larger model, extend the context to 512 "
        "characters, and compare top-k or nucleus sampling with a repetition penalty on the shared prompts. A "
        "key-value cache would speed up generation, and a sub-word tokenizer trained from scratch would remove "
        "most of the spelling errors."))

    story.append(Paragraph("2.4 Failure Analysis (Shibin Thomas)", H2))
    story.append(P(
        "We generated text for the 10 shared prompts, with one greedy and three sampled continuations (T = 0.8) of "
        "500 characters each. The three clearest failures are shown below; the full set of 40 samples is in "
        "`outputs/<run_id>/samples.md`."))
    fails = [
        ("Repetition (greedy decoding)",
         "In a small house near the forest, there was a big tree. The tree was very high\n"
         "and the tree was very high. The tree was very high up in the tree. The tree was\n"
         "very high and the tree was very high.\n\n"
         "One day, the tree saw a big tree. The tree was very high and the tree was very\n"
         "high. The tree was very high and the tree was very high. ...",
         "The phrase “the tree was very” appears 16 times and the repeated 4-gram rate of this sample is "
         "0.61, against 0.04 on average. Greedy decoding always takes the most likely character, so once a "
         "high-probability phrase has been written, starting it again is again the most likely continuation. "
         "Sampling, or a repetition penalty, would break the loop; the repeated 4-gram rate on the same prompts is "
         "the measure to check."),
        ("Broken and inconsistent spelling (T = 0.8)",
         "Tim was scared. He ran and ran until he came across a big lute. He saw a small\n"
         "tree which was very pretty. Tim wanted to take the lutter from him. He said yes,\n"
         "but he had to be careful.\n\n"
         "The luter was big and boring. Tim saw that the lutte was scary and hurt. ...",
         "The rare word “lute” is spelled four different ways (lute, lutter, luter, lutte). The model has "
         "no word list and rebuilds a rare word from frequent letter patterns each time it comes back. A lower "
         "temperature, longer training or sub-word tokens should reduce this."),
        ("Loss of coherence (T = 0.8)",
         "After school, Sam wanted to go to the zoo and see all the animals.\n\n"
         "They all walked closer and held on tight. Sam thought it was tremommeter than\n"
         "the diamond. ... Together, Jack and Sam started to fill their bucket with food.\n"
         "He was so proud of their work! ... Tom and Sam continued to throw their bucket\n"
         "until it was time to go hom",
         "Each sentence is fluent, but the setting jumps from the zoo to a diamond and a bucket, two new friends "
         "(Jack and Tom) appear without introduction, and the pronouns stop matching. The continuation is 500 "
         "characters long while the model only sees the last 256, so the beginning of the story has literally "
         "dropped out of its context. A longer context or a deeper model would be the fix to test."),
    ]
    for i, (title, text, obs) in enumerate(fails, 1):
        story += [Paragraph(f"Case {i}: {title}", H3), Preformatted(text, QUOTE), P(obs)]
    story.append(Paragraph("2.5 Failure Analysis (Denisha Ketan Tank)", H2))
    story.append(P(
        "Denisha generated continuations of the prompt \u201cHenry the mule was so pleased\u201d at temperatures "
        "0.7, 0.9 and 1.1 and analysed one failure in each."))
    dfails = [
        ("Repetition (T = 0.7)",
         'Henry the mule was so pleased with his paint and he said, "You are very sweet,\n'
         'Lily. You are very brave and smart. You are very smart and kind."\n\n'
         "Lily and Tom nodded. They got in the car and drove to their car. ...",
         "The phrase \u201cYou are very\u201d repeats three times in two sentences, and the repetition rate "
         "Denisha measured for this continuation is 0.237. \u201cdrove to their car\u201d after \u201cgot in the car\u201d is a "
         "local loop. Denisha's proposed test is to compare temperatures 0.7, 0.9 and 1.1 and add a repetition "
        "penalty, measuring the repeated 4-gram rate."),
        ("Abrupt topic change (T = 0.9)",
         "Henry the mule was so pleased that he spilled his hand.\n\nThe End.\n"
         "Once there was a girl. She always wanted to explore. One day she was walking in\n"
         "the park and she saw the sand spraying snowball covered popcorn. ...",
         "After one nonsensical sentence the model writes \u201cThe End.\u201d and starts an unrelated story. "
         "With a 128-character context the original character is forgotten almost immediately, and the "
         "noun phrase \u201csand spraying snowball covered popcorn\u201d is grammatical but meaningless."),
        ("Broken grammar and coherence (T = 1.1)",
         "Henry the mule was so pleased that he and helped her mommy cook a nice magic fort!\n"
         "After that, his mummy reminded him to listen carefully and jog with his owner.\n"
         "... She loved good dream about the airport. One day, she decided to use a whip to belong",
         "At high temperature the grammar breaks (\u201che and helped\u201d, \u201cloved good dream\u201d, "
         "\u201cto use a whip to belong\u201d) and the pronouns switch between him and her. Lower temperatures "
         "and longer training are the proposed fixes."),
    ]
    for i, (title, text, obs) in enumerate(dfails, 1):
        story += [Paragraph(f"Case {i}: {title}", H3), Preformatted(text, QUOTE), P(obs)]
    story.append(source(teammate=[f"{D1}/failure_analysis.md", f"{D1}/outputs/samples.json"]))

    story.append(Paragraph("2.6 Evidence", H2))
    story.append(evidence([
        ("Configuration", f"{T1}/configs/gpt_char_v1.yaml", False),
        ("Training log", f"reproducibility/raw_logs/task1_llm/shibin_thomas/{RUN1}/train.log", False),
        ("Per-step and per-epoch logs", f"reproducibility/raw_logs/task1_llm/shibin_thomas/{RUN1}", True),
        ("Run manifest", f"reproducibility/manifests/task1_llm/shibin_thomas/{RUN1}.json", False),
        ("Plots (loss, gradient norm, LR)", f"{T1}/outputs/{RUN1}", True),
        ("Generated samples", f"{T1}/outputs/{RUN1}/samples.md", False),
        ("Best checkpoint", f"{T1}/checkpoints/{RUN1}/best_model.pt", False),
        ("Results and design choices", f"{T1}/results.md", False),
        ("Executed notebook", f"{T1}/src/task1_gpt_tinystories.ipynb", False),
    ], [("Configuration", f"{D1}/config.yaml", False), ("Training log", "reproducibility/raw_logs/task1_denisha.log", False),
        ("Run manifest", "reproducibility/manifests/task1_denisha.json", False),
        ("Metrics and history", f"{D1}/outputs/metrics.csv", False), ("Loss curves", f"{D1}/outputs/loss_curves.png", False),
        ("Generated samples", f"{D1}/outputs/samples.json", False),
        ("Checkpoint", f"{D1}/checkpoints/gpt_from_scratch.pt", False), ("Results", f"{D1}/results.md", False),
        ("Notebook", f"{D1}/Task1_GPT_Colab.ipynb", False)]))

    # ===================================================================================== 3 TASK 2
    story += [PageBreak(), Paragraph("3 Task 2: Sentiment Classification on Yelp Polarity", H1)]
    story.append(P(
        "All Task 2 models classify a review from the Yelp Review Polarity dataset [6] as positive or negative and "
        "learn their word embeddings from random "
        "initialisation; no pretrained embeddings or language models are used. The official test split (38,000 "
        "reviews) is shared by the team. Each member carved a validation set out of the training split; Shibin "
        "used 50,000 reviews (seed 266) for early stopping and model selection. Preprocessing removes duplicates "
        "and reviews that also appear in the test set, lowercases the text, expands contractions, removes "
        "punctuation and stopwords while keeping negations and contrast words, and applies Snowball stemming. "
        "The vocabulary of 30,000 tokens is built from the training split only."))
    story.append(P(
        "Shibin trained three models that share the same 128-dimensional embedding, so that differences in the "
        "results come from the architecture. The baseline averages the word embeddings of a review and feeds them "
        "to a small MLP, in the spirit of fastText [8]. The first experimental model is a TextCNN [7] with filter widths 3, 4 and 5 and max "
        "pooling over time. The second is a two-layer bidirectional GRU with additive attention pooling [9]."))
    story.append(P(
        "Denisha also trained a mean-pooling baseline, a TextCNN and a two-layer bidirectional GRU with "
        "learned 128-dimensional embeddings, but with different hyperparameters: batch size 512, a "
        "reduce-on-plateau learning-rate schedule, up to 12 epochs, bfloat16 mixed precision, an MLP hidden size "
        "of 128 for the baseline, dropout 0.3 for the CNN, and a GRU classifier that reads the output at the last "
        "token instead of using attention. Her TextCNN has the same layer sizes as Shibin's, so the two differ only "
        "in their training hyperparameters. One important difference concerns evaluation: Denisha split the "
        "560,000 official training reviews into her own training, validation and test sets and evaluated on her "
        "held-out test set of 112,000 reviews, whereas Shibin evaluated on the official test split of 38,000 "
        "reviews. Both test sets come from the same distribution and are large, but the numbers are not computed "
        "on identical reviews."))

    story.append(Paragraph("3.1 Model Comparison", H2))
    tms2 = teammate_models("task2")
    archs = {"baseline_meanpool": "Embedding 128, masked mean pooling, MLP (64), sigmoid output",
             "exp1_textcnn": "Embedding 128, Conv1d widths 3/4/5 with 128 filters each, max-over-time pooling",
             "exp2_bigru_attn": "Embedding 128, 2-layer BiGRU (128 per direction), additive attention pooling"}
    hps = {"baseline_meanpool": "AdamW lr 2e-3, batch 512, dropout 0.3, up to 8 epochs",
           "exp1_textcnn": "AdamW lr 1e-3, batch 256, dropout 0.5, up to 5 epochs",
           "exp2_bigru_attn": "AdamW lr 1e-3, batch 256, dropout 0.3, up to 5 epochs"}
    names = {"baseline_meanpool": "Baseline (mean pool)", "exp1_textcnn": "Exp. 1 (TextCNN)",
             "exp2_bigru_attn": "Exp. 2 (BiGRU + attn.)"}
    rows = [[""] + [f"Shibin: {names[mm]}" for mm in T2_MODELS] + [esc(x["model"]) for x in tms2],
            ["Architecture"] + [archs[mm] for mm in T2_MODELS] + [esc(x.get("arch") or PEND) for x in tms2],
            ["Hyperparameters"] + [hps[mm] for mm in T2_MODELS] + [esc(x.get("hparams") or PEND) for x in tms2]]
    t2_rows = ["Accuracy", "Accuracy 95% CI", "Precision (macro)", "Recall (macro)", "F1 (macro)",
               "Precision (micro)", "Recall (micro)", "F1 (micro)", "Precision (weighted)", "Recall (weighted)",
               "F1 (weighted)", "Confusion matrix [TN FP; FN TP]", "ROC-AUC", "PR-AUC", "MCC", "MCC 95% CI",
               "Brier score", "ECE (15 bins)", "McNemar vs baseline (b / c, p)", "Parameters", "Training time (min)",
               "Epochs run (best)", "Train examples/sec", "Inference examples/sec", "Peak GPU memory (MB)", "Hardware"]
    for k in t2_rows:
        rows.append([k] + [esc(t2[k][mm]) for mm in T2_MODELS] + [tm_val(x, k) for x in tms2])
    rows.append(["Checkpoint (SHA-256)"] + [f"{t2_man['checkpoints_files'][mm]['sha256'][:12]}" for mm in T2_MODELS]
                + [tm_val(x, "Checkpoint (SHA-256)") for x in tms2])
    ncol = len(T2_MODELS) + len(tms2)
    story.append(tcap("Task 2 comparison. Shibin's models are evaluated on the official test split (n = 38,000) and "
                      "share: vocabulary 30K, maximum length 256 tokens (head and tail), 3% warm-up with cosine decay, "
                      f"weight decay 1e-4, gradient clipping 1.0, early stopping on validation macro-F1 (run `{RUN2}`). "
                      "Denisha's models are evaluated on her own held-out test split (n = 112,000) and share: vocabulary "
                      "30K, maximum length 256, weight decay 1e-4, early stopping on validation loss. Threshold 0.5 "
                      "for all models."))
    TAB["t2"] = _tab[0]
    story.append(table(rows, [1.3 * inch] + [(WIDTH - 1.3 * inch) / ncol] * ncol, first_col_bold=True,
                       font=7.4 if ncol > 3 else 8.2))
    story.append(source(f"{T2}/outputs/{RUN2}/comparison.csv", "task2_sentiment/shared_eval/metrics.py",
                        teammate=[f"{D2}/metrics_report.csv", f"{D2}/config.yaml"]))
    story.append(P(
        "Macro, micro and weighted scores are identical, or nearly so, because both test sets are balanced "
        "(19,000 reviews per class in the official split and 56,000 per class in Denisha's split): weighted "
        "averaging then uses equal weights, and micro averaging always equals accuracy for a single-label binary "
        "task."))

    sl = [["Slice", "Shibin: Baseline", "Shibin: TextCNN", "Shibin: BiGRU + attn.", "Denisha: Baseline",
           "Denisha: TextCNN", "Denisha: BiGRU"],
          ["Contains a negation", "7.47", "5.50", "4.48", "-", "-", "-"],
          ["Contains a contrast word", "7.89", "5.94", "4.87", "-", "-", "-"],
          ["Neither negation nor contrast", "4.93", "4.53", "4.16", "-", "-", "-"],
          ["Short reviews", "6.96", "5.29", "5.01", "6.96", "5.94", "5.32"],
          ["Medium-length reviews", "6.94", "5.22", "4.09", "6.69", "5.11", "4.67"],
          ["Long reviews", "7.00", "5.52", "4.45", "6.81", "5.14", "4.64"]]
    story.append(tcap("Error rate (%) on subsets of each member's test set. Shibin's length slices are at most 50, "
                      "51 to 150 and more than 150 words (n = 9,287 / 16,910 / 11,803); Denisha's use her own length "
                      "bins (n = 26,869 / 50,261 / 34,870). Denisha did not compute the negation and contrast slices."))
    story.append(table(sl, [1.7 * inch] + [(WIDTH - 1.7 * inch) / 6] * 6, font=8))
    story.append(source(f"{T2}/metrics_report.csv", teammate=[f"{D2}/metrics_report.csv"]))
    story.append(figure(f"{T2}/outputs/{RUN2}/training_curves.png",
                        "Training and validation loss and validation macro-F1 per epoch for the three Task 2 models",
                        max_h=2.7 * inch))

    story.append(Paragraph("3.2 Discussion", H2))
    story.append(P(
        "*Strengths.* Both members found the same ranking, recurrent model > TextCNN > baseline, on every "
        "discrimination metric, which makes the result robust to the different splits, hyperparameters and "
        "hardware. The best models reach 95.58% (Shibin, BiGRU with attention, official test set) and 95.19% "
        "(Denisha, BiGRU, her own test set) accuracy, and the baselines agree closely (93.04% and 93.21%). "
        "For Shibin's models the 95% bootstrap intervals do not overlap and paired McNemar tests on the same "
        "38,000 reviews give p-values below 1e-20 for each pair; Denisha's McNemar tests against her baseline give "
        "p-values below 1e-110. Compared with the baseline, the BiGRU removes 36% of all errors, and 40% and "
        "38% of the errors on reviews containing a negation or a contrast word, which is exactly where a bag of "
        "words cannot work. All three models are well calibrated (expected calibration error [10] at most 1.1%). Since the shared embedding "
        "table holds 3.84 of the roughly 4 million parameters in every model, the gains come from the way the text "
        "is read rather than from model size."))
    story.append(P(
        "*Weaknesses.* The recurrent models are the slowest: Shibin's BiGRU trains about 120 times slower than his "
        "baseline (19 minutes against 21 seconds) and is slightly over-confident, with 544 of its 1,681 errors "
        "made at a confidence above 90%. For both members short reviews are the hardest subset (5.0% and 5.3% "
        "error for the best models), and both error reviews find that mixed sentiment, negation and sarcasm cause "
        "most confident mistakes."))
    story.append(P(
        "*Limitations.* The labels come from star ratings, not from the text, so some are simply wrong: 5 of the 20 "
        "errors we reviewed contradict their own text, and 932 test reviews are misclassified by all three models. "
        "Without pretrained knowledge, idioms and sarcasm are hard to learn. We used a single seed per model and a "
        "fixed decision threshold of 0.5, and because the two members evaluated on different test sets, "
        "differences of a few tenths of a percent between members are not meaningful."))
    story.append(P(
        "*Next steps.* Marking the tokens that follow a contrast word or a negation is a cheap change we would test "
        "first, on the contrast slice with a McNemar test. We would also try a longer maximum length for long "
        "reviews, a small Transformer encoder trained from scratch, an ensemble of the three models with weights "
        "chosen on validation, temperature scaling, and several seeds per model."))

    story.append(Paragraph("3.3 Error Review (Shibin Thomas)", H2))
    errs = [
        (2, "neg", "1.000", "Label noise", "Wow love the place and everything is very clean and new! Great place to come and relax worth a try!"),
        (4, "neg", "1.000", "Mixed sentiment, implicit verdict", "About average so far as steakhouses go. My rib eye was very tasty but a little over cooked. ... Over all I'd say they are more for show than content. If you want to have a great high end steak I'd recommend Ruth Chris..."),
        (6, "pos", "0.000", "Label noise (edited review)", "EDIT: They really did change the service up since I last posted this. Horrible service. Used to be my favorite pizza in the city..."),
        (9, "pos", "0.001", "Sentiment target confusion", "Last night several parents came in with over 15 children ... The bartender expressed that it was not a place to have children running around..."),
        (13, "pos", "0.498", "Mixed sentiment, humour", "Their alcohol selection sucks ... dreamcrushed my cocktail dreams. ... However, the burger was nice, very juicy and rare."),
        (14, "neg", "0.502", "Implicit negativity, domain shift", "Overly sentimental, non stop references to her kids, how blessed she is ... too many covers from elite performers..."),
        (15, "pos", "0.497", "Idiom", "the pizza was to my door in 15 minutes! ... the cookie's are to die for."),
        (16, "neg", "0.999", "Faint praise", "Standard take out and it's cheap. Service is fast and friendly. I go here when I'm too lazy to walk to Zaw's..."),
        (18, "pos", "0.002", "Complex negation", "despite still not digging their ordering process, their food is just too good to disrespect with a 2 star review."),
        (19, "pos", "0.002", "Sarcasm", "For being a DUMP, should expect much more. Flys, stink, garbage, dirt ... Keepin it real dumpy!"),
    ]
    story.append(P(
        "We reviewed 20 test reviews misclassified by the best model (BiGRU with attention): the five most "
        "confident false positives, the five most confident false negatives, five errors closest to the 0.5 "
        "threshold, and five errors from the subset with the highest error rate (short reviews). Each review was "
        f"read and assigned an error type by hand. Ten of them are shown in Table {_tab[0] + 1}."))
    rows = [["#", "Label", "P(pos)", "Error type", "Review text (excerpt)"]]
    for e in errs:
        rows.append([str(e[0]), e[1], e[2], e[3], f'<i>{esc(e[4])}</i>'])
    story.append(tcap("Examples from the manual review of 20 misclassified test reviews."))
    story.append(table(rows, [0.3 * inch, 0.45 * inch, 0.5 * inch, 1.35 * inch, WIDTH - 2.6 * inch], font=8.2))
    story.append(source(f"{T2}/failure_analysis.md"))
    et = [["Error type", "Count"],
          ["Mixed sentiment (praise and complaints in the same review)", "7"],
          ["Label noise (text contradicts the star rating)", "5"],
          ["Faint praise or implicit negativity", "3"],
          ["Complex negation, target confusion, truncation, idiom, sarcasm", "1 each"]]
    story.append(tcap("Error types among the 20 reviewed errors."))
    story.append(table(et, [WIDTH - 1.0 * inch, 1.0 * inch]))
    story.append(P(
        "Five of the 20 errors are not really model errors, because the prediction agrees with the text and the "
        "star rating does not. The most common error the model could learn to avoid is mixed sentiment, where the "
        "verdict sits in the clause after “but” or “however”. As a testable fix we propose to "
        "prefix every token after the last contrast word with a marker, retrain the BiGRU with only this change, "
        "and compare the error rate on the contrast slice (currently 4.87%) with a McNemar test on the same test "
        "set."))
    story.append(Paragraph("3.4 Error Review (Denisha Ketan Tank)", H2))
    story.append(P(
        "Denisha reviewed 20 errors of her best model (BiGRU) chosen with the same scheme: five confident false "
        "positives, five confident false negatives, five near-threshold errors and five errors from one slice. "
        f"Table {_tab[0] + 1} shows ten of them."))
    import json as _json
    recs = _json.loads(tm_bytes(f"{D2}/outputs/error_review.json"))
    pick = {1: "Advice, implicit negation", 2: "No context", 5: "Label noise", 6: "Negation", 8: "Negation and contrast",
            10: "Slang, negative words in a positive review", 16: "Mixed sentiment", 19: "Non-English text",
            11: "Mixed sentiment, late verdict", 13: "Possible label noise"}
    group = {"confident_false_positive": "Confident FP", "confident_false_negative": "Confident FN",
             "near_threshold": "Near threshold", "slice_specific": "Slice (medium)"}
    rows = [["Test idx", "Label", "P(pos)", "Group", "Error type", "Review text (excerpt)"]]
    for i in [1, 2, 5, 6, 8, 10, 16, 19, 11, 13]:
        r = recs[i - 1]
        txt = " ".join(str(r["text"]).replace("\\n", " ").split())
        txt = txt if len(txt) <= 170 else txt[:167].rsplit(" ", 1)[0] + " ..."
        rows.append([r["index"], "pos" if str(r["label"]) == "1" else "neg", f"{float(r['probability']):.3f}",
                     group[r["review_bucket"]], pick[i], f"<i>{esc(txt)}</i>"])
    story.append(tcap("Examples from Denisha's manual review of 20 misclassified test reviews."))
    story.append(table(rows, [0.55 * inch, 0.4 * inch, 0.45 * inch, 0.8 * inch, 1.15 * inch, WIDTH - 3.35 * inch], font=8))
    story.append(source(teammate=[f"{D2}/failure_analysis.md", f"{D2}/outputs/error_review.json"]))
    story.append(P(
        "Denisha's review finds the same broad pattern as Shibin's. Very short reviews (\u201cYup\u201d) give "
        "the model almost nothing to work with, local negation (\u201cNot expensive\u201d) is read word by word, "
        "and long reviews that mix praise and criticism are judged by their most frequent sentiment words rather "
        "than by the final verdict. She also found one non-English review and two probable label errors. Her "
        "proposed fixes are negation-aware features, an attention or sentence-level pooling layer that can weight "
        "the concluding sentences, and an abstention analysis for extremely short reviews."))

    story.append(Paragraph("3.5 Evidence", H2))
    story.append(evidence([
        ("Configurations", f"{T2}/configs", True),
        ("Training and evaluation logs", f"reproducibility/raw_logs/task2_sentiment/shibin_thomas/{RUN2}", True),
        ("Run manifest", f"reproducibility/manifests/task2_sentiment/shibin_thomas/{RUN2}.json", False),
        ("All metrics for every model", f"{T2}/metrics_report.csv", False),
        ("ROC, PR, reliability and slice plots", f"{T2}/outputs/{RUN2}", True),
        ("Learned embedding neighbours", f"{T2}/outputs/{RUN2}/embedding_neighbours.md", False),
        ("Checkpoints", f"{T2}/checkpoints/{RUN2}", True),
        ("Results and analysis", f"{T2}/results.md", False),
        ("Executed notebook", f"{T2}/src/task2_yelp_sentiment.ipynb", False),
    ], [("Configuration", f"{D2}/config.yaml", False), ("Training log", "reproducibility/raw_logs/task2_denisha.log", False),
        ("Run manifest", "reproducibility/manifests/task2_denisha.json", False),
        ("All metrics for every model", f"{D2}/metrics_report.csv", False),
        ("Data analysis", f"{D2}/outputs/data_analysis.json", False),
        ("Error review records", f"{D2}/outputs/error_review.json", False),
        ("Checkpoints", f"{D2}/checkpoints", True), ("Results", f"{D2}/results.md", False),
        ("Notebook", f"{D2}/Task2_Sentiment_Colab.ipynb", False)]))

    # ===================================================================================== 4 TASK 3
    story += [PageBreak(), Paragraph("4 Task 3: CycleGAN Style Transfer between Monet Paintings and Photos", H1)]
    story.append(P(
        "The goal is unpaired translation between 300 Monet paintings (domain A) and 7,038 photos (domain B) "
        "with a CycleGAN [3] trained from random initialisation. Shibin's generators are ResNet encoder-decoders "
        "with nine residual blocks; upsampling uses nearest-neighbour resizing followed by a convolution instead of "
        "transposed convolutions, to avoid checkerboard artifacts. The discriminators are 70x70 PatchGANs [11]. "
        "Training combines a least-squares adversarial loss [12], a cycle-consistency loss (weight 10) and an identity "
        "loss (weight 5). The Kaggle score is computed with the instructor's script: FID and the script's MiFID on "
        "the first 300 sorted images in each direction, averaged, and the final score is the mean of the two "
        "values (lower is better). Besides FID [13] we report KID [14], generative precision and recall [15], "
        "density and coverage, LPIPS [16], the cycle-reconstruction error and a content similarity score. The "
        "submitted images are the unedited outputs of the trained generators; the pretrained Inception and LPIPS "
        "networks are only used to measure them."))
    story.append(P(
        "Denisha trained an independent CycleGAN with the same ResNet-9 generator depth but transposed-convolution "
        "upsampling, a PatchGAN discriminator with four stride-2 layers, an identity weight of 2.5, 60 epochs "
        "(30 constant and 30 with linear decay), gradient clipping at 5.0 and fp16 mixed precision on an RTX 4090. "
        "Her submission is the team's best Kaggle entry."))

    story.append(Paragraph("4.1 Model Comparison", H2))
    tms3 = teammate_models("task3")
    arch3 = ("Generators: ResNet-9 (c7s1-64, two stride-2 convs, 9 residual blocks, two resize-conv layers, tanh), "
             "InstanceNorm, reflection padding, 11.38M parameters each. Discriminators: 70x70 PatchGAN, 2.76M each.")
    hp3 = ("LSGAN, cycle weight 10, identity weight 5; Adam 2e-4 (0.5, 0.999); 20 epochs constant LR then 20 epochs "
           "linear decay, 2,000 unpaired pairs per epoch; batch 4; image pool 50; resize 286, random crop 256, flip")
    rows = [["", "Shibin: photo to Monet", "Shibin: Monet to photo"] + [esc(x["model"]) for x in tms3],
            ["Architecture", Paragraph(md(arch3), CELL), ""] + [esc(x.get("arch") or PEND) for x in tms3],
            ["Hyperparameters", Paragraph(md(hp3), CELL), ""] + [esc(x.get("hparams") or PEND) for x in tms3]]
    t3_rows = [("FID", "fid", 2), ("MiFID (instructor script)", "mifid_script", 4), ("KID", "kid", 4),
               ("Precision (k = 3)", "precision", 3), ("Recall (k = 3)", "recall", 3),
               ("Density / coverage (k = 5)", None, 3), ("Cycle reconstruction L1", "cycle_l1", 4),
               ("LPIPS, input vs translation", "lpips_input_vs_translation", 3),
               ("LPIPS, input vs reconstruction", "lpips_input_vs_reconstruction", 3),
               ("Content cosine similarity", "content_cosine_similarity", 3)]
    for label, key, nd in t3_rows:
        if key is None:
            vals = [f"{float(t3[(d, 'density')]):.3f} / {float(t3[(d, 'coverage')]):.3f}" for d in (B2A, A2B)]
        else:
            vals = [f"{float(t3[(d, key)]):.{nd}f}" for d in (B2A, A2B)]
        rows.append([label] + vals + [tm_val(x, label) for x in tms3])

    def trn(k):
        return t3[("training", k)]

    pub = kaggle.get("public_score")
    rank = kaggle.get("public_rank")
    both = [
        ("Kaggle file: FID / MiFID (average)", f"{fid_avg:.4f} / {mifid_avg:.4f}, score {kaggle_score:.2f}"),
        ("Kaggle leaderboard: score / rank", f"submitted (scores -{kaggle_score:.2f}); team best entry "
                                             f"{pub if pub is not None else PEND}, rank {rank if rank is not None else PEND}"),
        ("Final epoch losses: G / D_A / D_B", f"{float(trn('final_epoch_mean_loss_G')):.3f} / "
                                              f"{float(trn('final_epoch_mean_loss_D_A')):.3f} / {float(trn('final_epoch_mean_loss_D_B')):.3f}"),
        ("Final epoch cycle A / B, identity A / B", f"{float(trn('final_epoch_mean_cyc_A')):.3f} / {float(trn('final_epoch_mean_cyc_B')):.3f}, "
                                                    f"{float(trn('final_epoch_mean_idt_A')):.3f} / {float(trn('final_epoch_mean_idt_B')):.3f}"),
        ("Gradient norm G (mean / max), D (mean / max)", f"{float(trn('grad_norm_G_mean')):.1f} / {float(trn('grad_norm_G_max')):.0f}, "
                                                         f"{float(trn('grad_norm_D_mean')):.1f} / {float(trn('grad_norm_D_max')):.0f}"),
        ("Non-finite steps / total steps", f"{trn('nonfinite_steps')} / {int(float(trn('total_steps'))):,}"),
        ("Training time / throughput / peak GPU memory", f"{float(trn('training_time_min')):.1f} min / "
                                                         f"{float(trn('train_images_per_sec')):.1f} images/s / {float(trn('peak_gpu_allocated_mb')):,.0f} MB"),
        ("Total parameters", f"{int(float(trn('params_total'))):,}"),
        ("Human audit (2 raters, 30 samples)", shibin_audit_cell()),
        ("Hardware", trn("hardware")),
    ]
    for label, val in both:
        rows.append([label, Paragraph(esc(val), CELL), ""] + [tm_val(x, label) for x in tms3])
    ck = f"{T3}/checkpoints/{RUN3}"
    rows.append(["Checkpoints (SHA-256)", f"G_BA.pt {sha256(ck + '/G_BA.pt')[:12]}",
                 f"G_AB.pt {sha256(ck + '/G_AB.pt')[:12]}"] + [tm_val(x, "Checkpoints (SHA-256)") for x in tms3])
    ncol = 2 + len(tms3)
    story.append(tcap(f"Task 3 comparison. Directional metrics use the 300 scored images; the Kaggle score is "
                      f"(FID + MiFID) / 2. Shibin's run `{RUN3}`. In Denisha's column, two values are given as "
                      f"photo to Monet / Monet to photo."))
    TAB["t3"] = _tab[0]
    t = table(rows, [1.85 * inch] + [(WIDTH - 1.85 * inch) / ncol] * ncol, first_col_bold=True, font=8)
    first_both = 3 + len(t3_rows)
    t.setStyle(TableStyle([("SPAN", (1, 1), (2, 1)), ("SPAN", (1, 2), (2, 2))]
                          + [("SPAN", (1, i), (2, i)) for i in range(first_both, first_both + len(both))]))
    story.append(t)
    story.append(source(f"{T3}/full_metrics_report.csv", f"{T3}/submission.csv", f"{T3}/kaggle_leaderboard.json",
                        teammate=[f"{D3}/outputs/submission_metrics_official.json", f"{D3}/outputs/metrics_A2B.json",
                                  f"{D3}/outputs/metrics_B2A.json", f"{D3}/outputs/metrics.json", f"{D3}/results.md"]))
    story.append(Paragraph(
        "* Denisha's KID and precision/recall were computed with torch-fidelity against all 7,038 photos for "
        "Monet to photo, so they are not on exactly the same reference set as Shibin's. ** Denisha's cycle L1 was "
        "recorded on the [-1, 1] pixel scale (0.0914 and 0.0778) and is shown here divided by 2 to match Shibin's "
        "[0, 1] scale. *** Denisha's content cosine is computed on a different feature representation from "
        "Shibin's Inception features, so only the within-member comparison of directions is meaningful.", SMALL))

    story.append(Paragraph("4.2 Training Behaviour and Results", H2))
    story.append(figure(f"{T3}/outputs/{RUN3}/loss_curves.png",
                        "Task 3 adversarial, cycle and identity losses, total generator loss with the learning-rate "
                        "schedule, and gradient norms over 20,000 steps", max_h=3.2 * inch))
    fe = [r for r in t3_epochs if r["fid_B2A"]]
    fid_rows = [["Epoch"] + [r["epoch"] for r in fe],
                ["FID, photo to Monet"] + [f"{float(r['fid_B2A']):.1f}" for r in fe],
                ["FID, Monet to photo"] + [f"{float(r['fid_A2B']):.1f}" for r in fe]]
    n = len(fe)
    story.append(tcap("FID on the 300 scored images during training (monitoring only)."))
    story.append(table(fid_rows, [1.45 * inch] + [(WIDTH - 1.45 * inch) / n] * n, first_col_bold=True))
    story.append(source(f"reproducibility/raw_logs/task3_gan/shibin_thomas/{RUN3}/epochs.csv"))
    story.append(P(
        "Both discriminators settle below the least-squares balance point of 0.25 and keep improving slowly, while "
        "the generators' adversarial losses rise; by the end the Monet discriminator (loss 0.089) is clearly "
        "winning, a sign that it has started to memorise the 300 paintings. The cycle and identity losses fall "
        "smoothly over the whole run. FID improves quickly during the first 25 epochs and then stays within about "
        "4 points of 120, which is within the noise of a 300-image FID. No training step produced a NaN or "
        "infinite value."))
    story.append(figure(f"{D3}/outputs/loss_curves.png", "Task 3 generator, discriminator, cycle and identity losses "
                        "per epoch (Denisha Ketan Tank)", max_h=2.6 * inch, teammate=True))
    story.append(P(
        "Denisha's run shows the same balance: her discriminators finish at 0.084 (Monet) and 0.144 (photo), "
        "almost exactly the values of Shibin's run, while the cycle and identity losses decrease throughout. "
        "Her gradient-norm statistics were written as NaN in the metrics file even though the explicit NaN "
        "counter is zero; we report this as a logging defect rather than a training failure."))
    story.append(figure(f"{T3}/outputs/{RUN3}/final_samples_B2A.png",
                        "Photo to Monet. Top row: input photos; middle: generated paintings; bottom: reconstructions",
                        max_h=2.5 * inch))
    story.append(figure(f"{T3}/outputs/{RUN3}/final_samples_A2B.png",
                        "Monet to photo. Top row: input paintings; middle: generated photos; bottom: reconstructions",
                        max_h=2.5 * inch))

    import zipfile as _zip
    zf = _zip.ZipFile(io.BytesIO(tm_bytes(f"{D3}/outputs/images.zip")))
    first8 = sorted(p.name for p in (ROOT / T3 / "outputs/pred_B2A").glob("*.jpg"))[:8]
    grid = PILImage.new("RGB", (8 * 256, 2 * 256), "white")
    for i, name in enumerate(first8):
        grid.paste(PILImage.open(ROOT / T3 / "outputs/pred_B2A" / name).convert("RGB").resize((256, 256)), (i * 256, 0))
        grid.paste(PILImage.open(io.BytesIO(zf.read(f"{i:06d}.jpg"))).convert("RGB").resize((256, 256)), (i * 256, 256))
    story.append(figure("", "Photo to Monet for the same eight input photos (the first eight sorted photos). Top row: "
                        "Shibin's model; bottom row: Denisha's model", image=grid, max_h=2.0 * inch, ref=False))
    story.append(Paragraph("Source: " + path_ref(f"{T3}/outputs/pred_B2A", tree=True) + ", " +
                           path_ref(f"{D3}/outputs/images.zip", teammate=True) + f" (branch {TM_BRANCH})", SMALL))
    story.append(P(
        "Seen side by side, the two models make different trade-offs. Denisha's outputs are smoother, with "
        "little of the fine stipple pattern, and keep colours closer to the input photo: darker skies, greener "
        "fields and the contrast of the silhouettes survive. Shibin's outputs move further towards a pale pastel "
        "palette and carry a visible high-frequency stipple texture (Section 4.5), which reads as painterly up "
        "close but is not how Monet's brushwork looks. Smoother texture and colour statistics closer to real "
        "images are consistent with Denisha's lower FID."))

    story.append(Paragraph("4.3 Discussion", H2))
    story.append(P(
        "*Comparison.* Denisha's CycleGAN reaches a clearly better FID (98.4 and 103.2 against 118.1 and 117.3, "
        "average 100.8 against 117.7), which gives the team's Kaggle score of -50.60 and rank 34. The two runs "
        "differ mainly in the training recipe: 60 instead of 40 epochs, a lower identity weight (2.5 instead of "
        "5), which lets the generator move further from the input colours, and transposed-convolution "
        "upsampling. This agrees with the weaknesses Shibin's run showed (FID flat after epoch 30 and weak "
        "Monet style) and with the changes in his prepared second configuration. Denisha's run also reconstructs "
        "inputs more closely (cycle L1 0.046 and 0.039 on the [0, 1] scale against 0.062 and 0.053, LPIPS 0.22 "
        "and 0.28 against 0.38 and 0.47)."))
    story.append(P(
        "*Strengths.* Both trainings were stable from start to finish and no generator collapsed. The cycle constraint "
        "works: reconstructions keep the layout, objects and colours of the inputs (cycle L1 of 0.05 to 0.06), and "
        "photo-to-Monet translations preserve the content of the scene (content cosine 0.75) while moving the "
        "colours towards Monet's palette. Our implementation of the scoring protocol matches the instructor's script, "
        "so the submitted numbers can be verified."))
    story.append(P(
        "*Weaknesses.* Only 27% of the generated paintings fall inside the feature manifold of real Monets "
        "(precision 0.27). The main reasons are a regular stippled texture that replaces Monet's brush strokes "
        "and the washing out of dark photos. In the other direction the generated photos cover only a third of the "
        "photo distribution (recall 0.32) and still look painted. The Monet discriminator over-fits the small "
        "training set."))
    story.append(P(
        "*Limitations.* FID on 300 images has high variance, each member trained a single seed on different GPUs, "
        f"and some secondary metrics were computed with different tools (see the notes under Table {TAB['t3']}). The script's MiFID is "
        "a mean cosine distance between unrelated real and generated images and hardly changes between models, so "
        "the Kaggle score is driven almost entirely by FID. The human audit (Section 4.4) uses only 30 samples per model."))
    story.append(P(
        "*Next steps.* Based on these findings we prepared a second configuration (`configs/cyclegan_v2.yaml`) "
        "with differentiable augmentation for the discriminators [18], an exponential moving average of the "
        "generator weights, a lower identity weight of 2.5, 60 epochs, and selection of the exported epoch on "
        "held-out photos. A multi-scale discriminator and a low-frequency content loss are further options."))

    story.append(Paragraph("4.4 Human Audit", H2))
    story.append(P(
        "Each model was audited on 30 fixed samples. Each rater scored style (does the output look like the "
        "target domain), content preservation and visual artifacts from 1 to 5, without seeing the other "
        "rater's scores. Shibin's audit uses blinded panels (input next to output) drawn with a fixed seed from "
        "the scored images, 15 per direction, with a hidden key for the direction. Denisha's audit uses her "
        "first 30 photo-to-Monet outputs."))
    arows = [["Model and rater", "Style", "Content", "Artifacts"]]
    if aud_r1:
        for lab, d in (("all 30", None), ("photo to Monet (15)", "photo->monet"), ("Monet to photo (15)", "monet->photo")):
            m = aud_means(aud_r1, d)
            arows.append([f"Shibin's model, rater 1 (Shibin), {lab}"] + [f"{x:.2f}" for x in m])
    if aud_r2:
        m = aud_means(aud_r2)
        arows.append(["Shibin's model, rater 2 (Denisha), all 30"] + [f"{x:.2f}" for x in m])
    else:
        arows.append(["Shibin's model, rater 2 (Denisha)", PEND, PEND, PEND])
    if den_aud:
        cr = den_aud["criteria"]
        keys = ("style", "content", "artifact")
        arows.append(["Denisha's model, rater 1 (Denisha)"] + [f"{cr[k]['mean_rater_1']:.2f}" for k in keys])
        arows.append(["Denisha's model, rater 2 (Shibin)"] + [f"{cr[k]['mean_rater_2']:.2f}" for k in keys])
        arows.append(["Denisha's model, exact / within-1 agreement"] +
                     [f"{cr[k]['exact_agreement']:.0%} / {cr[k]['within_1_agreement']:.0%}" for k in keys])
        arows.append(["Denisha's model, Cohen's kappa (unweighted)"] +
                     [f"{cr[k]['cohen_kappa']:.2f}" if cr[k]["cohen_kappa"] is not None else "undefined" for k in keys])
    story.append(tcap("Human audit: mean scores (1 to 5) and inter-rater agreement."))
    story.append(table(arows, [3.2 * inch] + [(WIDTH - 3.2 * inch) / 3] * 3))
    story.append(source(f"{AUD}/rater1.csv", f"{AUD}/rater2.csv", f"{AUD}/audit_key.csv",
                        "report/human_audit/denisha_model_human_audit.csv",
                        "report/human_audit/denisha_model_human_audit_agreement.json"))
    story.append(P(
        "Both models were rated highly for content preservation and moderately high for style. For Shibin's "
        "model, rater 1 scored artifacts highest and style lowest, and the photo-to-Monet direction slightly "
        "higher for style and content than Monet-to-photo, which matches the automatic metrics (more content "
        "kept, weaker painterly realism). On Denisha's model the two raters agree within one point on 93% to "
        "100% of the samples, but the exact agreement is only 40% for content and Cohen's kappa is close to zero "
        "for all three criteria. This is a known limitation of kappa rather than real disagreement: rater 1 gave "
        "every image the same content score of 5 and used only two values for style and artifacts, so there is "
        "almost no variance for agreement to exceed chance. Rater 2 was stricter on content (mean 4.33 against "
        "5.00) and more lenient on artifacts (4.30 against 4.03). With 30 samples and scores concentrated at 4 "
        "and 5, a wider use of the scale or more samples would be needed for a meaningful kappa."
        + ("" if aud_r2 else " The second rating of Shibin's model is still to be added.")))

    story.append(Paragraph("4.5 Failure Analysis (Shibin Thomas)", H2))
    story.append(P(
        "We ranked all translations by cycle error, by content similarity and by the distance to the nearest real "
        "image of the target domain, and inspected the worst four images for each criterion. In each figure below "
        "the columns show the input (top), the translation (middle) and the reconstruction (bottom)."))
    story.append(figure(f"{T3}/outputs/{RUN3}/failure_B2A_nn_dist.png",
                        "Dark and night photos translated to Monet style", width=WIDTH * 0.62, max_h=3.4 * inch))
    story.append(P(
        "*Dark photos.* Low-light scenes such as a river at dusk or a camel silhouette against a sunset are turned "
        "into bright, washed-out images covered in coloured speckles, and silhouettes lose their edges. These are "
        "the least realistic outputs. The Monet set contains almost no night scenes, so the generator brightens "
        "every image. Colour augmentation of the discriminator's inputs is the fix we included in the second "
        "configuration."))
    story.append(figure(f"{T3}/outputs/{RUN3}/failure_B2A_cycle_l1.png",
                        "Photos with the highest cycle-reconstruction error", width=WIDTH * 0.62, max_h=3.4 * inch))
    story.append(P(
        "*Recoloured warm tones.* Strong reds and oranges are replaced by Monet's blues and lilacs: an orange sky "
        "becomes blue and a red poppy turns pale orange. The reconstructions nevertheless recover the original "
        "colours, which means the generator hides the information in a faint signal that the cycle loss can read "
        "back, an effect described by Chu et al. [17]. When this hidden signal is not enough, the cycle error is "
        "highest (0.13 to 0.17, about 2.5 times the average)."))
    story.append(figure(f"{T3}/outputs/{RUN3}/failure_A2B_content_cos.png",
                        "Monet paintings with the lowest content similarity after translation to photos",
                        width=WIDTH * 0.62, max_h=3.4 * inch))
    story.append(P(
        "*Foggy paintings.* Pale, low-contrast paintings of fog on the Thames become saturated orange sunsets or "
        "turquoise storms, and a thin poplar tree disappears. Photos are rarely foggy, so the generator invents "
        "contrast and colour. We also observed a regular stipple texture in most photo-to-Monet outputs and visible "
        "brush strokes and signatures in most Monet-to-photo outputs. The full analysis of all five failure types "
        "is in `failure_analysis.md`."))

    story.append(Paragraph("4.6 Evidence", H2))
    story.append(evidence([
        ("Configuration", f"{T3}/configs/cyclegan_v1.yaml", False),
        ("Training and evaluation logs", f"reproducibility/raw_logs/task3_gan/shibin_thomas/{RUN3}", True),
        ("Run manifest", f"reproducibility/manifests/task3_gan/shibin_thomas/{RUN3}.json", False),
        ("Submitted predictions", f"{T3}/outputs", True),
        ("Kaggle submission file", f"{T3}/submission.csv", False),
        ("Plots and sample grids", f"{T3}/outputs/{RUN3}", True),
        ("Generator checkpoints", f"{T3}/checkpoints/{RUN3}", True),
        ("Human audit material", f"{T3}/outputs/human_audit", True),
        ("Results and analysis", f"{T3}/results.md", False),
        ("Failure analysis", f"{T3}/failure_analysis.md", False),
        ("Executed notebook", f"{T3}/src/task3_cyclegan.ipynb", False),
    ], [("Configuration", f"{D3}/config_competition.yaml", False),
        ("Training log", "reproducibility/raw_logs/task3_denisha.log", False),
        ("Run manifest", "reproducibility/manifests/task3_denisha.json", False),
        ("Official evaluation", f"{D3}/outputs/submission_metrics_official.json", False),
        ("Kaggle submission file", f"{D3}/outputs/submission.csv", False),
        ("Generated images (photo to Monet)", f"{D3}/outputs/images.zip", False),
        ("Loss history and curves", f"{D3}/outputs/history.json", False),
        ("Human audit sheet", f"{D3}/outputs/human_audit.csv", False), ("Results", f"{D3}/results.md", False),
        ("Notebook", f"{D3}/Task3_VSCode.ipynb", False)]))

    # ===================================================================================== 5 checklist, refs
    story += [PageBreak(), Paragraph("5 Pre-Submission Checklist", H1)]
    kaggle_done = kaggle.get("public_rank") is not None
    merged = (ROOT / "task1_llm" / "member_denisha").exists()
    folders = ("Done for both members" if merged else
               f"Shibin: branch {INFO['repo_branch']}; Denisha: member_denisha folders on branch {TM_BRANCH}")
    audit = ("Done for both models" if (aud_r1 and aud_r2 and den_aud) else
             "Denisha's model: done (2 raters). Shibin's model: rater 1 done, rater 2 pending" if (aud_r1 and den_aud)
             else "Pending")
    ck_rows = [["Item", "Status"],
               ["Every member's folder exists under all three tasks with code, configuration, logs and results.md", folders],
               ["No two members share the same architecture and hyperparameters",
                "Confirmed (Section 2 to 4); the two TextCNNs share layer sizes but differ in batch size, dropout, "
                "schedule, epochs and precision"],
               ["Task 1 uses no prebuilt Transformer or attention modules", "Done for both members"],
               ["Task 2 uses no pretrained embeddings or language models; three models per member", "Done for both members"],
               ["All required metrics are reported for every model in Tasks 1 to 3",
                f"Done (Tables {TAB['t1']}, {TAB['t2']} and {TAB['t3']}); a few secondary values that a member did not record are marked"],
               ["Task 3 Kaggle submission made and leaderboard rank recorded",
                f"Done: team score {kaggle.get('public_score')}, rank {kaggle.get('public_rank')}" if kaggle_done
                else "Submitted; rank to be recorded"],
               ["Task 3 two-rater human audit (30 samples)", audit],
               ["Comparison tables with architecture, hyperparameters and metrics for all three tasks",
                f"Done (Tables {TAB['t1']}, {TAB['t2']} and {TAB['t3']})"],
               ["Raw logs and manifests committed and unmodified", "Done"]]
    story.append(table(ck_rows, [WIDTH - 2.6 * inch, 2.6 * inch]))

    story.append(Paragraph("References", H1))
    refs = [
        "A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin. Attention is all you need. In <i>NeurIPS</i>, 2017.",
        "R. Eldan and Y. Li. TinyStories: How small can language models be and still speak coherent English? <i>arXiv:2305.07759</i>, 2023.",
        "J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros. Unpaired image-to-image translation using cycle-consistent adversarial networks. In <i>ICCV</i>, 2017.",
        "A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, and I. Sutskever. Language models are unsupervised multitask learners. OpenAI technical report, 2019.",
        "R. Xiong et al. On layer normalization in the Transformer architecture. In <i>ICML</i>, 2020.",
        "X. Zhang, J. Zhao, and Y. LeCun. Character-level convolutional networks for text classification. In <i>NeurIPS</i>, 2015.",
        "Y. Kim. Convolutional neural networks for sentence classification. In <i>EMNLP</i>, 2014.",
        "A. Joulin, E. Grave, P. Bojanowski, and T. Mikolov. Bag of tricks for efficient text classification. In <i>EACL</i>, 2017.",
        "D. Bahdanau, K. Cho, and Y. Bengio. Neural machine translation by jointly learning to align and translate. In <i>ICLR</i>, 2015.",
        "C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger. On calibration of modern neural networks. In <i>ICML</i>, 2017.",
        "P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros. Image-to-image translation with conditional adversarial networks. In <i>CVPR</i>, 2017.",
        "X. Mao, Q. Li, H. Xie, R. Y. K. Lau, Z. Wang, and S. P. Smolley. Least squares generative adversarial networks. In <i>ICCV</i>, 2017.",
        "M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter. GANs trained by a two time-scale update rule converge to a local Nash equilibrium. In <i>NeurIPS</i>, 2017.",
        "M. Bińkowski, D. J. Sutherland, M. Arbel, and A. Gretton. Demystifying MMD GANs. In <i>ICLR</i>, 2018.",
        "T. Kynkäänniemi, T. Karras, S. Laine, J. Lehtinen, and T. Aila. Improved precision and recall metric for assessing generative models. In <i>NeurIPS</i>, 2019.",
        "R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang. The unreasonable effectiveness of deep features as a perceptual metric. In <i>CVPR</i>, 2018.",
        "C. Chu, A. Zhmoginov, and M. Sandler. CycleGAN, a master of steganography. <i>NeurIPS Workshop</i>, 2017.",
        "S. Zhao, Z. Liu, J. Lin, J.-Y. Zhu, and S. Han. Differentiable augmentation for data-efficient GAN training. In <i>NeurIPS</i>, 2020.",
    ]
    for i, r in enumerate(refs, 1):
        story.append(Paragraph(f"[{i}] {r}", REF))

    def on_page(canvas, doc):
        if doc.page == 1:
            return
        canvas.saveState()
        canvas.setFont("Serif", 9)
        canvas.drawCentredString(letter[0] / 2, 0.55 * inch, str(doc.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(str(out), pagesize=letter, leftMargin=1.0 * inch, rightMargin=1.0 * inch,
                            topMargin=0.9 * inch, bottomMargin=0.9 * inch,
                            title=f"DATA 266 Lab 1 Report, Team {INFO['team_number']}",
                            author=", ".join(mm["name"] for mm in members), subject="DATA 266 Lab 1",
                            creator="", producer="")
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return out


if __name__ == "__main__":
    p = build()
    print(f"wrote {p.relative_to(ROOT).as_posix()} ({p.stat().st_size / 1e6:.1f} MB)")
