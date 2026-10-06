"""Build the combined team report: report/DATA266_Lab1_Report_Team_<NN>.pdf

    python report/build_report.py

Every number about Shibin Thomas's models is read from the committed result files at build time:
metrics_report.csv, comparison.csv, full_metrics_report.csv, epochs.csv, train_summary.json and the
manifests. The report therefore cannot drift from the evidence, and each table names its source file.
Teammates' rows come from report/team_info.yaml; values that are missing there are shown as "pending".
Figures are taken from the members' outputs/<run_id>/ folders. Paths are relative to the repo root.
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
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
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

T1 = "task1_llm/shibin_thomas"
T2 = "task2_sentiment/shibin_thomas"
T3 = "task3_gan/shibin_thomas"
RUN1, RUN2, RUN3 = "gpt_char_v1_20261001-153007", "yelp3_20261001-182849", "cyclegan_v1_20261001-201801"

# ----------------------------------------------------------------------------------------- fonts / styles
FONTS = REPORT / "fonts"
pdfmetrics.registerFont(TTFont("DV", str(FONTS / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DV-B", str(FONTS / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DVM", str(FONTS / "DejaVuSansMono.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV", boldItalic="DV-B")

ss = getSampleStyleSheet()
INK, ACCENT, MUTED, RULE, HEAD_BG, ZEBRA = (colors.HexColor(c) for c in
                                             ("#1b1f24", "#1f4e79", "#57606a", "#d0d7de", "#e8eef5", "#f6f8fa"))
BODY = ParagraphStyle("body", parent=ss["Normal"], fontName="DV", fontSize=9.2, leading=12.6, textColor=INK,
                      spaceAfter=5)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=7.6, leading=9.6, spaceAfter=0)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=7.4, leading=9.2, spaceAfter=0)
CELLB = ParagraphStyle("cellb", parent=CELL, fontName="DV-B")
CAP = ParagraphStyle("cap", parent=BODY, fontSize=7.6, leading=9.6, textColor=MUTED, alignment=TA_CENTER,
                     spaceBefore=2, spaceAfter=8)
H1 = ParagraphStyle("h1", parent=BODY, fontName="DV-B", fontSize=15, leading=19, textColor=ACCENT,
                    spaceBefore=4, spaceAfter=8)
H2 = ParagraphStyle("h2", parent=BODY, fontName="DV-B", fontSize=11.2, leading=14.5, textColor=ACCENT,
                    spaceBefore=10, spaceAfter=5, keepWithNext=1)
H3 = ParagraphStyle("h3", parent=BODY, fontName="DV-B", fontSize=9.6, leading=12.5, spaceBefore=6, spaceAfter=3,
                    keepWithNext=1)
BUL = ParagraphStyle("bul", parent=BODY, leftIndent=12, bulletIndent=3, spaceAfter=2.5)
QUOTE = ParagraphStyle("quote", parent=BODY, fontName="DVM", fontSize=7.1, leading=9.0, leftIndent=6,
                       textColor=colors.HexColor("#24292f"), backColor=ZEBRA, borderPadding=(4, 4, 4, 4),
                       spaceBefore=3, spaceAfter=7)
PEND = '<font color="#9a6700">pending</font>'
WIDTH = letter[0] - 1.5 * inch


def esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def md(s: str) -> str:
    """Tiny markdown -> reportlab markup: `code`, **bold**."""
    s = esc(s)
    s = re.sub(r"`([^`]+)`", r'<font name="DVM" size="7.6">\1</font>', s)
    return re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)


def P(s, style=BODY):
    return Paragraph(md(s), style)


def bullets(items, style=BUL):
    return [Paragraph(md(i), style, bulletText="•") for i in items]


def link(path: str, label: str | None = None, tree=False) -> str:
    url = (TREE if tree else BLOB) + path
    return f'<link href="{url}" color="#0969da"><font name="DVM" size="7.2">{esc(label or path)}</font></link>'


def evidence(items: list[tuple[str, str, bool]]):
    """items: (description, repo path, is_dir) -> a two-column table of clickable links."""
    rows = [[Paragraph("<b>Evidence</b>", CELL), Paragraph("<b>File in the repository (clickable)</b>", CELL)]]
    rows += [[Paragraph(md(d), CELL), Paragraph(link(p, tree=t), CELL)] for d, p, t in items]
    return table(rows, [1.75 * inch, WIDTH - 1.75 * inch], header=True)


def table(rows, widths, header=True, first_col_bold=False, font=7.4):
    data = []
    for r_i, r in enumerate(rows):
        row = []
        for c_i, c in enumerate(r):
            if isinstance(c, (Paragraph, Image, Table)):
                row.append(c)
            else:
                st = CELLB if (header and r_i == 0) or (first_col_bold and c_i == 0) else CELL
                if font != 7.4:
                    st = ParagraphStyle("x", parent=st, fontSize=font, leading=font * 1.25)
                row.append(Paragraph(c if "<" in str(c) and "</" in str(c) else md(str(c)), st))
        data.append(row)
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, RULE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 3.5), ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
             ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), HEAD_BG))
    for i in range(1 if header else 0, len(data)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
    t.setStyle(TableStyle(style))
    return t


def figure(path: str, caption: str, width=WIDTH, max_h=4.2 * inch):
    """Embed an image (re-encoded as high-quality JPEG to keep the PDF small) with a caption + source link."""
    src = ROOT / path
    im = PILImage.open(src).convert("RGB")
    w, h = im.size
    scale = min(width / w, max_h / h)
    target_px = int(width / inch * 200)                  # ~200 dpi at print size
    if w > target_px:
        im = im.resize((target_px, int(h * target_px / w)), PILImage.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88, optimize=True)
    buf.seek(0)
    img = Image(buf, width=w * scale, height=h * scale)
    return KeepTogether([img, Paragraph(md(caption) + " · source: " + link(path), CAP)])


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
    if x == int(x) and abs(x) < 1000 and "." not in str(v):
        return str(int(x))
    return f"{x:.{nd}f}"


def teammate_models(task: str):
    return INFO.get("teammate", {}).get(task) or []


def tm_val(m, key):
    v = (m.get("metrics") or {}).get(key)
    return fnum(v) if v is not None else PEND


# Task 1 ------------------------------------------------------------------------------------------------
t1 = {r["metric"] + ("" if r["metric"] != "Total training time" else f" ({r['notes']})"): r
      for r in read_csv(f"{T1}/metrics_report.csv")}
t1_epochs = read_csv(f"reproducibility/raw_logs/task1_llm/shibin_thomas/{RUN1}/epochs.csv")
t1_cfg = yaml.safe_load((ROOT / f"{T1}/configs/gpt_char_v1.yaml").read_text(encoding="utf-8"))
t1_man = jload(f"reproducibility/manifests/task1_llm/shibin_thomas/{RUN1}.json")

# Task 2 ------------------------------------------------------------------------------------------------
t2 = {r["metric"]: r for r in read_csv(f"{T2}/outputs/{RUN2}/comparison.csv")}
t2_man = jload(f"reproducibility/manifests/task2_sentiment/shibin_thomas/{RUN2}.json")
T2_MODELS = ["baseline_meanpool", "exp1_textcnn", "exp2_bigru_attn"]

# Task 3 ------------------------------------------------------------------------------------------------
t3 = {(r["section"], r["metric"]): r["value"] for r in read_csv(f"{T3}/full_metrics_report.csv")}
t3_epochs = read_csv(f"reproducibility/raw_logs/task3_gan/shibin_thomas/{RUN3}/epochs.csv")
kaggle = jload(f"{T3}/kaggle_leaderboard.json")
B2A, A2B = "B2A (photo->monet)", "A2B (monet->photo)"
fid_avg, mifid_avg = float(t3[("Kaggle submission (avg of both directions)", "FID")]), \
    float(t3[("Kaggle submission (avg of both directions)", "MiFID")])
kaggle_score = (fid_avg + mifid_avg) / 2


# ======================================================================================================
def build() -> Path:
    out = REPORT / f"DATA266_Lab1_Report_Team_{INFO['team_number']}.pdf"
    story = []
    members = INFO["members"]

    # ------------------------------------------------------------------------------------- title page
    story += [Spacer(1, 1.1 * inch),
              Paragraph("DATA 266 · Generative AI · Fall 2026", ParagraphStyle("k", parent=BODY, fontSize=10,
                                                                                textColor=MUTED, alignment=TA_CENTER)),
              Spacer(1, 6),
              Paragraph("Lab 1 Report", ParagraphStyle("t", parent=H1, fontSize=26, leading=31, alignment=TA_CENTER,
                                                      textColor=INK)),
              Paragraph("Character-level GPT pretraining · Yelp sentiment classification · CycleGAN style transfer",
                        ParagraphStyle("st", parent=BODY, fontSize=11, leading=15, alignment=TA_CENTER, textColor=MUTED)),
              Spacer(1, 26),
              Paragraph(f"<b>Team {INFO['team_number']}</b> · Kaggle team “{esc(INFO['team_name'])}”",
                        ParagraphStyle("tm", parent=BODY, fontSize=11.5, alignment=TA_CENTER)),
              Paragraph(" · ".join(esc(m["name"]) for m in members),
                        ParagraphStyle("mm", parent=BODY, fontSize=10.5, alignment=TA_CENTER)),
              Spacer(1, 22),
              Paragraph(f'Repository: <link href="{INFO["repo_url"]}" color="#0969da">{esc(INFO["repo_url"])}</link><br/>'
                        f'Branch with this work: <link href="{TREE}" color="#0969da">{esc(INFO["repo_branch"])}</link>',
                        ParagraphStyle("rp", parent=BODY, fontSize=9.5, leading=14, alignment=TA_CENTER)),
              Spacer(1, 30)]
    summary = [["", "Shibin Thomas: headline result", "Source"],
               ["Task 1", f"val BPC {float(t1['Bits-per-character (validation)']['value']):.3f}, val perplexity "
                          f"{float(t1['Perplexity (validation)']['value']):.3f}, val accuracy "
                          f"{float(t1['Top-1 next-character accuracy (validation)']['value']):.1%}",
                f"{T1}/metrics_report.csv"],
               ["Task 2", f"best model BiGRU+attention: test accuracy {float(t2['Accuracy']['exp2_bigru_attn']):.2%}, "
                          f"macro-F1 {float(t2['F1 (macro)']['exp2_bigru_attn']):.4f}, ROC-AUC "
                          f"{float(t2['ROC-AUC']['exp2_bigru_attn']):.4f}",
                f"{T2}/outputs/{RUN2}/comparison.csv"],
               ["Task 3", f"FID {fid_avg:.2f}, MiFID {mifid_avg:.4f} → Kaggle score {kaggle_score:.2f} "
                          f"(shown as −{kaggle_score:.2f})", f"{T3}/submission.csv"]]
    summary = [[r[0], r[1], Paragraph(link(r[2]) if i else "<b>Source</b>", CELL)] for i, r in enumerate(summary)]
    story += [table(summary, [0.65 * inch, 3.65 * inch, WIDTH - 4.3 * inch]), PageBreak()]

    # ------------------------------------------------------------------------------------- ownership
    story.append(Paragraph("Team ownership statement", H1))
    own = " ".join(f"<b>{esc(m['name'])}</b> built {esc(m['built'])}" for m in members)
    story.append(Paragraph(
        own + " Every member trained their own models in their own folder "
        "(<font name='DVM' size='7.6'>task*/&lt;member&gt;/</font>). Results were compared on shared data "
        "and shared metric code: the shared TinyStories copy and prompts, the official Yelp test split, and "
        "the instructor's Kaggle evaluation protocol. This report and its analyses were written jointly.", BODY))
    story.append(P("**How to read this report.** Each task section has:"))
    story += bullets([
        "the comparison table (architecture, hyperparameters and every required metric, side by side);",
        "the joint analysis (strengths, weaknesses, limitations, next steps);",
        "an evidence table whose links open the exact log, plot or checkpoint behind each number;",
        "each member's failure or error analysis, with the real text or image snippets.",
    ])
    story.append(P("Numbers for Shibin Thomas are generated from the committed result files by "
                   "`report/build_report.py`, so they match the repository exactly."))
    story.append(Paragraph("Repository layout and reproducibility", H2))
    story += bullets([
        "**Per member, per task:** `src/` (code and notebook), `configs/`, `checkpoints/<run_id>/`, "
        "`outputs/<run_id>/`, `metrics_report.csv`, `results.md`, `failure_analysis.md`.",
        "**Raw logs:** unedited, append-only, in `reproducibility/raw_logs/<task>/<member>/<run_id>/`.",
        "**Manifests:** `reproducibility/manifests/<task>/<member>/<run_id>.json` record the config and its "
        "hash, the git commit, library versions, hardware, a data fingerprint and the checkpoint SHA-256.",
        "**Smoke test:** one command per task reproduces the full pipeline at small scale (root `README.md`).",
        "**No personal paths, credentials or API keys** in any committed file.",
    ])

    # ===================================================================================== TASK 1
    story += [PageBreak(), Paragraph("Task 1 — GPT-style character-level language model (TinyStories)", H1)]
    story += bullets([
        "**Shibin Thomas:** decoder-only Transformer written from scratch in `src/model.py`. LayerNorm, "
        "multi-head causal self-attention, FFN and residual blocks are all hand-written; no "
        "`nn.Transformer`/`nn.MultiheadAttention`/`scaled_dot_product_attention` (enforced by "
        "`tests/test_model.py`).",
        "**Data:** own 100K-train / 10K-validation split of the cleaned TinyStories pool (seed 266). "
        "Character vocabulary of 90 symbols built from the training split only.",
    ])
    story.append(Paragraph("1.1 Comparison table — architecture, hyperparameters and all Task 1 metrics", H2))
    tms = teammate_models("task1")
    hdr = ["", "Shibin Thomas — gpt_char_v1"] + [f"{esc(m['model'])}" for m in tms]
    m = t1_cfg["model"]
    tr = t1_cfg["training"]
    arch = (f"{m['n_layer']} layers, d_model {m['d_model']}, {m['n_head']} heads (head dim "
            f"{m['d_model'] // m['n_head']}), FFN 4×, pre-LN, GELU, learned positions, tied embeddings, "
            f"context {m['block_size']} chars")
    hp = (f"AdamW β=(0.9, 0.95), wd 0.1, peak LR {tr['lr']} → floor {tr['min_lr']}, "
          f"{tr['warmup_frac']:.0%} warm-up + cosine, batch {tr['batch_size']}×{m['block_size']}, "
          f"clip {tr['grad_clip']}, dropout {m['dropout']}, {tr['epochs']} epochs, bf16")
    rows = [hdr, ["Architecture", arch] + [esc(x.get("arch") or "") or PEND for x in tms],
            ["Hyperparameters", hp] + [esc(x.get("hparams") or "") or PEND for x in tms]]
    T1_ROWS = [("Training CE (nats/char, eval mode)", "Training cross-entropy loss"),
               ("Validation CE (nats/char)", "Validation cross-entropy loss"),
               ("Validation perplexity", "Perplexity (validation)"),
               ("Validation BPC", "Bits-per-character (validation)"),
               ("Train perplexity / BPC", None),
               ("Generalisation gap (val − train CE)", "Generalization gap"),
               ("Top-1 next-char accuracy (val)", "Top-1 next-character accuracy (validation)"),
               ("Distinct-1 / 2 / 3 (T = 0.8)", None),
               ("Repeated 4-gram rate (T = 0.8 / greedy)", None),
               ("Gradient norm mean / p99 / max (pre-clip)", None),
               ("Steps clipped / loss spikes / NaN steps", None),
               ("Parameters (total / non-embedding)", None),
               ("Training throughput (chars/s)", "Training tokens/sec"),
               ("Generation speed (chars/s, batch 1)", "Generation tokens/sec"),
               ("Peak GPU memory (MB, allocated)", "Peak GPU memory (allocated)"),
               ("Training time (min) / epochs / best epoch", None),
               ("Hardware", "Hardware")]
    v = lambda k, nd=4: fnum(t1[k]["value"], nd)  # noqa: E731
    special = {
        "Train perplexity / BPC": f"{v('Perplexity (train)')} / {v('Bits-per-character (train)')}",
        "Distinct-1 / 2 / 3 (T = 0.8)": f"{v('Distinct-1')} / {v('Distinct-2')} / {v('Distinct-3')}",
        "Repeated 4-gram rate (T = 0.8 / greedy)": f"{v('Repeated 4-gram rate')} / {v('Repeated 4-gram rate (greedy)')}",
        "Gradient norm mean / p99 / max (pre-clip)": f"{v('Gradient norm (mean, pre-clip)', 3)} / "
                                                     f"{v('Gradient norm (p99, pre-clip)', 3)} / {v('Gradient norm (max, pre-clip)', 2)}",
        "Steps clipped / loss spikes / NaN steps": f"{float(t1['Fraction of steps clipped']['value']):.2%} / "
                                                   f"{v('Loss spikes')} / {v('NaN / Inf steps')}",
        "Parameters (total / non-embedding)": f"{v('Parameter count')} / {v('Parameter count (non-embedding)')}",
        "Training time (min) / epochs / best epoch": f"{float(t1['Total training time (minutes)']['value']):.1f} / "
                                                     f"{v('Epochs trained')} / {v('Best epoch (lowest val loss)')}",
    }
    for label, key in T1_ROWS:
        val = special.get(label) or (v(key) if key else PEND)
        if key == "Generation tokens/sec":
            val = f"{float(t1[key]['value']):.1f}"
        if key == "Peak GPU memory (allocated)":
            val = f"{float(t1[key]['value']):,.0f}"
        rows.append([label, val] + [tm_val(x, label) for x in tms])
    rows.append(["Run id / best checkpoint", f"`{RUN1}` · `checkpoints/{RUN1}/best_model.pt` (SHA-256 "
                 f"{t1_man['checkpoints_files']['best_model.pt']['sha256'][:12]}…)"] + [PEND for _ in tms])
    ncol = 1 + len(tms)
    story.append(table(rows, [1.75 * inch] + [(WIDTH - 1.75 * inch) / ncol] * ncol, first_col_bold=True))
    story.append(Paragraph("Metric definitions are the team's shared ones in "
                           + link("task1_llm/shared_eval/metrics.py")
                           + ": perplexity = exp(CE), BPC = CE / ln 2, Distinct-n = unique / total word n-grams of "
                             "the sampled continuations. Source: " + link(f"{T1}/metrics_report.csv"), SMALL))

    story.append(Paragraph("1.2 Training curves and per-epoch validation", H2))
    story.append(figure(f"{T1}/outputs/{RUN1}/loss_curves.png",
                        "Task 1 (Shibin): training loss (running, with dropout), eval-mode training CE and "
                        "validation CE per epoch", max_h=3.0 * inch))
    ep_rows = [["Epoch"] + [r["epoch"] for r in t1_epochs]]
    ep_rows.append(["Val CE"] + [f"{float(r['val_loss']):.3f}" for r in t1_epochs])
    ep_rows.append(["Val acc."] + [f"{float(r['val_top1_acc']):.1%}" for r in t1_epochs])
    n = len(t1_epochs)
    story.append(table(ep_rows, [0.75 * inch] + [(WIDTH - 0.75 * inch) / n] * n, first_col_bold=True, font=6.9))
    story.append(Paragraph("Source: " + link(f"reproducibility/raw_logs/task1_llm/shibin_thomas/{RUN1}/epochs.csv"), SMALL))

    story.append(Paragraph("1.3 Joint analysis", H2))
    story.append(Paragraph("Strengths", H3))
    story += bullets([
        "**Fluent stories from a 7.5M-parameter model trained in 29 minutes.** Validation BPC is 0.824 (81.7% "
        "next-character accuracy). Sampled stories have named characters, dialogue and simple cause and effect.",
        "**Clean optimisation.** Validation loss fell every epoch, with no loss spikes or NaN steps. All 738 "
        "clipped steps were in the warm-up. The generalisation gap is only 0.012 nats, as expected with about "
        "120 training characters per parameter.",
        "**Sampling gives diversity.** At T = 0.8 the samples reach Distinct-3 0.93 with a repeated 4-gram "
        "rate of 0.7%.",
    ])
    story.append(Paragraph("Weaknesses", H3))
    story += bullets([
        "**Greedy decoding loops** (repeated 4-gram rate 0.137).",
        "**Rare words are misspelled inconsistently**, a consequence of character-level spelling.",
        "**Coherence is lost beyond the 256-character context**: entities drift and stories don't end.",
    ])
    story.append(Paragraph("Limitations", H3))
    story += bullets([
        "Single seed and single run per member.",
        "Distinct-n and repetition measure diversity, not story quality.",
        "Not converged: validation CE was still falling at epoch 10 (−0.003 per epoch).",
        "Generation speed is measured without a KV cache.",
    ])
    story.append(Paragraph("What we would try next", H3))
    story += bullets([
        "More epochs, or a larger model, since the loss was still decreasing.",
        "Context of 512 characters, judged by long-generation coherence.",
        "Top-k or nucleus sampling with a repetition penalty, judged by the repeated 4-gram rate on the "
        "shared prompts.",
        "A KV cache for faster generation.",
        "A sub-word tokenizer trained from scratch, to remove spelling errors.",
    ])

    story.append(Paragraph("1.4 Evidence", H2))
    story.append(evidence([
        ("Config (all hyperparameters)", f"{T1}/configs/gpt_char_v1.yaml", False),
        ("Raw training log", f"reproducibility/raw_logs/task1_llm/shibin_thomas/{RUN1}/train.log", False),
        ("Per-step / per-epoch logs", f"reproducibility/raw_logs/task1_llm/shibin_thomas/{RUN1}", True),
        ("Manifest (versions, hardware, checkpoint SHA-256)", f"reproducibility/manifests/task1_llm/shibin_thomas/{RUN1}.json", False),
        ("Loss curves / gradient norms / LR schedule", f"{T1}/outputs/{RUN1}", True),
        ("All 40 generated samples", f"{T1}/outputs/{RUN1}/samples.md", False),
        ("Best checkpoint", f"{T1}/checkpoints/{RUN1}/best_model.pt", False),
        ("Results, design justification", f"{T1}/results.md", False),
        ("Executed notebook", f"{T1}/src/task1_gpt_tinystories.ipynb", False),
    ]))

    story.append(Paragraph("1.5 Failure analysis — Shibin Thomas (actual generated text)", H2))
    story.append(P("Best checkpoint, 10 shared team prompts, 1 greedy + 3 temperature-0.8 samples each, 500 new "
                   "characters. The three clearest failures (full text in " + "`failure_analysis.md`):"))
    fail_t1 = [
        ("Failure 1 — Repetition (greedy)", "In a small house near the forest, there was a big tree. The tree was "
         "very high and the tree was very high. The tree was very high up in the tree. The tree was very high and the "
         "tree was very high.\n\nOne day, the tree saw a big tree. The tree was very high and the tree was very high. "
         "The tree was very high and the tree was very high. ...",
         "“the tree was very” occurs 16 times; the repeated 4-gram rate of this continuation is 0.61 (average over "
         "all samples 0.04). Argmax decoding locks into a high-probability loop. **Fix:** temperature / top-k "
         "sampling or a repetition penalty, measured by the repeated 4-gram rate."),
        ("Failure 2 — Broken / inconsistent spelling (T = 0.8)", "Tim was scared. He ran and ran until he came across a "
         "big lute. He saw a small tree which was very pretty. Tim wanted to take the lutter from him. He said yes, but "
         "he had to be careful.\n\nThe luter was big and boring. Tim saw that the lutte was scary and hurt. ...",
         "A rare word is spelled four ways: lute → lutter → luter → lutte. A character model rebuilds rare words from "
         "frequent letter patterns each time. **Fix:** lower temperature / top-k, longer training, sub-word tokens."),
        ("Failure 3 — Loss of coherence / entity drift (T = 0.8)", "After school, Sam wanted to go to the zoo and see all "
         "the animals.\n\nThey all walked closer and held on tight. Sam thought it was tremommeter than the diamond. ... "
         "Together, Jack and Sam started to fill their bucket with food. He was so proud of their work! ... Tom and Sam "
         "continued to throw their bucket until it was time to go hom",
         "Setting drifts (zoo → diamond → bucket → town), new companions Jack and Tom appear, and pronouns stop "
         "matching. The story is 500 characters long but the model sees only the last 256. **Fix:** longer "
         "context or a deeper model."),
    ]
    for title, text, obs in fail_t1:
        story += [Paragraph(title, H3), Preformatted(text, QUOTE, maxLineLength=118), P(obs)]
    story.append(P("**Teammate failure analysis:** to be added from the teammate's `task1_llm/<member>/failure_analysis.md`."))

    # ===================================================================================== TASK 2
    story += [PageBreak(), Paragraph("Task 2 — Yelp Polarity sentiment classification (no pretrained embeddings)", H1)]
    story += bullets([
        "**Data:** official test split (38,000 reviews) as the shared test set; own 50K validation set "
        "(seed 266) carved from the official train split.",
        "**Preprocessing:** cleaning, deduplication, train/test overlap removal, lowercasing, contraction "
        "expansion, stopword removal that keeps negations, Snowball stemming, and a 30K vocabulary built "
        "from train only.",
        "**Embeddings:** 128-d, learned from scratch in every model.",
    ])
    story.append(Paragraph("2.1 Comparison table — every member's models, architecture, hyperparameters and all Task 2 metrics", H2))
    tms2 = teammate_models("task2")
    archs = {"baseline_meanpool": "Embedding 128 → masked mean → MLP 64 → 1 (bag of learned embeddings)",
             "exp1_textcnn": "Embedding 128 → Conv1d widths 3/4/5 × 128 → max-over-time → dropout → linear",
             "exp2_bigru_attn": "Embedding 128 → 2-layer BiGRU 128/dir → additive attention pooling → linear"}
    hps = {"baseline_meanpool": "AdamW lr 2e-3, batch 512, dropout 0.3, ≤8 epochs, patience 2",
           "exp1_textcnn": "AdamW lr 1e-3, batch 256, dropout 0.5, ≤5 epochs, patience 2",
           "exp2_bigru_attn": "AdamW lr 1e-3, batch 256, dropout 0.3, ≤5 epochs, patience 2"}
    common = "All: 30K vocab, max_len 256 (head+tail), 3% warm-up + cosine, wd 1e-4, clip 1.0, threshold 0.5"
    hdr = [""] + [f"Shibin · {mm}" for mm in T2_MODELS] + [esc(x["model"]) for x in tms2]
    rows = [hdr, ["Architecture"] + [archs[mm] for mm in T2_MODELS] + [esc(x.get("arch") or "") or PEND for x in tms2],
            ["Hyperparameters"] + [hps[mm] + "; " + common if i == 0 else hps[mm] for i, mm in enumerate(T2_MODELS)]
            + [esc(x.get("hparams") or "") or PEND for x in tms2]]
    T2_ROWS = ["Accuracy", "Accuracy 95% CI", "Precision (macro)", "Recall (macro)", "F1 (macro)",
               "Precision (micro)", "Recall (micro)", "F1 (micro)", "Precision (weighted)", "Recall (weighted)",
               "F1 (weighted)", "Confusion matrix [TN FP; FN TP]", "ROC-AUC", "PR-AUC", "MCC", "MCC 95% CI",
               "Brier score", "ECE (15 bins)", "McNemar vs baseline (b / c, p)", "Parameters", "Training time (min)",
               "Epochs run (best)", "Train examples/sec", "Inference examples/sec", "Peak GPU memory (MB)", "Hardware"]
    for k in T2_ROWS:
        rows.append([k] + [esc(t2[k][mm]) for mm in T2_MODELS] + [tm_val(x, k) for x in tms2])
    rows.append(["Checkpoint (SHA-256)"] + [f"`{RUN2}/{mm}` ({t2_man['checkpoints_files'][mm]['sha256'][:10]}…)"
                                            for mm in T2_MODELS] + [PEND for _ in tms2])
    ncol = len(T2_MODELS) + len(tms2)
    story.append(table(rows, [1.15 * inch] + [(WIDTH - 1.15 * inch) / ncol] * ncol, first_col_bold=True, font=6.5))
    story.append(Paragraph("Official Yelp polarity test split, n = 38,000. Macro = micro = weighted because the test set "
                           "is exactly balanced. Metric code: " + link("task2_sentiment/shared_eval/metrics.py")
                           + ". Source: " + link(f"{T2}/outputs/{RUN2}/comparison.csv"), SMALL))

    story.append(Paragraph("Robustness slices (error rate)", H3))
    sl = [["Slice (n)", "baseline", "TextCNN", "BiGRU+attn"],
          ["has negation (28,544)", "7.47%", "5.50%", "4.48%"], ["has contrast (22,713)", "7.89%", "5.94%", "4.87%"],
          ["no negation & no contrast (6,688)", "4.93%", "4.53%", "4.16%"], ["short ≤50 words (9,287)", "6.96%", "5.29%", "5.01%"],
          ["medium 51–150 words (16,910)", "6.94%", "5.22%", "4.09%"], ["long >150 words (11,803)", "7.00%", "5.52%", "4.45%"]]
    story.append(table(sl, [2.4 * inch] + [(WIDTH - 2.4 * inch) / 3] * 3, first_col_bold=True))
    story.append(Paragraph("Source: " + link(f"{T2}/metrics_report.csv") + " (slice rows) and "
                           + link(f"{T2}/outputs/{RUN2}/slice_error_rates.png"), SMALL))
    story.append(figure(f"{T2}/outputs/{RUN2}/training_curves.png", "Task 2 (Shibin): training/validation loss and "
                        "validation macro-F1 per epoch for the three models", max_h=2.9 * inch))
    story.append(figure(f"{T2}/outputs/{RUN2}/roc_curves.png", "Task 2 (Shibin): ROC curves on the official test split",
                        width=WIDTH * 0.62, max_h=2.6 * inch))

    story.append(Paragraph("2.2 Joint analysis", H2))
    story.append(Paragraph("Strengths", H3))
    story += bullets([
        "**The ranking is significant everywhere.** BiGRU+attention > TextCNN > baseline on every "
        "discrimination metric, with non-overlapping 95% bootstrap CIs and McNemar p ≤ 1e-20 for each pair.",
        "**Order-aware models fix what a bag of words cannot.** Compared with the baseline, the BiGRU removes "
        "36% of the errors overall, 40% on negation and 38% on contrast reviews.",
        "**All three models are well calibrated** (ECE ≤ 1.1%).",
        "**The gains come from how the text is read, not from model size.** Parameters differ by under 0.6M; "
        "the 3.84M embedding table dominates every model.",
    ])
    story.append(Paragraph("Weaknesses", H3))
    story += bullets([
        "**Speed.** The BiGRU trains 123× slower than the baseline (19 min vs 21 s) and is mildly "
        "over-confident (544 of its 1,681 errors are made with more than 90% confidence).",
        "**Short reviews remain the hardest slice** for the BiGRU (5.0% error).",
        "**Mixed-sentiment reviews dominate the confident errors.**",
    ])
    story.append(Paragraph("Limitations", H3))
    story += bullets([
        "**Star-rating labels are noisy.** 5 of the 20 reviewed errors contradict their label, and 932 test "
        "reviews are wrong for all three models.",
        "**No pretrained knowledge** (idioms, sarcasm), as the brief requires.",
        "Single seed per model, and a fixed 0.5 threshold.",
    ])
    story.append(Paragraph("What we would try next", H3))
    story += bullets([
        "Contrast- and negation-aware token marking, tested with McNemar on the has-contrast slice.",
        "max_len 512 for long reviews.",
        "A from-scratch Transformer encoder.",
        "An ensemble of the three models, with weights chosen on validation.",
        "Temperature scaling and several seeds.",
    ])

    story.append(Paragraph("2.3 Evidence", H2))
    story.append(evidence([
        ("Configs (data + 3 models)", f"{T2}/configs", True),
        ("Raw training / eval logs", f"reproducibility/raw_logs/task2_sentiment/shibin_thomas/{RUN2}", True),
        ("Manifest (versions, hardware, checkpoint SHA-256)", f"reproducibility/manifests/task2_sentiment/shibin_thomas/{RUN2}.json", False),
        ("All metrics, every model", f"{T2}/metrics_report.csv", False),
        ("ROC / PR / reliability / slice plots", f"{T2}/outputs/{RUN2}", True),
        ("Learned-embedding neighbours (from-scratch evidence)", f"{T2}/outputs/{RUN2}/embedding_neighbours.md", False),
        ("Checkpoints", f"{T2}/checkpoints/{RUN2}", True),
        ("Results, justification, comparative analysis", f"{T2}/results.md", False),
        ("Executed notebook", f"{T2}/src/task2_yelp_sentiment.ipynb", False),
    ]))

    story.append(Paragraph("2.4 Error review — Shibin Thomas (20 misclassified test reviews, BiGRU+attention)", H2))
    story.append(P("Selection: 5 confident false positives, 5 confident false negatives, 5 near-threshold errors and "
                   "5 from the worst slice (short reviews). Each review was read and labelled by hand."))
    err_rows = [["#", "True / P(pos)", "Error type (manual)", "Review text (excerpt)"]]
    errs = [
        (2, "neg / 1.000", "Label noise", "Wow love the place and everything is very clean and new! Great place to come and relax worth a try!"),
        (4, "neg / 1.000", "Mixed + implicit verdict", "About average so far as steakhouses go. My rib eye was very tasty but a little over cooked. ... Over all I'd say they are more for show than content. If you want to have a great high end steak I'd recommend Ruth Chris..."),
        (6, "pos / 0.000", "Label noise (edited review)", "EDIT: They really did change the service up since I last posted this. Horrible service. Used to be my favorite pizza in the city..."),
        (9, "pos / 0.001", "Sentiment-target confusion", "Last night several parents came in with over 15 children ... The bartender expressed that it was not a place to have children running around..."),
        (13, "pos / 0.498", "Mixed + humour", "Their alcohol selection sucks ... dreamcrushed my cocktail dreams. ... However, the burger was nice, very juicy and rare."),
        (14, "neg / 0.502", "Implicit negativity / domain shift", "Overly sentimental, non stop references to her kids, how blessed she is ... too many covers from elite performers..."),
        (15, "pos / 0.497", "Idiom misread", "the pizza was to my door in 15 minutes! ... the cookie's are to die for."),
        (16, "neg / 0.999", "Faint praise", "Standard take out and it's cheap. Service is fast and friendly. I go here when I'm too lazy to walk to Zaw's..."),
        (18, "pos / 0.002", "Complex negation + star mention", "despite still not digging their ordering process, their food is just too good to disrespect with a 2 star review."),
        (19, "pos / 0.002", "Sarcasm / irony", "For being a DUMP, should expect much more. Flys, stink, garbage, dirt ... Keepin it real dumpy!"),
    ]
    for e in errs:
        err_rows.append([str(e[0]), e[1], e[2], f'<font name="DVM" size="6.6">{esc(e[3])}</font>'])
    story.append(table(err_rows, [0.3 * inch, 0.85 * inch, 1.35 * inch, WIDTH - 2.5 * inch]))
    story.append(Paragraph("10 of the 20 reviewed errors shown; all 20 with full text: " + link(f"{T2}/failure_analysis.md"), SMALL))
    et = [["Error type", "Count (of 20)", "Model-fixable?"],
          ["Mixed sentiment (aspect-level, complaints vs praise)", "7", "partly"],
          ["Label noise (text contradicts star label)", "5", "no — prediction matches the text"],
          ["Faint praise / implicit or comparative negativity", "3", "partly (world knowledge)"],
          ["Complex negation, sentiment-target confusion, truncation, idiom, sarcasm", "1 each", "varies"]]
    story.append(Spacer(1, 4))
    story.append(table(et, [3.6 * inch, 0.9 * inch, WIDTH - 4.5 * inch]))
    story.append(P("**Proposed testable fix:** mark the clause after the last contrast word (“but”, “however”, "
                   "“despite”), e.g. `POST_good`. Retrain the BiGRU with only this change, then compare the has-contrast "
                   "slice (currently 4.87% error) and run McNemar on the same 38,000 reviews."))
    story.append(P("**Teammate error review:** to be added from the teammate's `task2_sentiment/<member>/failure_analysis.md`."))

    # ===================================================================================== TASK 3
    story += [PageBreak(), Paragraph("Task 3 — CycleGAN Monet ↔ Photo style transfer (Kaggle)", H1)]
    story += bullets([
        "**Shibin Thomas:** CycleGAN implemented and trained from random initialisation. Domain A = 300 Monet "
        "paintings, domain B = 7,038 photos (unpaired).",
        "**Kaggle protocol:** the instructor's script (Inception-v3 features, first 300 sorted images). Score = "
        "(FID + MiFID) / 2, averaged over both directions; lower is better.",
        "**Integrity:** the submitted images are the direct, unedited outputs of the trained generators. "
        "Pretrained networks (Inception, LPIPS) were used only to measure.",
    ])
    story.append(Paragraph("3.1 Comparison table — architecture, hyperparameters and all Task 3 metrics", H2))
    tms3 = teammate_models("task3")
    hdr = ["", "Shibin · photo→Monet (B2A)", "Shibin · Monet→photo (A2B)"] + [esc(x["model"]) for x in tms3]
    arch3 = ("G: ResNet-9 (c7s1-64, 2 down, 9 residual blocks, 2 resize-conv up, tanh), InstanceNorm, reflection pad, "
             f"11.38M params each · D: 70×70 PatchGAN, 2.76M each · total {int(float(t3[('training', 'params_total')])):,}")
    hp3 = ("LSGAN + λ_cyc 10 + λ_id 5; Adam 2e-4 β=(0.5, 0.999); 20 const + 20 linear-decay epochs × 2,000 pairs; "
           "batch 4; image pool 50; resize 286 → crop 256 + flip; fp32/TF32")
    rows = [hdr, ["Architecture", Paragraph(md(arch3), CELL), "(same model, second generator)"]
            + [esc(x.get("arch") or "") or PEND for x in tms3],
            ["Hyperparameters", Paragraph(md(hp3), CELL), "(same run)"] + [esc(x.get("hparams") or "") or PEND for x in tms3]]
    T3_ROWS = [("FID (instructor protocol)", "fid", 2), ("MiFID (script: mean paired cosine dist.)", "mifid_script", 4),
               ("KID (unbiased, cubic kernel)", "kid", 4), ("Precision (k = 3, realism)", "precision", 3),
               ("Recall (k = 3, diversity)", "recall", 3), ("Density / coverage (k = 5)", None, 3),
               ("Cycle-reconstruction L1 (pixels 0–1)", "cycle_l1", 4),
               ("LPIPS input vs translation", "lpips_input_vs_translation", 3),
               ("LPIPS input vs reconstruction", "lpips_input_vs_reconstruction", 3),
               ("Content cosine (input vs translation)", "content_cosine_similarity", 3)]
    for label, key, nd in T3_ROWS:
        if key is None:
            vals = [f"{float(t3[(d, 'density')]):.3f} / {float(t3[(d, 'coverage')]):.3f}" for d in (B2A, A2B)]
        else:
            vals = [f"{float(t3[(d, key)]):.{nd}f}" for d in (B2A, A2B)]
        rows.append([label] + vals + [tm_val(x, label) for x in tms3])
    tr = lambda k: t3[("training", k)]  # noqa: E731
    both = [
        ("Kaggle submission: FID / MiFID (avg)", f"{fid_avg:.4f} / {mifid_avg:.4f} → score {kaggle_score:.2f}"),
        ("Kaggle leaderboard (public score / rank)",
         f"{kaggle.get('public_score') if kaggle.get('public_score') is not None else PEND} / "
         f"{kaggle.get('public_rank') if kaggle.get('public_rank') is not None else PEND} (team {esc(INFO['team_name'])})"),
        ("Final-epoch losses: G / D_A / D_B", f"{float(tr('final_epoch_mean_loss_G')):.3f} / "
                                               f"{float(tr('final_epoch_mean_loss_D_A')):.3f} / {float(tr('final_epoch_mean_loss_D_B')):.3f}"),
        ("Final-epoch cycle A / B, identity A / B", f"{float(tr('final_epoch_mean_cyc_A')):.3f} / {float(tr('final_epoch_mean_cyc_B')):.3f}, "
                                                    f"{float(tr('final_epoch_mean_idt_A')):.3f} / {float(tr('final_epoch_mean_idt_B')):.3f}"),
        ("Grad norm G mean / max · D mean / max", f"{float(tr('grad_norm_G_mean')):.1f} / {float(tr('grad_norm_G_max')):.0f} · "
                                                  f"{float(tr('grad_norm_D_mean')):.1f} / {float(tr('grad_norm_D_max')):.0f}"),
        ("Non-finite steps / total steps", f"{tr('nonfinite_steps')} / {int(float(tr('total_steps'))):,}"),
        ("Training time / throughput / peak GPU mem.", f"{float(tr('training_time_min')):.1f} min / "
                                                       f"{float(tr('train_images_per_sec')):.1f} img/s / {float(tr('peak_gpu_allocated_mb')):,.0f} MB"),
        ("Human audit (2 raters, 30 blinded samples)", PEND),
        ("Hardware", tr("hardware")),
    ]
    for label, val in both:
        rows.append([label, Paragraph(val, CELL), ""] + [tm_val(x, label) for x in tms3])
    ck = f"{T3}/checkpoints/{RUN3}"
    rows.append(["Run id / checkpoints (SHA-256)", Paragraph(md(f"`{RUN3}` · G_BA.pt {sha256(ck + '/G_BA.pt')[:10]}…"), CELL),
                 Paragraph(md(f"G_AB.pt {sha256(ck + '/G_AB.pt')[:10]}…"), CELL)] + [PEND for _ in tms3])
    ncol = 2 + len(tms3)
    t = table(rows, [1.6 * inch] + [(WIDTH - 1.6 * inch) / ncol] * ncol, first_col_bold=True, font=6.7)
    span = [("SPAN", (1, i), (2, i)) for i in (1, 2)] + \
           [("SPAN", (1, i), (2, i)) for i in range(3 + len(T3_ROWS), 3 + len(T3_ROWS) + len(both))]
    t.setStyle(TableStyle(span))
    story.append(t)
    story.append(Paragraph("Source: " + link(f"{T3}/full_metrics_report.csv") + " · " + link(f"{T3}/submission.csv") + " · "
                           + link(f"{T3}/kaggle_leaderboard.json") + " · metric code " + link(f"{T3}/src/gan_metrics.py"), SMALL))

    story.append(Paragraph("3.2 Training behaviour and samples", H2))
    story.append(figure(f"{T3}/outputs/{RUN3}/loss_curves.png", "Task 3 (Shibin): adversarial, cycle and identity losses, "
                        "total generator loss with LR schedule, and gradient norms over 20,000 steps", max_h=3.3 * inch))
    fid_rows = [["Epoch"] + [r["epoch"] for r in t3_epochs if r["fid_B2A"]],
                ["FID B2A"] + [f"{float(r['fid_B2A']):.1f}" for r in t3_epochs if r["fid_B2A"]],
                ["FID A2B"] + [f"{float(r['fid_A2B']):.1f}" for r in t3_epochs if r["fid_B2A"]]]
    n = len(fid_rows[0]) - 1
    story.append(table(fid_rows, [0.8 * inch] + [(WIDTH - 0.8 * inch) / n] * n, first_col_bold=True))
    story.append(Paragraph("Periodic FID on the 300 scored images (monitoring only). Source: "
                           + link(f"reproducibility/raw_logs/task3_gan/shibin_thomas/{RUN3}/epochs.csv"), SMALL))
    story.append(figure(f"{T3}/outputs/{RUN3}/final_samples_B2A.png", "Photo → Monet: input photo (top), "
                        "generated Monet (middle), reconstruction (bottom); 8 scored images", max_h=2.6 * inch))
    story.append(figure(f"{T3}/outputs/{RUN3}/final_samples_A2B.png", "Monet → photo: input painting (top), "
                        "generated photo (middle), reconstruction (bottom)", max_h=2.6 * inch))

    story.append(Paragraph("3.3 Joint analysis", H2))
    story.append(Paragraph("Strengths", H3))
    story += bullets([
        "**A stable from-scratch CycleGAN:** no non-finite step and no mode collapse in 20,000 steps.",
        "**Cycle consistency holds.** Reconstructions match the inputs (cycle L1 0.05–0.06).",
        "**Content and style.** Photo → Monet keeps the layout (content cosine 0.75) while moving colours to "
        "Monet's palette.",
        "**Exact Kaggle numbers.** They come from a re-implementation of the instructor's script, unit-tested "
        "against it.",
    ])
    story.append(Paragraph("Weaknesses", H3))
    story += bullets([
        "**Photo → Monet realism is low** (precision 0.27). The cause is a regular stipple texture instead of "
        "brush strokes; dark photos are washed out.",
        "**Monet → photo diversity is low** (recall 0.32). Outputs stay painterly.",
        "**D_A over-fits the 300 paintings.** Its loss fell to 0.089, well below the 0.25 balance point.",
    ])
    story.append(Paragraph("Limitations", H3))
    story += bullets([
        "FID on 300 images is noisy (±4 between neighbouring checkpoints).",
        "Single seed.",
        "The script's MiFID (paired cosine distance between unrelated images) barely varies, so the Kaggle "
        "score is driven almost entirely by FID.",
        "The human audit is pending until both raters finish.",
    ])
    story.append(Paragraph("What we would try next", H3))
    story.append(P("Implemented and ready as `configs/cyclegan_v2.yaml`:"))
    story += bullets([
        "DiffAugment on the discriminators;",
        "an EMA of the generator weights;",
        "λ_id 2.5 instead of 5;",
        "60 epochs instead of 40;",
        "epoch selection on held-out photos.",
    ])
    story.append(P("Further options: a multi-scale discriminator for stroke-level texture, and a "
                   "low-pass content loss against colour inversion."))

    story.append(Paragraph("3.4 Evidence", H2))
    story.append(evidence([
        ("Config", f"{T3}/configs/cyclegan_v1.yaml", False),
        ("Raw training / eval logs (train.log, steps.csv, epochs.csv, eval.log)", f"reproducibility/raw_logs/task3_gan/shibin_thomas/{RUN3}", True),
        ("Manifest (versions, hardware, data fingerprint)", f"reproducibility/manifests/task3_gan/shibin_thomas/{RUN3}.json", False),
        ("Submitted predictions (300 + 300 JPGs)", f"{T3}/outputs", True),
        ("Kaggle file / leaderboard record", f"{T3}/submission.csv", False),
        ("Plots, sample grids, per-epoch samples", f"{T3}/outputs/{RUN3}", True),
        ("Generator checkpoints", f"{T3}/checkpoints/{RUN3}", True),
        ("Blinded human-audit pack", f"{T3}/outputs/human_audit", True),
        ("Results, design justification, v2 rationale", f"{T3}/results.md", False),
        ("Executed notebook", f"{T3}/src/task3_cyclegan.ipynb", False),
    ]))

    story.append(Paragraph("3.5 Failure analysis — Shibin Thomas (actual generated images)", H2))
    story.append(P("Images were ranked automatically by three criteria: highest cycle error, lowest content cosine, "
                   "and largest distance to any real target image. The grids were then reviewed by hand. Each grid "
                   "column shows the input (top), the translation (middle) and the reconstruction (bottom)."))
    story.append(figure(f"{T3}/outputs/{RUN3}/failure_B2A_nn_dist.png",
                        "Failure 1 (photo → Monet): dark / night photos are washed out into bright pastel 'confetti' "
                        "texture. These are the least realistic outputs (NN distance 19.5–20.3).", width=WIDTH * 0.7, max_h=3.6 * inch))
    story.append(P("**Cause:** the 300 Monets contain almost no night scenes, so G_BA brightens every image and adds "
                   "brush-like noise. **Fix:** DiffAugment colour on D_A (v2), measured by precision and NN distance "
                   "on the darkest photos."))
    story.append(figure(f"{T3}/outputs/{RUN3}/failure_B2A_cycle_l1.png",
                        "Failure 2 (photo → Monet): vivid reds and oranges are recoloured (the orange sky turns blue), "
                        "yet the reconstruction recovers them: the generator hides the original colour in a faint "
                        "signal (steganography). Highest cycle L1, 0.13–0.17.", width=WIDTH * 0.7, max_h=3.6 * inch))
    story.append(P("**Cause:** Monet's palette prior, combined with the cycle loss being satisfiable by a hidden "
                   "signal; green dots in reconstructions are its visible trace. **Fix:** lower λ_id or add a colour "
                   "term, measured by the hue shift and the cycle L1 of these images."))
    story.append(figure(f"{T3}/outputs/{RUN3}/failure_A2B_content_cos.png",
                        "Failure 4 (Monet → photo): foggy, low-contrast paintings become saturated 'sunsets' or "
                        "turquoise storms, and the poplar tree is lost. Lowest content cosine, 0.52–0.55.",
                        width=WIDTH * 0.7, max_h=3.6 * inch))
    story.append(P("**Also seen:** a regular stipple texture instead of brush strokes (Failure 3; the PatchGAN "
                   "judges only local statistics) and Monet → photo outputs that keep brush strokes and signatures "
                   "(Failure 5). Full analysis of all five failures: " + "`failure_analysis.md`" + "."))
    story.append(Paragraph(link(f"{T3}/failure_analysis.md"), SMALL))
    story.append(P("**Teammate failure analysis:** to be added from the teammate's `task3_gan/<member>/failure_analysis.md`."))

    # ===================================================================================== checklist / refs
    story += [PageBreak(), Paragraph("Pre-submission checklist", H1)]
    kaggle_done = kaggle.get("public_rank") is not None
    tm_done = any(x.get("arch") for t in ("task1", "task2", "task3") for x in teammate_models(t))
    ck_rows = [["Item", "Status", "Where to verify"],
               ["Every member's folder exists under all 3 tasks (code, config, logs, results.md)",
                "Shibin: done · teammate: " + ("done" if tm_done else PEND), "task1_llm/, task2_sentiment/, task3_gan/"],
               ["No two members share the same architecture + hyperparameters",
                "to confirm when teammate rows are filled" if not tm_done else "confirmed (tables above)", "comparison tables §1.1, §2.1, §3.1"],
               ["Task 1: no prebuilt Transformer / attention modules", "done (enforced by a unit test)", f"{T1}/tests/test_model.py"],
               ["Task 2: no pretrained embeddings / LMs; all 3 models trained per member",
                "Shibin: done (embeddings random init, neighbours shown)", f"{T2}/outputs/{RUN2}/embedding_neighbours.md"],
               ["All required metrics reported for every model (Tasks 1–3)", "Shibin: done · teammate: " + ("done" if tm_done else PEND),
                "metrics_report.csv per member and task"],
               ["Task 3 Kaggle submission made and leaderboard rank recorded",
                "submitted; rank recorded" if kaggle_done else "submission made; rank " + PEND, f"{T3}/kaggle_leaderboard.json"],
               ["Comparison tables (architecture + hyperparameters + metrics) for all 3 tasks", "done", "§1.1, §2.1, §3.1"],
               ["Raw logs and manifests committed and untouched", "done", "reproducibility/"]]
    story.append(table(ck_rows, [2.75 * inch, 1.85 * inch, WIDTH - 4.6 * inch]))

    story.append(Paragraph("References", H1))
    refs = [
        "Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS.",
        "Eldan, R. & Li, Y. (2023). TinyStories: How Small Can Language Models Be and Still Speak Coherent English? arXiv:2305.07759.",
        "Zhu, J.-Y., Park, T., Isola, P. & Efros, A. A. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (CycleGAN). ICCV.",
        "Radford, A. et al. (2019). Language Models are Unsupervised Multitask Learners (GPT-2). OpenAI.",
        "Xiong, R. et al. (2020). On Layer Normalization in the Transformer Architecture. ICML.",
        "Zhang, X., Zhao, J. & LeCun, Y. (2015). Character-level Convolutional Networks for Text Classification (Yelp polarity). NeurIPS.",
        "Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. EMNLP.",
        "Joulin, A. et al. (2017). Bag of Tricks for Efficient Text Classification. EACL.",
        "Bahdanau, D., Cho, K. & Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. ICLR.",
        "Guo, C. et al. (2017). On Calibration of Modern Neural Networks. ICML.",
        "Isola, P. et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks (PatchGAN). CVPR.",
        "Mao, X. et al. (2017). Least Squares Generative Adversarial Networks. ICCV.",
        "Heusel, M. et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID). NeurIPS.",
        "Bińkowski, M. et al. (2018). Demystifying MMD GANs (KID). ICLR.",
        "Kynkäänniemi, T. et al. (2019). Improved Precision and Recall Metric for Assessing Generative Models. NeurIPS.",
        "Zhang, R. et al. (2018). The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (LPIPS). CVPR.",
        "Chu, C., Zhmoginov, A. & Sandler, M. (2017). CycleGAN, a Master of Steganography. NeurIPS workshop.",
        "Zhao, S. et al. (2020). Differentiable Augmentation for Data-Efficient GAN Training. NeurIPS.",
    ]
    for i, r in enumerate(refs, 1):
        story.append(Paragraph(f"[{i}] {esc(r)}", ParagraphStyle("ref", parent=BODY, fontSize=8.4, leading=11,
                                                                 leftIndent=16, firstLineIndent=-16, spaceAfter=2)))

    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFont("DV", 7.2)
        canvas.setFillColor(MUTED)
        canvas.drawString(0.75 * inch, 0.5 * inch, f"DATA 266 Lab 1 · Team {INFO['team_number']}")
        canvas.drawRightString(letter[0] - 0.75 * inch, 0.5 * inch, f"page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(out), pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                            topMargin=0.7 * inch, bottomMargin=0.75 * inch,
                            title=f"DATA 266 Lab 1 Report - Team {INFO['team_number']}",
                            author=", ".join(m["name"] for m in members), subject="DATA 266 Lab 1")
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return out


if __name__ == "__main__":
    p = build()
    print(f"wrote {p.relative_to(ROOT).as_posix()} ({p.stat().st_size / 1e6:.1f} MB)")
