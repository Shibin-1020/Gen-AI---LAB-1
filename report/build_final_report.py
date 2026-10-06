"""Final team report: Denisha's version of the report, unchanged, with Shibin Thomas's matching
supplementary pages inserted next to Denisha's supplementary pages.

    python report/build_final_report.py

Input:  report/sources/DATA266_Lab1_Report_Team_09_denisha_version.pdf (all 30 pages kept byte-for-byte)
Output: report/DATA266_Lab1_Report_Team_09.pdf

Shibin's pages use the same two page designs as Denisha's supplementary pages. All numbers come from
Shibin's committed result files (metrics_report.csv, comparison.csv, full_metrics_report.csv, rater1.csv).
"""
from __future__ import annotations

import csv
import io
from pathlib import Path

from PIL import Image as PILImage
from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "report/sources/DATA266_Lab1_Report_Team_09_denisha_version.pdf"
OUT = ROOT / "report/DATA266_Lab1_Report_Team_09.pdf"
T1, T2, T3 = "task1_llm/shibin_thomas", "task2_sentiment/shibin_thomas", "task3_gan/shibin_thomas"
RUN1, RUN2, RUN3 = "gpt_char_v1_20261001-153007", "yelp3_20261001-182849", "cyclegan_v1_20261001-201801"
W = letter[0] - 1.5 * inch

NAVY, NAVY_ROW, GREY = colors.Color(32 / 255, 56 / 255, 100 / 255), colors.Color(242 / 255, 245 / 255, 249 / 255), \
    colors.Color(0.5, 0.5, 0.5)
LBLUE, GRID_B = colors.Color(217 / 255, 234 / 255, 247 / 255), colors.Color(119 / 255, 119 / 255, 119 / 255)

# style A (Helvetica, navy header) and style B (Times, light-blue header), as in Denisha's pages
A_T = ParagraphStyle("at", fontName="Helvetica-Bold", fontSize=16, leading=20, spaceAfter=8)
A_B = ParagraphStyle("ab", fontName="Helvetica", fontSize=9, leading=11.5, spaceAfter=6)
A_C = ParagraphStyle("ac", fontName="Helvetica", fontSize=7.4, leading=9)
A_H = ParagraphStyle("ah", parent=A_C, textColor=colors.white)
B_T = ParagraphStyle("bt", fontName="Times-Bold", fontSize=18, leading=22, spaceAfter=8)
B_S = ParagraphStyle("bs", fontName="Times-Bold", fontSize=13, leading=16, spaceBefore=8, spaceAfter=6)
B_B = ParagraphStyle("bb", fontName="Times-Roman", fontSize=10, leading=12.5, spaceAfter=6)
B_C = ParagraphStyle("bc", fontName="Times-Roman", fontSize=8, leading=9.6)
B_N = ParagraphStyle("bn", parent=B_B, fontSize=8.6, leading=10.5, textColor=colors.Color(0.35, 0.35, 0.35))


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def tA(rows, widths):
    data = [[Paragraph(esc(c), A_H if i == 0 else A_C) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("BOX", (0, 0), (-1, -1), 0.5, colors.Color(0.2, 0.2, 0.2)),
          ("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.Color(0.84, 0.84, 0.84)), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    st += [("BACKGROUND", (0, i), (-1, i), NAVY_ROW) for i in range(2, len(rows), 2)]
    t.setStyle(TableStyle(st))
    return t


def tB(rows, widths):
    data = [[Paragraph(esc(c), B_C) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), LBLUE), ("GRID", (0, 0), (-1, -1), 0.5, GRID_B),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 4),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    return t


def read_csv(p):
    with open(ROOT / p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


t1 = {r["metric"] + ("" if r["metric"] != "Total training time" else f" ({r['notes']})"): r["value"]
      for r in read_csv(f"{T1}/metrics_report.csv")}
t2 = {r["metric"]: r for r in read_csv(f"{T2}/outputs/{RUN2}/comparison.csv")}
t3 = {(r["section"], r["metric"]): r["value"] for r in read_csv(f"{T3}/full_metrics_report.csv")}
B2A, A2B = "B2A (photo->monet)", "A2B (monet->photo)"
f1 = lambda k, nd=4: f"{float(t1[k]):.{nd}f}"  # noqa: E731
f3 = lambda d, k, nd=3: f"{float(t3[(d, k)]):.{nd}f}"  # noqa: E731
FID, MIFID = float(t3[("Kaggle submission (avg of both directions)", "FID")]), \
    float(t3[("Kaggle submission (avg of both directions)", "MiFID")])
key = {r["sample_id"]: r["direction"] for r in read_csv(f"{T3}/outputs/human_audit/audit_key.csv")}
r1 = read_csv(f"{T3}/outputs/human_audit/rater1.csv")
r2 = [r for r in read_csv(f"{T3}/outputs/human_audit/rater2.csv") if r.get("style", "").strip()]


def means(rows, d=None):
    s = [r for r in rows if d is None or key[r["sample_id"]] == d]
    return [sum(int(r[c]) for r in s) / len(s) for c in ("style", "content", "artifacts")]


# ------------------------------------------------------------------------------------------- page groups
def contribution():
    return [Paragraph("Shibin Thomas - Contribution Overview", A_T),
            Paragraph("This contribution page complements Denisha's contribution overview. It records Shibin Thomas's "
                      "independent work under the same Lab 1 requirements; Denisha's pages are retained unchanged.", A_B),
            tA([["Area", "Shibin contribution"],
                ["Task 1", "Cleaned TinyStories and drew his own 100,000/10,000 split. Implemented a character-level "
                           "decoder-only GPT from scratch (hand-written multi-head causal attention, LayerNorm, FFN and "
                           "residual blocks, checked by unit tests) and trained it for 10 epochs. Produced every required "
                           "metric, 40 generated samples, loss and gradient curves and the failure analysis."],
                ["Task 2", "Built the preprocessing pipeline (deduplication, train-test overlap removal, negation-"
                           "preserving stopword removal, stemming) and trained a mean-pooling baseline, a TextCNN and a "
                           "BiGRU with attention from scratch. Evaluated on the official test split with bootstrap "
                           "confidence intervals, calibration, slice analysis and McNemar tests, and reviewed 20 errors."],
                ["Task 3", "Implemented and trained a CycleGAN from scratch (ResNet-9 generators with resize-convolution, "
                           "70x70 PatchGAN discriminators). Re-implemented and unit-tested the instructor's FID/MiFID "
                           "protocol, added KID, precision/recall, density/coverage and LPIPS, wrote the visual failure "
                           "analysis and prepared an improved second configuration."],
                ["Shared infrastructure", "Set up the shared repository layout, the shared TinyStories download, "
                                          "the shared metric modules for Tasks 1 and 2, run manifests and smoke tests."],
                ["Reproducibility", "Maintained the shibin_thomas folders with configurations, notebooks, checkpoints, "
                                    "raw logs, manifests, metrics reports, results and failure-analysis files."],
                ["Kaggle", f"Submitted his model's evaluator output (FID {FID:.2f}, MiFID {MIFID:.4f}; score "
                           f"{(FID + MIFID) / 2:.2f}) under team PairProgramming_Team_09."],
                ["Human audit", "Built the blinded 30-panel audit pack and the rating form; rated his own model as "
                                "rater 1 and rated Denisha's 30 images as rater 2."],
                ["Shared team evidence", "Both members use the shared repository: "
                                         "https://github.com/Shibin-1020/Gen-AI---LAB-1 (branch shibin-lab1)"]],
               [1.35 * inch, W - 1.35 * inch]),
            Spacer(1, 10),
            Paragraph("Shibin evidence is stored under task1_llm/shibin_thomas, task2_sentiment/shibin_thomas and "
                      "task3_gan/shibin_thomas, with raw logs and manifests under reproducibility/.", A_B)]


def task1():
    a = [Paragraph("Shibin Thomas - Task 1 Detailed Findings", A_T),
         Paragraph("Model and data. Shibin trained a decoder-only character Transformer on 100,000 TinyStories training "
                   "stories and 10,000 validation stories (seed 266) drawn from 1,989,367 cleaned stories. The model uses "
                   "learned token and position embeddings, six pre-LayerNorm blocks with eight-head causal "
                   "self-attention, model width 320, feed-forward width 1,280 with GELU, dropout 0.1 and tied input/output "
                   "embeddings. The context length is 256 characters.", A_B),
         tA([["Configuration", "Value"],
             ["Optimizer / schedule", "AdamW; learning rate 6e-4; weight decay 0.1; betas (0.9, 0.95); 2% linear warm-up "
                                      "(about 1,100 steps); cosine decay to 6e-5"],
             ["Batch / training", "Batch 64 x 256 characters; 10 epochs (55,030 steps); gradient clipping 1.0; bf16 "
                                  "autocast; RTX 5090"],
             ["Results", f"Train CE {f1('Training cross-entropy loss')}; validation CE "
                         f"{f1('Validation cross-entropy loss')}; perplexity {f1('Perplexity (validation)')}; BPC "
                         f"{f1('Bits-per-character (validation)')}; next-character accuracy "
                         f"{float(t1['Top-1 next-character accuracy (validation)']) * 100:.2f}%"],
             ["Stability", f"Mean pre-clip gradient norm {f1('Gradient norm (mean, pre-clip)', 3)}; max "
                           f"{f1('Gradient norm (max, pre-clip)', 2)}; clipped steps "
                           f"{float(t1['Fraction of steps clipped']) * 100:.2f}% (all during warm-up); loss spikes 0; NaN steps 0"],
             ["Efficiency", f"{int(float(t1['Parameter count'])):,} parameters; "
                            f"{float(t1['Training tokens/sec']):,.0f} training characters/s; "
                            f"{float(t1['Total training time (minutes)']):.1f} minutes on RTX 5090"],
             ["Artifacts", "metrics_report.csv, samples.md, samples.json, loss_curves.png, grad_norm.png, lr_schedule.png, "
                           "best_model.pt, raw logs and manifest"]],
            [1.35 * inch, W - 1.35 * inch]),
         Spacer(1, 10),
         Paragraph("Analysis. Validation loss fell in every epoch (0.715 to 0.571) with a generalisation gap of only "
                   "0.012 nats, so the model is data-rich and not over-fitting. Greedy decoding repeats phrases, rare "
                   "words are spelled inconsistently, and stories lose coherence beyond the 256-character context; "
                   "temperature sampling at 0.8 gives varied text (Distinct-3 0.93).", A_B)]
    b = [PageBreak(), Paragraph("Shibin Thomas — Task 1 Full Configuration and Results", B_T),
         Paragraph("This section is an expanded record of Shibin's independent implementation, placed after Denisha's "
                   "Task 1 pages so the two members' work remains distinguishable and traceable.", B_B),
         Paragraph("Architecture and training configuration", B_S),
         tB([["Item", "Shibin configuration"],
             ["Data", "TinyStories; non-ASCII and very short stories removed (1,989,367 of 2,119,719 kept); 100,000 "
                      "training and 10,000 validation stories, seed 266"],
             ["Tokenisation", "Character vocabulary built from the training split: 88 characters + <eos> + <unk> = 90; "
                              "sequence length 256; non-overlapping windows with a random offset each epoch"],
             ["Architecture", "Decoder-only GPT-style Transformer; 6 layers; 8 attention heads (head dim 40); model "
                              "dimension 320; feed-forward dimension 1,280; pre-LayerNorm; tied embeddings; dropout 0.10"],
             ["Optimisation", "AdamW; learning rate 6e-4; weight decay 0.1 (matrices only); betas (0.9, 0.95); 2% warm-up; "
                              "cosine decay to 10% of peak LR; GPT-2 initialisation"],
             ["Training", "Batch size 64; 10 epochs; gradient clipping 1.0; bf16 autocast with fp32 LayerNorm/softmax/loss; "
                          "RTX 5090"],
             ["Reproducibility", "Config, raw logs, manifest with checkpoint SHA-256, best checkpoint, loss curves, samples, "
                                 "metrics and failure analysis saved under shibin_thomas"]],
            [1.4 * inch, W - 1.4 * inch]),
         Paragraph("Shibin quantitative results", B_S),
         tB([["Metric", "Value"],
             ["Final training cross-entropy (eval mode)", f1("Training cross-entropy loss", 6)],
             ["Final validation cross-entropy", f1("Validation cross-entropy loss", 6)],
             ["Validation perplexity", f1("Perplexity (validation)", 6)],
             ["Validation bits per character", f1("Bits-per-character (validation)", 6)],
             ["Generalisation gap (val - train CE)", f1("Generalization gap", 6)],
             ["Top-1 next-character accuracy", f1("Top-1 next-character accuracy (validation)", 6)],
             ["Distinct-1 / 2 / 3 (T = 0.8)", f"{f1('Distinct-1')} / {f1('Distinct-2')} / {f1('Distinct-3')}"],
             ["Repeated 4-gram rate (sampled / greedy)", f"{f1('Repeated 4-gram rate')} / {f1('Repeated 4-gram rate (greedy)')}"],
             ["Trainable parameters", f"{int(float(t1['Parameter count'])):,}"],
             ["Training time", f"{float(t1['Total training time (seconds, wall clock incl. per-epoch eval)']):,.1f} seconds"],
             ["Training throughput", f"{float(t1['Training tokens/sec']):,.0f} characters/second"],
             ["Generation throughput", f"{float(t1['Generation tokens/sec']):.1f} characters/second"],
             ["Peak GPU memory", f"{float(t1['Peak GPU memory (allocated)']):,.0f} MB"],
             ["Device and numerical check", "CUDA/RTX 5090; NaN/Inf steps 0; loss spikes 0"]],
            [2.6 * inch, W - 2.6 * inch]),
         Spacer(1, 8),
         Paragraph(f"Evidence locations: {T1}/configs/gpt_char_v1.yaml; metrics_report.csv; outputs/{RUN1}/ (loss_curves.png, "
                   f"grad_norm.png, samples.md); checkpoints/{RUN1}/best_model.pt; results.md; failure_analysis.md; "
                   f"reproducibility/raw_logs/ and reproducibility/manifests/.", B_N)]
    c = [PageBreak(), Paragraph("Shibin Thomas — Task 1 Analysis and Failure Findings", B_T),
         Paragraph("The model learned the character distribution well: validation cross-entropy is 0.571 nats per "
                   "character and next-character accuracy is 81.73%. A validation perplexity of 1.77 means the model is, "
                   "on average, choosing between fewer than two plausible next characters.", B_B),
         Paragraph("Failure analysis", B_S),
         tB([["Observed case", "Interpretation and limitation"],
             ["Greedy repetition", "“The tree was very high and the tree was very high” repeats 13 times "
                                   "(repeated 4-gram rate 0.61 for that sample). Argmax decoding locks into a "
                                   "high-probability loop; sampling at T = 0.8 lowers the run-level rate from 0.137 to 0.007."],
             ["Rare-word spelling instability", "The rare word “lute” is spelled lute, lutter, luter and lutte "
                                                "in one sample: a character model rebuilds rare words from frequent "
                                                "letter patterns each time."],
             ["Long-range coherence", "A 500-character continuation jumps from a zoo to a diamond and a bucket and "
                                      "introduces new friends (Jack, Tom) because only the last 256 characters are visible."],
             ["Numerical stability", "No NaN/Inf steps and no loss spikes in 55,030 steps; all 738 clipped steps "
                                     "occurred during warm-up."]],
            [1.6 * inch, W - 1.6 * inch]),
         Paragraph("Submission coverage", B_S),
         Paragraph("Shibin's Task 1 contribution contains the executed notebook, source code, unit tests, configuration, "
                   "best checkpoint, metrics report, per-step and per-epoch logs, loss, gradient and learning-rate curves, "
                   "40 generated samples, vocabulary, run manifest, results narrative and failure analysis.", B_B)]
    return a + b + c


def task2():
    names = {"baseline_meanpool": "Baseline", "exp1_textcnn": "TextCNN", "exp2_bigru_attn": "BiGRU + attention"}
    arch = {"baseline_meanpool": "128-d embedding; masked mean pool; 64-d MLP; dropout 0.3",
            "exp1_textcnn": "128-d embedding; 128 filters each of widths 3/4/5; max pool; dropout 0.5",
            "exp2_bigru_attn": "128-d embedding; 2-layer BiGRU, 128 per direction; additive attention; dropout 0.3"}
    M = list(names)
    v = lambda k, m: t2[k][m]  # noqa: E731
    a = [Paragraph("Shibin Thomas - Task 2 Detailed Findings", A_T),
         Paragraph("Preprocessing and protocol. Shibin evaluated on the official Yelp Polarity test split (38,000 reviews) "
                   "and used a 50,000-review validation split carved from the training data (seed 266). Duplicates and "
                   "train-test overlaps were removed, contractions expanded, stopwords removed while keeping negation and "
                   "contrast words, and Snowball stemming applied; the 30,000-token vocabulary was built from training "
                   "data only and embeddings were learned from random initialisation.", A_B),
         tA([["Model", "Architecture / settings", "Accuracy", "Macro F1", "ROC-AUC", "MCC"]] +
            [[names[m], arch[m], f"{float(v('Accuracy', m)) * 100:.2f}%", v("F1 (macro)", m), v("ROC-AUC", m), v("MCC", m)]
             for m in M], [1.0 * inch, W - 4.0 * inch, 0.75 * inch, 0.75 * inch, 0.75 * inch, 0.75 * inch]),
         Spacer(1, 8),
         tA([["Additional evidence", "Shibin result"],
             ["Best validation macro-F1", "0.9522 for BiGRU + attention (epoch 4)"],
             ["Training time", f"{float(v('Training time (min)', M[0])) * 60:.0f} s baseline; "
                               f"{float(v('Training time (min)', M[1])) * 60:.0f} s TextCNN; "
                               f"{float(v('Training time (min)', M[2])):.1f} min BiGRU (RTX 5090)"],
             ["Gradient checks", "0 NaN/Inf steps for all models; max pre-clip gradient norm 3.52"],
             ["Significance", f"McNemar vs baseline: TextCNN {v('McNemar vs baseline (b / c, p)', M[1])}; "
                              f"BiGRU {v('McNemar vs baseline (b / c, p)', M[2])}"],
             ["Error analysis", "20 cases: 5 confident false positives, 5 confident false negatives, 5 near-threshold, "
                                "5 from the short-review slice"],
             ["Interpretation", "Order-aware models remove most errors on negation (-40%) and contrast (-38%) reviews; "
                                "short and mixed-sentiment reviews remain the hardest."]],
            [1.5 * inch, W - 1.5 * inch]),
         Spacer(1, 8),
         Paragraph("Evidence includes metrics_report.csv, comparison.csv, mcnemar.json, predictions_test.npz, ROC/PR/"
                   "reliability/slice plots, embedding_neighbours.md, three checkpoints, results.md, failure_analysis.md, "
                   "raw logs and the run manifest.", A_B)]
    b = [PageBreak(), Paragraph("Shibin Thomas — Task 2 Full Model Comparison", B_T),
         Paragraph("Shibin independently trained the three required sentiment models with an identical data protocol and "
                   "embedding size, so differences come from the architecture.", B_B),
         Paragraph("Shared protocol and model settings", B_S),
         tB([["Item", "Setting"],
             ["Data processing", "Official test split (38,000); validation 50,000 (seed 266); vocabulary 30,000 (min "
                                 "frequency 3); maximum length 256 with head (128) + tail (128) truncation"],
             ["Training", "AdamW; weight decay 1e-4 (not on embeddings or biases); 3% warm-up then cosine decay to 5%; "
                          "gradient clipping 1.0; early stopping on validation macro-F1 (patience 2); threshold 0.5"],
             ["Baseline", "Masked mean of embeddings, MLP 64, dropout 0.3; lr 2e-3; batch 512; up to 8 epochs"],
             ["TextCNN", "Filter widths 3, 4, 5 with 128 filters each; max-over-time pooling; dropout 0.5; lr 1e-3; "
                         "batch 256; up to 5 epochs"],
             ["BiGRU + attention", "Two bidirectional GRU layers (128 per direction), additive attention pooling, dropout "
                                   "0.3; lr 1e-3; batch 256; up to 5 epochs"]],
            [1.4 * inch, W - 1.4 * inch]),
         Paragraph("Full test-set results", B_S),
         tB([["Model", "Accuracy", "Macro F1", "ROC AUC", "MCC", "Brier", "ECE", "Params", "Time (min)"]] +
            [[names[m], v("Accuracy", m), v("F1 (macro)", m), v("ROC-AUC", m), v("MCC", m), v("Brier score", m),
              v("ECE (15 bins)", m), v("Parameters", m), f"{float(v('Training time (min)', m)):.2f}"] for m in M],
            [1.2 * inch] + [(W - 1.2 * inch) / 8] * 8),
         Spacer(1, 6),
         Paragraph("All three runs had zero NaN/Inf steps on the RTX 5090. The BiGRU with attention is Shibin's strongest "
                   "model on every discrimination metric (95% CI of accuracy [0.9537, 0.9577]); the baseline is the best "
                   "calibrated (ECE 0.0045).", B_B)]
    c = [PageBreak(), Paragraph("Shibin Thomas — Task 2 Error Review and Evidence", B_T),
         Paragraph("The 20 reviewed errors of the BiGRU with attention are listed with their text, label, probability "
                   "and hand-assigned error type in failure_analysis.md.", B_B),
         Paragraph("Interpretation of the model comparison", B_S),
         tB([["Finding", "Evidence-based interpretation"],
             ["Mean-pooling baseline", "Captures strong lexical sentiment but ignores word order: error rate 7.47% on "
                                       "reviews with negation and 7.89% with a contrast word."],
             ["TextCNN", "Local 3-5 token phrases remove 23% of the baseline's errors, most on short reviews (6.96% to 5.29%)."],
             ["BiGRU + attention", "Reading the whole review in order removes 36% of the baseline's errors, 40% on "
                                   "negation and 38% on contrast reviews."],
             ["Remaining errors", "Mixed sentiment (7 of 20), label noise (5), faint praise (3), and one each of complex "
                                  "negation, target confusion, truncation, idiom and sarcasm."]],
            [1.5 * inch, W - 1.5 * inch]),
         Paragraph("Robustness slices (error rate, %)", B_S),
         tB([["Slice", "Baseline", "TextCNN", "BiGRU + attention"],
             ["Contains a negation (28,544)", "7.47", "5.50", "4.48"],
             ["Contains a contrast word (22,713)", "7.89", "5.94", "4.87"],
             ["Short, at most 50 words (9,287)", "6.96", "5.29", "5.01"],
             ["Medium, 51 to 150 words (16,910)", "6.94", "5.22", "4.09"],
             ["Long, more than 150 words (11,803)", "7.00", "5.52", "4.45"]],
            [2.4 * inch] + [(W - 2.4 * inch) / 3] * 3),
         Paragraph("Evidence locations", B_S),
         Paragraph(f"{T2}/configs/; metrics_report.csv; comparison.md; outputs/{RUN2}/ (comparison.csv, mcnemar.json, "
                   f"predictions_test.npz, roc_curves.png, pr_curves.png, reliability.png, slice_error_rates.png, "
                   f"embedding_neighbours.md); checkpoints/{RUN2}/; results.md; failure_analysis.md; reproducibility/raw_logs/ "
                   f"and reproducibility/manifests/.", B_N)]
    return a + b + c


def task3():
    a = [Paragraph("Shibin Thomas - Task 3 Detailed Findings", A_T),
         Paragraph("Architecture and training. Shibin trained a CycleGAN from random initialisation on 300 Monet paintings "
                   "and 7,038 photos. Each direction has a ResNet generator with nine residual blocks and resize-convolution "
                   "upsampling; each domain has a 70x70 PatchGAN discriminator. The objective combines a least-squares "
                   "adversarial loss, cycle-consistency loss (lambda 10) and identity loss (lambda 5).", A_B),
         tA([["Configuration", "Value"],
             ["Training", "40 epochs (20 constant + 20 linear decay); 2,000 unpaired pairs per epoch; batch size 4; 256x256"],
             ["Optimizer", "Adam; learning rate 0.0002; beta1 0.5; beta2 0.999"],
             ["Augmentation / buffer", "Resize 286, random crop 256, horizontal flip; image pool 50"],
             ["Stability", "fp32 with TF32 matmuls; 0 non-finite steps in 20,000; checkpoint every epoch with resume"],
             ["Official evaluation", "Instructor protocol re-implemented and unit-tested: first 300 sorted images per direction"],
             ["Results", f"B2A FID {f3(B2A, 'fid', 3)} / MiFID {f3(B2A, 'mifid_script', 4)}; A2B FID {f3(A2B, 'fid', 3)} / "
                         f"MiFID {f3(A2B, 'mifid_script', 4)}; combined FID {FID:.4f} / MiFID {MIFID:.4f}"],
             ["Kaggle", f"Submitted under PairProgramming_Team_09; evaluator score {(FID + MIFID) / 2:.2f}"],
             ["Evidence", "submission.csv, full_metrics_report.csv, eval_metrics.json, steps.csv/epochs.csv, loss and epoch "
                          "curves, pred_A2B/pred_B2A (600 images), G_AB.pt/G_BA.pt, logs and manifest"]],
            [1.45 * inch, W - 1.45 * inch]),
         Spacer(1, 10),
         Paragraph("Failure analysis. Dark and night photos are washed out into a bright speckled texture, saturated reds "
                   "and oranges are recoloured towards Monet's blues while the reconstruction recovers them (CycleGAN "
                   "steganography), a fine stipple pattern replaces brush strokes, foggy paintings become saturated in the "
                   "photo direction, and Monet-to-photo outputs stay painterly.", A_B)]
    # six representative photo-to-Monet outputs (same first six photos as Denisha's previews)
    names = sorted(p.name for p in (ROOT / T3 / "outputs/pred_B2A").glob("*.jpg"))[:6]
    grid = PILImage.new("RGB", (3 * 256 + 2 * 8, 2 * 256 + 8), "white")
    for i, n in enumerate(names):
        grid.paste(PILImage.open(ROOT / T3 / "outputs/pred_B2A" / n).convert("RGB").resize((256, 256)),
                   ((i % 3) * 264, (i // 3) * 264))
    buf = io.BytesIO()
    grid.save(buf, "JPEG", quality=90)
    buf.seek(0)
    b = [PageBreak(), Paragraph("Shibin Task 3 - Representative Generated Images", A_T),
         Paragraph("Six photo-to-Monet outputs of Shibin's model for the first six sorted photos, the same inputs as "
                   "Denisha's previews; the complete 300 + 300 translations are in task3_gan/shibin_thomas/outputs.", A_B),
         Image(buf, width=W * 0.78, height=W * 0.78 * grid.size[1] / grid.size[0])]
    c = [PageBreak(), Paragraph("Shibin Thomas — Task 3 Full Configuration and Official Evaluation", B_T),
         Paragraph("Shibin independently trained a CycleGAN for unpaired Monet-to-photo and photo-to-Monet translation. "
                   "The reported FID and MiFID follow the instructor's evaluation protocol on the first 300 sorted images.", B_B),
         Paragraph("Architecture and training configuration", B_S),
         tB([["Item", "Shibin configuration"],
             ["Dataset", "300 Monet paintings in domain A; 7,038 photos in domain B; unpaired sampling"],
             ["Model", "Two ResNet-9 generators (11.38M parameters each, resize-convolution upsampling, InstanceNorm, "
                       "reflection padding) and two 70x70 PatchGAN discriminators (2.76M each)"],
             ["Image/training", "Image size 256; batch size 4; 2,000 pairs per epoch; 40 epochs; decay begins at epoch 21"],
             ["Optimisation", "Adam with learning rate 0.0002; betas (0.5, 0.999); cycle weight 10; identity weight 5; LSGAN"],
             ["Stability", "Image pool 50; fp32/TF32; gradient norms logged every step; resumable checkpoints"],
             ["Export", "All 300 Monets and the first 300 sorted photos translated deterministically; JPEG quality 95"]],
            [1.4 * inch, W - 1.4 * inch]),
         Paragraph("Official evaluation and competition result", B_S),
         tB([["Direction/result", "Score"],
             ["B2A FID", f3(B2A, "fid", 3)], ["B2A MiFID", f3(B2A, "mifid_script", 4)],
             ["A2B FID", f3(A2B, "fid", 3)], ["A2B MiFID", f3(A2B, "mifid_script", 4)],
             ["Combined FID", f"{FID}"], ["Combined MiFID", f"{MIFID}"],
             ["Evaluator score (FID + MiFID) / 2", f"{(FID + MIFID) / 2:.2f}"],
             ["Additional (B2A / A2B)", f"KID {f3(B2A, 'kid', 4)} / {f3(A2B, 'kid', 4)}; precision {f3(B2A, 'precision')} / "
                                        f"{f3(A2B, 'precision')}; recall {f3(B2A, 'recall')} / {f3(A2B, 'recall')}; "
                                        f"cycle L1 {f3(B2A, 'cycle_l1', 4)} / {f3(A2B, 'cycle_l1', 4)}"]],
            [2.4 * inch, W - 2.4 * inch])]
    m1 = means(r1)
    mp, ma = means(r1, "photo->monet"), means(r1, "monet->photo")
    rows = [["Rater / subset", "Style", "Content", "Artifacts"],
            ["Rater 1 (Shibin), all 30", *[f"{x:.2f}" for x in m1]],
            ["Rater 1 (Shibin), photo to Monet (15)", *[f"{x:.2f}" for x in mp]],
            ["Rater 1 (Shibin), Monet to photo (15)", *[f"{x:.2f}" for x in ma]]]
    rows.append(["Rater 2 (Denisha), all 30", *[f"{x:.2f}" for x in means(r2)]] if len(r2) == 30
                else ["Rater 2 (Denisha)", "pending", "pending", "pending"])
    d = [PageBreak(), Paragraph("Shibin Thomas — Task 3 Artifacts, Visual Review and Human Audit", B_T),
         Paragraph("The exported folders contain 300 + 300 generated images. The evaluator, local metric files, Kaggle "
                   "submission, generator checkpoints and the blinded human audit are recorded below.", B_B),
         Paragraph("Artifact and reproducibility table", B_S),
         tB([["Artifact", "Purpose/location"],
             ["submission.csv", "Kaggle submission with the required ID, FID and MiFID columns"],
             ["full_metrics_report.csv", "Instructor-protocol and additional metrics by translation direction"],
             ["steps.csv, epochs.csv, loss and epoch curves", "Per-step losses and gradient norms; per-epoch FID and cycle error"],
             ["pred_A2B / pred_B2A", "300 + 300 generated images (the scored sets)"],
             ["G_AB.pt / G_BA.pt", "Final generator weights, committed in checkpoints/cyclegan_v1_20261001-201801"],
             ["human_audit/", "30 blinded panels, hidden direction key, rater1.csv and rater2.csv"]],
            [2.0 * inch, W - 2.0 * inch]),
         Paragraph("Human audit of Shibin's model (scores 1 to 5)", B_S),
         tB(rows, [2.6 * inch] + [(W - 2.6 * inch) / 3] * 3),
         Spacer(1, 6),
         Paragraph("Shibin also rated Denisha's 30 photo-to-Monet images as rater 2 (means: style 4.03, content 4.33, "
                   "artifacts 4.30). The completed two-rater sheet is report/human_audit/denisha_model_human_audit.csv.", B_B)]
    return a + b + c + d


def final():
    audit = ("Complete: 30 blinded panels, both raters." if len(r2) == 30 else
             "Rater 1 complete (30 panels); rater 2 (Denisha) to be added. Shibin also completed rater 2 for Denisha's model.")
    return [Paragraph("Final Contribution Checklist — Shibin Thomas", B_T),
            tB([["Requirement", "Verified status / evidence"],
                ["Part 1 member folder and artifacts", "Complete: code, unit tests, configuration, executed notebook, "
                                                       "checkpoint, metrics, samples, curves, logs, manifest, results and "
                                                       "failure analysis in task1_llm/shibin_thomas."],
                ["Part 2 three classifiers", "Complete: baseline, TextCNN and BiGRU with attention; metrics, checkpoints, "
                                             "histories and the 20-case review in task2_sentiment/shibin_thomas."],
                ["Part 3 CycleGAN artifacts", "Complete: generators, 600 generated images, evaluator metrics, submission "
                                              "file, logs, manifest and failure analysis in task3_gan/shibin_thomas."],
                ["Two-rater human audit", audit],
                ["Official evaluation script", "Instructor protocol re-implemented and unit-tested against the script; "
                                               "submission.csv produced from it."],
                ["Kaggle submission evidence", f"Shibin evaluator score {(FID + MIFID) / 2:.2f} recorded; team "
                                               f"PairProgramming_Team_09 best entry -50.6033, rank 34."],
                ["Checkpoints", "All Shibin checkpoints are committed in the repository (no external delivery needed)."]],
               [1.9 * inch, W - 1.9 * inch])]


def render(flow, footer):
    buf = io.BytesIO()

    def on_page(canvas, doc):
        if footer:
            canvas.saveState()
            canvas.setFont("Helvetica", 7.5)
            canvas.setFillColor(GREY)
            canvas.drawString(0.75 * inch, 0.45 * inch, "Team 09 - Shibin Thomas supplementary findings")
            canvas.restoreState()

    SimpleDocTemplate(buf, pagesize=letter, leftMargin=0.75 * inch, rightMargin=0.75 * inch, topMargin=0.75 * inch,
                      bottomMargin=0.75 * inch).build(flow, onFirstPage=on_page, onLaterPages=on_page)
    buf.seek(0)
    return PdfReader(buf)


def main() -> None:
    base = PdfReader(str(BASE))
    assert len(base.pages) == 30, "unexpected base report"
    inserts = {3: render(contribution(), True), 10: render(task1(), True), 18: render(task2(), True),
               28: render(task3(), True), 30: render(final(), False)}
    w = PdfWriter()
    for i, page in enumerate(base.pages, 1):
        w.add_page(page)
        if i in inserts:
            for p in inserts[i].pages:
                w.add_page(p)
    w.add_metadata({"/Title": "DATA 266 Lab 1 Report, Team 09", "/Author": "Shibin Thomas, Denisha Ketan Tank"})
    with open(OUT, "wb") as f:
        w.write(f)
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}: {len(w.pages)} pages "
          f"({len(base.pages)} from Denisha's version + {len(w.pages) - len(base.pages)} Shibin pages)")


if __name__ == "__main__":
    main()
