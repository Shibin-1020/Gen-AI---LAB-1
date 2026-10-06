from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether

OUT = Path(__file__).with_name('DATA266_Lab1_Report_Team_09_Revised.pdf')
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='TitleCenter', parent=styles['Title'], alignment=TA_CENTER, fontSize=20, leading=25, spaceAfter=18))
styles.add(ParagraphStyle(name='SubCenter', parent=styles['Normal'], alignment=TA_CENTER, fontSize=11, leading=15))
styles.add(ParagraphStyle(name='Small', parent=styles['BodyText'], fontSize=8.5, leading=11))
styles.add(ParagraphStyle(name='Tiny', parent=styles['BodyText'], fontSize=7.2, leading=9))
styles.add(ParagraphStyle(name='Callout', parent=styles['BodyText'], backColor=colors.HexColor('#fff4cc'), borderColor=colors.HexColor('#d6a700'), borderWidth=0.6, borderPadding=7, leading=13, spaceBefore=6, spaceAfter=8))

def P(text, style='BodyText'):
    return Paragraph(text, styles[style])

def table(data, widths=None, font=7.5):
    cooked=[]
    for row in data:
        cooked.append([x if hasattr(x,'wrap') else Paragraph(str(x), styles['Tiny']) for x in row])
    t=Table(cooked, colWidths=widths, repeatRows=1, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#203864')),('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('GRID',(0,0),(-1,-1),0.35,colors.HexColor('#a6a6a6')),('VALIGN',(0,0),(-1,-1),'TOP'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white, colors.HexColor('#f2f5f9')]),
        ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
        ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4),
    ]))
    return t

def footer(canvas, doc):
    canvas.saveState(); canvas.setFont('Helvetica',8); canvas.setFillColor(colors.grey)
    canvas.drawString(0.65*inch,0.4*inch,'DATA 266 Lab 1 - Team 09')
    canvas.drawRightString(7.85*inch,0.4*inch,f'Page {doc.page}')
    canvas.restoreState()

story=[]
story += [P('DATA 266: Generative AI, Fall 2026', 'SubCenter'), P('Lab 1 Report', 'TitleCenter'), P('Character-Level Language Model Pretraining, Sentiment Classification, and CycleGAN Image Style Transfer', 'SubCenter'), Spacer(1,18), P('<b>Team 09: PairProgramming_Team_09</b>', 'SubCenter'), P('Shibin Thomas and Denisha Ketan Tank', 'SubCenter'), P('Shared repository: https://github.com/Shibin-1020/Gen-AI---LAB-1', 'SubCenter'), P('October 2026', 'SubCenter'), PageBreak()]

story += [P('1. Team ownership and reproducibility','Heading1'), P('Shibin Thomas and Denisha Ketan Tank each independently implemented and trained a model for every task in their own member folders. Shibin contributed the shared repository structure, shared evaluation utilities, smoke tests, and his independent Task 1, Task 2, and Task 3 runs. Denisha contributed an independent from-scratch character GPT, three sentiment classifiers, an independent CycleGAN run, official Task 3 evaluation, generated outputs, and a Kaggle submission. The comparison and report were assembled as a team.'), P('Repository layout follows task1_llm/<member>, task2_sentiment/<member>, and task3_gan/<member>. Raw logs and manifests are stored under reproducibility/. Dataset files are not redistributed in the report; the approved download/preparation scripts and run metadata are included in the repository.'), P('The official Lab 1 PDF requires: independent work for all three tasks; executed notebooks and saved weights; metrics and failure analyses; Task 2 manual review of 20 errors; Task 3 evaluation with the provided script; a 30-sample, two-rater human audit; Kaggle score/rank; and one combined PDF report.'), P('<b>Submission status:</b> the results below are evidence-backed. The Task 3 two-rater audit is intentionally marked pending because scores must be collected from two real independent raters; no fabricated ratings are reported.', 'Callout')]

story += [P('2. Task 1: character-level GPT','Heading1'), P('Both members trained decoder-only Transformer language models from scratch on TinyStories characters. The models use learned token/position embeddings, causal self-attention, layer normalization, feed-forward blocks, residual connections, and a character vocabulary constructed from training data. No pretrained language model was used.'), P('Denisha configuration and result', 'Heading2')]
story.append(table([
    ['Item','Denisha result'],['Architecture','6-layer decoder-only Transformer; d_model 320; 8 heads; context 256; approximately 3,250,688 parameters'],['Training','AdamW configuration in task1_llm/member_denisha/config.yaml; 10 epochs; CUDA/Tesla T4 execution'],['Training loss','0.7128'],['Validation cross-entropy','0.6822 nats/character'],['Validation perplexity / BPC','1.9783 / 0.9842'],['Next-character accuracy','0.7822 (78.22%)'],['Gradient checks','gradient_nan_count = 0; final norm 0.2318; max norm 0.2872'],['Training time','3995.3 seconds; 2.25e5 training tokens/second'],['Evidence','task1_llm/member_denisha/outputs/metrics.csv, history.json, samples.json, loss_curves.png, final checkpoint, raw log and manifest'],
], [1.7*inch,5.4*inch]))
story += [P('Denisha’s failure analysis covers repetition, spelling instability, and loss of long-range coherence. The model’s 256-character context limits story continuity; character-level generation also makes rare-word spelling difficult. Smoke tests passed before training and the notebook generated the required artifacts.'), PageBreak()]

story += [P('3. Task 2: Yelp sentiment classification','Heading1'), P('Each member trained three models from scratch using learned embeddings: a mean-pooling baseline, a TextCNN, and an experimental GRU model. The official Yelp Polarity test split was evaluated with accuracy, macro/micro/weighted precision, recall and F1, ROC-AUC, PR-AUC, MCC, calibration, confusion matrices, confidence intervals, slice metrics, and McNemar comparisons.'), P('Denisha results on the official test split', 'Heading2')]
story.append(table([
    ['Model','Accuracy','Macro F1','ROC-AUC','MCC','Best val loss','Time'],
    ['Baseline mean pool','0.9321','0.9321','0.9804','0.8642','0.1804','26.5 s'],
    ['Experimental TextCNN','0.9468','0.9468','0.9883','0.8937','0.1376','180.3 s'],
    ['Experimental GRU','0.9519','0.9519','0.9901','0.9037','0.1261','159.0 s'],
], [1.75*inch,0.75*inch,0.75*inch,0.75*inch,0.65*inch,0.9*inch,0.75*inch]))
story += [P('The GRU was Denisha’s best model. It achieved 95.19% accuracy, zero gradient NaNs, and the strongest MCC. Its error rates were highest on short reviews, while medium and long reviews were slightly easier. The 20-case error review contains five false positives, five false negatives, five near-threshold cases, and five slice-specific cases, as required.'), P('Evidence: task2_sentiment/member_denisha/outputs/metrics_report.csv, metrics.json, error_review.json, data_analysis.json, history files, three checkpoints, results.md, failure_analysis.md, raw log, and manifest.'), PageBreak()]

story += [P('4. Task 3: CycleGAN image style transfer','Heading1'), P('The task translates between 300 Monet paintings and 7,038 photographs without paired examples. Denisha trained a CycleGAN with two generators and two PatchGAN discriminators using adversarial, cycle-consistency, and identity losses. Generated images were evaluated with the instructor-provided Part 3 evaluation notebook.'), P('Denisha official evaluation', 'Heading2')]
story.append(table([
    ['Metric','Photo -> Monet (B2A)','Monet -> Photo (A2B)','Official combined'],
    ['FID','98.360','103.228','100.7938'],['MiFID','0.4069','0.4189','0.4129'],['Scored images','300','300','300 per direction'],['Evaluator device','CPU','CPU','Provided evaluation script'],['Kaggle team result','-','-','PairProgramming_Team_09: -50.6033, rank 34'],
], [1.65*inch,1.65*inch,1.65*inch,2.15*inch]))
story += [P('Additional Denisha metrics include directional FID, KID, generative precision/recall, LPIPS evaluation records, cycle histories, loss curves, generated image archives, and the official two-row submission.csv. The Kaggle submission was produced from the trained CycleGAN outputs and was not manually edited.'), P('Failure analysis identified poor handling of dark scenes, texture and brush-stroke artifacts, washed-out colors, and content loss in foggy or low-contrast images. The model’s training checkpoint and epoch checkpoints are stored in the artifacts; the approximately 340 MB latest checkpoint should be stored in Google Drive rather than normal GitHub storage.'), P('Human audit requirement', 'Heading2'), P('The Lab requires two independent raters to score 30 fixed samples for style, content, and artifacts, followed by inter-rater agreement. The repository currently contains the 30-row audit template but no ratings. This section must be updated after both raters independently inspect the images; scores must not be invented or copied.'), PageBreak()]

story += [P('5. Cross-task comparison and limitations','Heading1'), P('Denisha’s Task 1 run provides a compact from-scratch character model with stable gradients and a validation BPC of 0.9842. Her Task 2 GRU is the strongest of her three classifiers at 95.19% accuracy and 0.9037 MCC. Her Task 3 run produced a valid official submission and a combined official FID of 100.7938, while the public Kaggle result for PairProgramming_Team_09 was -50.6033 at rank 34.'), P('The main limitations are single-run training for each model, different hardware between members, the small Monet domain, variance in FID from only 300 scored images, and the remaining human-audit evidence. Results should therefore be interpreted as reproducible single-run experiments rather than estimates of uncertainty over random seeds.'), P('Recommended next steps are to complete the two-rater audit, record Cohen’s kappa or percent agreement, attach the large checkpoint through Google Drive, execute/save any notebook versions required by the instructor, and package the final ZIP with Parts 1–3 and this report.'), P('6. Evidence map','Heading1')]
story.append(table([
    ['Requirement','Denisha evidence'],
    ['Task 1 notebook, weights, outputs','task1_llm/member_denisha/Task1_GPT_Colab.ipynb; checkpoints/gpt_from_scratch.pt; outputs/metrics.csv, history.json, samples.json, loss_curves.png; results.md; failure_analysis.md'],
    ['Task 2 notebook, three weights, metrics','task2_sentiment/member_denisha/Task2_Sentiment_Colab.ipynb; three checkpoint files; outputs/metrics_report.csv, metrics.json, error_review.json; results.md; failure_analysis.md'],
    ['Task 3 code, weights, outputs','task3_gan/member_denisha/Task3_CycleGAN_Colab.ipynb; run_task3.py; checkpoint archive; outputs/images.zip; official metrics and submission.csv'],
    ['Reproducibility','reproducibility/raw_logs/ and reproducibility/manifests/'],
    ['Shared repository','https://github.com/Shibin-1020/Gen-AI---LAB-1'],
    ['Pending before final ZIP','Two-rater human audit, agreement statistic, final Drive checkpoint link, and final packaging validation'],
], [1.85*inch,5.25*inch]))
story += [Spacer(1,12), P('References: DATA266 Lab 1 official handout; Vaswani et al., Attention Is All You Need; Zhu et al., Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks; the instructor-provided Part 3 evaluation script; official Yelp Polarity and TinyStories data sources.' , 'Small')]

doc=SimpleDocTemplate(str(OUT), pagesize=letter, rightMargin=.65*inch, leftMargin=.65*inch, topMargin=.65*inch, bottomMargin=.65*inch, title='DATA 266 Lab 1 Report Team 09 Revised')
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(OUT)
