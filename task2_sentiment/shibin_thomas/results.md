# Task 2 — Yelp Polarity sentiment classification · Shibin Thomas

Binary sentiment classification (0 = negative, 1 = positive) on Yelp Review Polarity with **no pretrained
embeddings or pretrained language models**. I built three models: a baseline and two experimental models.
All of them learn their word embeddings from scratch on my training split.

| | |
|---|---|
| Configs | [`configs/data.yaml`](configs/data.yaml), [`baseline_meanpool.yaml`](configs/baseline_meanpool.yaml), [`exp1_textcnn.yaml`](configs/exp1_textcnn.yaml), [`exp2_bigru_attn.yaml`](configs/exp2_bigru_attn.yaml) |
| Code / notebook | [`src/`](src/) · [`src/task2_yelp_sentiment.ipynb`](src/task2_yelp_sentiment.ipynb) |
| All metrics, all models | [`metrics_report.csv`](metrics_report.csv) · side by side: [`comparison.md`](comparison.md) |
| 20-error review | [`failure_analysis.md`](failure_analysis.md) |
| Plots | `outputs/<run_id>/` (ROC, PR, reliability, training curves, slices, confusion matrices) · EDA: `outputs/eda/main/` |
| Raw logs / manifest | `reproducibility/raw_logs/task2_sentiment/shibin_thomas/<run_id>/`, `reproducibility/manifests/task2_sentiment/shibin_thomas/<run_id>.json` |

## 1. Data analysis and preprocessing (2.1)

**Splits.** The official **test** split (38,000 reviews) is the test set, so it is identical for every team
member and our numbers are comparable. My **validation** set (50,000 reviews, stratified, seed 266) is carved
out of the official train split. It is used for early stopping and model selection, and never for testing.

<!-- DATA:START -->
_Filled in automatically from `data_processed/main/meta.json` when the full run finishes._
<!-- DATA:END -->

EDA plots: `outputs/eda/main/class_distribution.png`, `length_distribution.png`, `token_length_vs_maxlen.png`.

| Step | What I did | Why |
|---|---|---|
| Missing / malformed | rows with missing or non-string text, invalid labels, or empty text after normalisation are dropped; exact duplicates inside train and train reviews that also appear in test are removed | clean labels; no train→test leakage that would inflate test scores |
| Normalise | the Yelp dump contains literal `\n` and `\"` sequences and HTML entities (`&amp;`): converted to real characters | otherwise `n`, `quot` and `amp` would become frequent fake "words" |
| Lowercase | everything lowercased | "GREAT" and "great" carry the same sentiment; halves the vocabulary |
| Contractions | `didn't → did not`, `they're → they are` | punctuation removal would otherwise turn "didn't" into "didn" + "t" and lose the negation |
| Punctuation / special chars | everything except `a–z`, `0–9` and whitespace removed | removes noise; digits kept because "5 stars" / "1 star" are strong cues |
| Stopwords | scikit-learn English list **minus negations and contrast words** (*not, no, never, nothing, without, but, however, although, very, too, …*) | removing *not* would turn "not good" into "good" and flip the label; contrast words mark where the verdict changes |
| Stemming | NLTK Snowball (Porter2) stemmer: *loved / loving / loves → love* | merges inflections, so rare forms share statistics; stemming beats lemmatisation here because it needs no POS tagging and the token only has to be a consistent ID, not a dictionary word |
| Tokenisation | whitespace split of the processed text; vocabulary from **my training split only**, min frequency 3, top 30,000; `0 = <pad>`, `1 = <unk>` | building the vocabulary on train only avoids leakage; min frequency 3 drops typos and one-off names |
| Encoding | 256 token ids per review; longer reviews keep the **first 128 + last 128** tokens (head+tail) | 256 covers the large majority of reviews (see above); reviewers often state their verdict at the end, which plain head truncation would cut |
| Embeddings | `nn.Embedding(30,000 × 128)` per model, random init, **learned from scratch** jointly with the classifier; `<pad>` row fixed at 0 | no pretrained vectors allowed; 128-d is enough for a 30K vocabulary of sentiment-bearing stems and keeps the embedding at 3.8 M parameters |

## 2. Models and justification (2.2.1–2.2.2)

All three models use the same preprocessed data, vocabulary and a 128-dimensional embedding trained from
scratch, so the differences in the results come from the **architecture**. Each outputs a single logit,
P(positive) = sigmoid(logit), trained with binary cross-entropy.

### Baseline — `baseline_meanpool`: bag of learned embeddings + MLP
`embedding (128) → masked mean over the review's tokens → dropout → Linear(128→64) → ReLU → dropout → Linear(64→1)`
* **Why:** this is the neural version of a bag-of-words model (fastText-style, Joulin et al. 2017). It is
  fast, hard to overfit and surprisingly strong on polarity tasks, because most of the signal is in which
  words appear ("delicious", "rude").
* **Weakness by design:** word order is ignored, so "not good" and "good, not bad" are almost the same bag.
  That is exactly what the experimental models are meant to fix.
* **Embedding choice:** mean pooling makes the embedding table do all the work. Words with similar sentiment
  end up with similar vectors because their average must separate the classes.

### Experiment 1 — `exp1_textcnn`: convolutional n-gram detector (Kim, 2014)
`embedding (128) → three Conv1d layers (widths 3, 4, 5; 128 filters each; ReLU) → max-over-time pooling (padding masked) → dropout 0.5 → Linear(384→1)`
* **Change vs baseline:** architecture. Each filter is a learned detector for a 3–5-token phrase such as
  "not worth", "highly recommend" or "never go back". Max pooling keeps the strongest match anywhere in
  the review.
* **Why:** it captures local word order (negation and intensifier phrases) that the baseline cannot see,
  while staying fully parallel and fast. Widths 3–5 cover the typical span of a negation plus its target
  after stopword removal.
* **Regularisation:** dropout 0.5 on the 384 pooled features, the standard choice for TextCNN.

### Experiment 2 — `exp2_bigru_attn`: bidirectional GRU + attention pooling
`embedding (128) → dropout → 2-layer bidirectional GRU (128 per direction, packed sequences) → additive attention over the 256-d states (padding masked) → dropout → Linear(256→1)`
* **Change vs baseline:** architecture.
  * The recurrent encoder reads the whole review in order, in both directions, so each word's
    representation depends on its full context. "Great" after "but" is read differently from "great" in
    "would be great if".
  * Attention pooling (Bahdanau et al. 2015; Yang et al. 2016) learns which positions carry the verdict,
    instead of averaging everything (baseline) or taking only the strongest local feature (CNN).
* **Why GRU and not LSTM:** similar accuracy on sentence classification, with fewer parameters and faster
  training.
* **Why 2 layers:** the second layer combines phrase-level features into clause-level ones.
* **Why attention:** it also gives interpretable weights (see the notebook).

## 3. Hyperparameters

| | Baseline (mean-pool) | Exp 1 (TextCNN) | Exp 2 (BiGRU + attn) | Reason |
|---|---|---|---|---|
| Embedding | 30K × 128, from scratch | 30K × 128, from scratch | 30K × 128, from scratch | identical, to isolate the architecture |
| Encoder | mean pooling | widths 3/4/5 × 128 filters | 2 × BiGRU 128/direction + attention 128 | see §2 |
| Dropout | 0.3 | 0.5 | 0.3 | CNN has more pooled features to regularise |
| Batch size | 512 | 256 | 256 | the baseline is cheap per example, so larger batches are fine |
| Peak LR (AdamW) | 2e-3 | 1e-3 | 1e-3 | shallow models tolerate a larger LR; RNNs are more sensitive |
| Schedule | 3% linear warm-up, cosine to 5% of peak | same | same | stable start, fine convergence |
| Weight decay | 1e-4 (not on embeddings, biases) | same | same | decaying sparse embedding rows hurts rare words |
| Gradient clipping | 1.0 | 1.0 | 1.0 | protects the RNN against exploding gradients |
| Max epochs / early stopping | 8 / patience 2 on val macro-F1 | 5 / 2 | 5 / 2 | ~510K training reviews per epoch: few epochs are enough |
| Threshold | 0.5 | 0.5 | 0.5 | balanced classes; all metrics use the same threshold |

## 4. Results on the official test split (2.2.3)

Metric definitions follow the team's shared `task2_sentiment/shared_eval/metrics.py`:
* micro-F1 equals accuracy for single-label binary classification;
* ECE uses 15 equal-width confidence bins;
* bootstrap CIs use 1,000 resamples;
* McNemar: b = baseline right & model wrong, c = baseline wrong & model right.

<!-- METRICS:START -->
_Filled in automatically from `metrics_report.csv` when the full run finishes._
<!-- METRICS:END -->

Plots: ![training curves](outputs/RUN_ID/training_curves.png)
![ROC](outputs/RUN_ID/roc_curves.png) ![PR](outputs/RUN_ID/pr_curves.png)
![reliability](outputs/RUN_ID/reliability.png) ![slices](outputs/RUN_ID/slice_error_rates.png)

## 5. Comparative analysis and observations (2.3)

_Written after the full run, from the numbers above:_
* comparison of the two experimental models against the baseline and each other: accuracy / macro-F1 with
  CIs, McNemar significance, calibration (Brier / ECE), robustness slices, cost (parameters, time,
  examples/sec, memory);
* strengths, weaknesses and limitations of each model;
* possible improvements and future work.

## 6. Comparison with teammates (team)

| Member | Model | Architecture summary | Embedding | Key hyperparameters | Test accuracy | Macro-F1 | MCC |
|---|---|---|---|---|---|---|---|
| Shibin Thomas | baseline_meanpool | masked mean of embeddings → MLP 64 | 128-d, scratch | lr 2e-3, batch 512 | | | |
| Shibin Thomas | exp1_textcnn | Conv 3/4/5 × 128, max-pool | 128-d, scratch | lr 1e-3, dropout 0.5 | | | |
| Shibin Thomas | exp2_bigru_attn | 2-layer BiGRU 128 + attention | 128-d, scratch | lr 1e-3, dropout 0.3 | | | |
| _teammate_ | | | | | | | |

## 7. Hardware (2.2.5)

The exact CPU and GPU used to train **each** model are recorded automatically: the `hardware` and `cpu` rows of
`metrics_report.csv` (per model), each model's `outputs/<run_id>/<model>/train_summary.json`, and the run
manifest (GPU model and memory, CUDA / cuDNN versions, CPU model, OS).

## References
* Zhang, Zhao & LeCun, *Character-level Convolutional Networks for Text Classification* (Yelp polarity dataset), NeurIPS 2015.
* Kim, *Convolutional Neural Networks for Sentence Classification*, EMNLP 2014.
* Joulin et al., *Bag of Tricks for Efficient Text Classification* (fastText), EACL 2017.
* Bahdanau, Cho & Bengio, *Neural Machine Translation by Jointly Learning to Align and Translate*, ICLR 2015.
* Yang et al., *Hierarchical Attention Networks for Document Classification*, NAACL 2016.
* Cho et al., *Learning Phrase Representations using RNN Encoder–Decoder* (GRU), EMNLP 2014.
* Guo et al., *On Calibration of Modern Neural Networks* (ECE), ICML 2017.
* McNemar, *Note on the sampling error of the difference between correlated proportions*, Psychometrika 1947.
