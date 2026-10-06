# Task 2 results - Denisha Ketan Tank

## Experimental setup

The Yelp Polarity data was split into stratified train, validation, and test sets. Text
was lowercased, tokenized with a deterministic regular expression, filtered with the
configured stopword list, and lightly stemmed. The vocabulary was built from the
training split only; embeddings were initialized and learned from scratch. No
pretrained embeddings or language models were used.

The three models were deliberately different:

| Model | Architecture | Main hyperparameters |
|---|---|---|
| Baseline | Mean-pooled embedding + MLP | vocab 30,000; embedding 128; hidden 128; dropout 0.2 |
| Experimental CNN | Embedding + parallel 1-D convolutions | embedding 128; 128 channels; kernels 3/4/5; dropout 0.3 |
| Experimental GRU | Bidirectional 2-layer GRU | embedding 128; hidden 128 per direction; dropout 0.3 |

Shared training settings were AdamW, learning rate 0.001, weight decay 0.0001,
maximum sequence length 256, batch size 512, bfloat16 AMP when supported, validation
loss-based learning-rate reduction, and early stopping.

## Results

| Model | Accuracy | Macro-F1 | ROC-AUC | PR-AUC | MCC | Brier | ECE | Best val. loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.93210 | 0.93210 | 0.98036 | 0.97975 | 0.86420 | 0.05098 | 0.00639 | 0.18041 |
| CNN | 0.94683 | 0.94683 | 0.98827 | 0.98769 | 0.89366 | 0.03959 | 0.00805 | 0.13756 |
| GRU | **0.95186** | **0.95186** | **0.99011** | **0.98979** | **0.90372** | **0.03646** | 0.00810 | **0.12613** |

All three runs reported zero gradient NaNs. The GRU is the selected final model because
it achieved the best accuracy, macro-F1, ROC-AUC, PR-AUC, MCC, Brier score, and validation
loss. Its 95% accuracy confidence interval was [0.95073, 0.95309], its macro-F1 interval
was [0.95073, 0.95309], and its MCC interval was [0.90147, 0.90619].

## Comparative analysis

The baseline is fast and provides a useful lower-capacity reference, but mean pooling
loses word order and local phrase structure. The CNN improves over the baseline by
capturing short n-gram patterns such as intensifiers and local negation. The GRU performs
best because its recurrent state can preserve order and longer dependencies. Its lower
validation loss and stronger ranking metrics show that the improvement is not limited to
the 0.5 classification threshold.

Performance was strongest on medium and long reviews. For the GRU, macro-F1 was 0.94461
on short reviews, 0.95331 on medium reviews, and 0.95198 on long reviews. This suggests
that the model benefits from additional context; short reviews remain the main robustness
weakness.

Calibration was good for all models. The GRU had Brier score 0.03646 and expected
calibration error 0.00810. The CNN had a slightly lower ECE, so the GRU's stronger
classification performance should be reported together with the calibration caveat.

The paired McNemar tests showed significant improvements over the baseline: CNN versus
baseline p < 1e-111 and GRU versus baseline p < 1e-180. These tests indicate that the
experimental models corrected substantially more baseline errors than they introduced.

## Hardware and efficiency

The recorded device was an NVIDIA Tesla T4 CUDA GPU in Google Colab. Recorded peak memory was 235.16
MB for the baseline, 431.43 MB for the CNN, and 1,716.77 MB for the GRU. Training times
were 36.99 s, 496.98 s, and 148.88 s respectively in the supplied run. The raw log is
preserved at `reproducibility/raw_logs/task2_denisha.log`, and the run manifest is at
`reproducibility/manifests/task2_denisha.json`.

## Final choice and next steps

The GRU is the final model for Task 2. A testable next step is to preserve negation and
phrase boundaries more explicitly—for example, add a small learned attention-pooling
head or compare the current stemming/stopword policy against a version that retains all
function words—and evaluate the change on the same fixed split.

The manual review is documented in `failure_analysis.md` using the updated review file.
It contains all 20 required unique records: five confident false positives, five confident
false negatives, five near-threshold errors, and five slice-specific errors. The recurring
failure modes are short-context ambiguity, negation, mixed aspect sentiment, slang, and
positive phrases that conflict with the overall review conclusion.
