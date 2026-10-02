# Task 2 model comparison - run `yelp3_20261001-182849`

Official Yelp polarity test split, n = 38,000. McNemar: b = baseline right & model wrong, c = baseline wrong & model right.

| Metric | baseline_meanpool (baseline) | exp1_textcnn (experimental) | exp2_bigru_attn (experimental) |
|---|---|---|---|
| Accuracy | 0.9304 | 0.9467 | 0.9558 |
| Accuracy 95% CI | [0.9276, 0.9330] | [0.9446, 0.9488] | [0.9537, 0.9577] |
| Precision (macro) | 0.9304 | 0.9467 | 0.9558 |
| Recall (macro) | 0.9304 | 0.9467 | 0.9558 |
| F1 (macro) | 0.9304 | 0.9467 | 0.9558 |
| F1 macro 95% CI | [0.9276, 0.9330] | [0.9446, 0.9488] | [0.9537, 0.9577] |
| Precision (micro) | 0.9304 | 0.9467 | 0.9558 |
| Recall (micro) | 0.9304 | 0.9467 | 0.9558 |
| F1 (micro) | 0.9304 | 0.9467 | 0.9558 |
| Precision (weighted) | 0.9304 | 0.9467 | 0.9558 |
| Recall (weighted) | 0.9304 | 0.9467 | 0.9558 |
| F1 (weighted) | 0.9304 | 0.9467 | 0.9558 |
| Confusion matrix [TN FP; FN TP] | [17,718 1,282; 1,363 17,637] | [17,951 1,049; 976 18,024] | [18,029 971; 710 18,290] |
| ROC-AUC | 0.9800 | 0.9883 | 0.9917 |
| PR-AUC | 0.9802 | 0.9885 | 0.9920 |
| MCC | 0.8608 | 0.8934 | 0.9116 |
| MCC 95% CI | [0.8553, 0.8659] | [0.8891, 0.8975] | [0.9076, 0.9155] |
| Brier score | 0.0520 | 0.0399 | 0.0337 |
| ECE (15 bins) | 0.0045 | 0.0049 | 0.0110 |
| McNemar vs baseline (b / c, p) | - (baseline) | 613 / 1,233, p=4.67e-47 | 557 / 1,521, p=4.65e-99 |
| Parameters | 3,848,321 | 4,037,377 | 4,367,873 |
| Training time (min) | 0.3545 | 0.9865 | 19.1767 |
| Epochs run (best) | 8 (8) | 5 (5) | 5 (4) |
| Train examples/sec | 279,909 | 54,074 | 2,271 |
| Inference examples/sec | 2,126,409 | 193,858 | 29,489 |
| Peak GPU memory (MB) | 1,149 | 1,277 | 1,392 |
| Peak CPU RSS (MB, process) | 2,723 | 2,723 | 2,723 |
| Hardware | NVIDIA GeForce RTX 5090 | NVIDIA GeForce RTX 5090 | NVIDIA GeForce RTX 5090 |

## Robustness: macro-F1 / error rate per slice

| Slice (n) | baseline_meanpool macro-F1 / error | exp1_textcnn macro-F1 / error | exp2_bigru_attn macro-F1 / error |
|---|---|---|---|
| all (38,000) | 0.9304 / 0.0696 | 0.9467 / 0.0533 | 0.9558 / 0.0442 |
| short (<=50 words) (9,287) | 0.9277 / 0.0696 | 0.9449 / 0.0529 | 0.9478 / 0.0501 |
| medium (51-150 words) (16,910) | 0.9306 / 0.0694 | 0.9478 / 0.0522 | 0.9591 / 0.0409 |
| long (>150 words) (11,803) | 0.9274 / 0.0700 | 0.9428 / 0.0552 | 0.9541 / 0.0445 |
| has negation (28,544) | 0.9228 / 0.0747 | 0.9433 / 0.0550 | 0.9539 / 0.0448 |
| has contrast (but/however/...) (22,713) | 0.9203 / 0.0789 | 0.9400 / 0.0594 | 0.9509 / 0.0487 |
| no negation & no contrast (6,688) | 0.9263 / 0.0493 | 0.9322 / 0.0453 | 0.9376 / 0.0416 |
| exclamation-heavy (>=3 '!') (6,944) | 0.9490 / 0.0497 | 0.9631 / 0.0359 | 0.9707 / 0.0284 |
