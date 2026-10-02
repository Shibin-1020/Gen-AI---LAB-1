# Task 2 — Yelp Polarity sentiment classification (no pretrained embeddings / LMs)

```
task2_sentiment/
├── data/                  shared raw dataset
│   ├── download_yelp.py
│   └── raw/               yelp_polarity_{train,test}.jsonl (git-ignored) + *.meta.json (committed)
├── shared_eval/
│   └── metrics.py         team-agreed metrics: accuracy, P/R/F1 (macro/micro/weighted), confusion matrix,
│                          ROC-AUC, PR-AUC, MCC, Brier, ECE, bootstrap CIs, McNemar, slice definitions
└── <member_name>/         each member's own 3 models (baseline + 2 experimental)
```

```bash
python task2_sentiment/data/download_yelp.py                  # 560K train + 38K test -> data/raw/
python task2_sentiment/data/download_yelp.py --max-rows 4000  # small files for smoke tests
```

## Team evaluation protocol
* **Test set:** the official Yelp polarity test split (38,000 reviews) for everyone; validation is carved
  out of the official train split (own seed per member).
* **Threshold:** 0.5 on P(positive). Every metric comes from `shared_eval/metrics.py`:
  * ECE uses 15 bins;
  * bootstrap CIs use 1,000 resamples (seed 0);
  * McNemar is computed against your own baseline.
* **Slices** are defined on the raw text:
  * review length: short (≤50 words), medium (51–150), long (>150);
  * has negation; has contrast word;
  * neither negation nor contrast;
  * exclamation-heavy.
* No pretrained embeddings or pretrained language models.

## Members
| Member | Folder | Baseline | Experimental 1 | Experimental 2 |
|---|---|---|---|---|
| Shibin Thomas | [`shibin_thomas/`](shibin_thomas/) | mean-pooled embeddings + MLP | TextCNN (3/4/5-gram filters) | 2-layer BiGRU + attention |
| _teammate_ | | | | |
