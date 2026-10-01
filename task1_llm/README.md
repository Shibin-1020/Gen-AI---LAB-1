# Task 1 — Build a GPT-style LLM from scratch (TinyStories, character level)

```
task1_llm/
├── data/                      shared raw dataset (same file for every member)
│   ├── download_tinystories.py
│   └── raw/                   tinystories_train.jsonl (git-ignored, ~2 GB) + *.meta.json (committed: rows, SHA-256)
├── shared_eval/               team-agreed evaluation, so every member's numbers are comparable
│   ├── metrics.py             perplexity, bits-per-char, generalisation gap, Distinct-1/2/3, repeated 4-gram rate
│   └── prompts.json           10 fixed generation prompts
└── <member_name>/             each member's own, independent model (see the brief §5 for the layout)
```

## Shared data
```bash
python task1_llm/data/download_tinystories.py                 # full HF train split -> data/raw/tinystories_train.jsonl
python task1_llm/data/download_tinystories.py --max-stories 3000   # small file for smoke tests
```
Each member then draws **their own** 100K train / 10K validation split from this file with their own seed,
and stores the split indices in their own `data_processed/`.

## Team evaluation protocol (agree before training)
* Report every metric in the brief, using `shared_eval/metrics.py`:
  * CE is per **character**, in nats; perplexity = exp(CE); BPC = CE / ln 2.
  * Generalisation gap = val CE − train CE, with train CE measured in eval mode.
  * Distinct-n is computed at corpus level over word n-grams of the sampled continuations.
  * Repeated 4-gram rate is computed per continuation, then averaged.
* Generate from `shared_eval/prompts.json` with greedy decoding and temperature 0.8.
* Use 500 new characters per sample and 3 samples per prompt.
* Gradient norm = global L2 norm before clipping, logged every step.
* Loss spike = loss > 1.5 × EMA(0.99) of the loss, after warm-up.

## Members
| Member | Folder | Model summary |
|---|---|---|
| Shibin Thomas | [`shibin_thomas/`](shibin_thomas/) | 6 layers, d=320, 8 heads, context 256, ~7.5M params, pre-LN, tied embeddings |
| _teammate_ | | |
