# Task 1 — GPT-style character-level LLM from scratch · Shibin Thomas

Character-level, decoder-only Transformer (GPT) trained from scratch on my own 100K / 10K split of
TinyStories. Every Transformer component (LayerNorm, multi-head causal self-attention, FFN, residual
blocks) is hand-written in [`src/model.py`](src/model.py); no prebuilt Transformer/attention module is used
(enforced by [`tests/test_model.py`](tests/test_model.py)).

| | |
|---|---|
| Config | [`configs/gpt_char_v1.yaml`](configs/gpt_char_v1.yaml) |
| Code / notebook | [`src/`](src/) · [`src/task1_gpt_tinystories.ipynb`](src/task1_gpt_tinystories.ipynb) |
| Metrics (all) | [`metrics_report.csv`](metrics_report.csv) |
| Failure analysis | [`failure_analysis.md`](failure_analysis.md) |
| Plots / samples | `outputs/<run_id>/loss_curves.png`, `grad_norm.png`, `lr_schedule.png`, `samples.md` |
| Raw logs / manifest | `reproducibility/raw_logs/task1_llm/shibin_thomas/<run_id>/`, `reproducibility/manifests/task1_llm/shibin_thomas/<run_id>.json` |
| Checkpoint | `checkpoints/<run_id>/best_model.pt` (epoch with the lowest validation loss) |

## 1. Data preprocessing (1.1)

| Step | What I did | Why |
|---|---|---|
| Load | HF `roneneldan/TinyStories` train split, streamed to a shared JSONL file (`task1_llm/data/`) | one raw copy for the team, identical line order for everyone |
| Clean | typographic unicode → ASCII (’ “ ” — … → ' " " - ...), whitespace normalised; stories still containing non-ASCII characters or shorter than 50 chars are dropped (counts in `data_processed/main/meta.json`) | keeps the vocabulary at ~95 printable characters instead of hundreds of rare symbols that the model could never learn well |
| Own split | 100,000 train + 10,000 validation stories sampled **without replacement** from the clean pool with my own seed (`split_seed: 266`); raw line indices saved in `data_processed/main/split_indices.npz` | disjoint by construction, exactly reproducible, different from teammates' splits |
| Tokenise | character level; `char_to_idx` / `idx_to_char` built from the **training split only** (`data_processed/main/vocab.json`); `0 = <eos>`, `1 = <unk>` | `<eos>` marks story boundaries so the model learns to end stories; `<unk>` covers a validation char unseen in training (count reported in meta.json) |
| Encode | each story → integer ids + `<eos>`, concatenated into one `uint8` stream per split (`train.bin`, `val.bin`) | compact, whole split fits in GPU memory → no data-loader bottleneck |
| Sequences | fixed-length windows of 256 chars: `x = s[i:i+256]`, `y = s[i+1:i+257]` (`CharStream` in `src/data_prep.py`; examples in `data_processed/main/example_sequences.txt`) | standard next-token objective: every position predicts the following character |

**Epoch definition.** Each epoch cuts the training stream into non-overlapping 256-char windows starting at a random
offset in [0, 256), shuffles them and iterates once — every training character is a prediction target once per epoch
(up to the last incomplete batch), and the window boundaries differ every epoch.

## 2. Model architecture (1.2)

```
ids (B, 256) ─► token embedding (V×320) + learned positional embedding (256×320) ─► dropout
             ─► 6 × [ x = x + MultiHeadCausalSelfAttention(LayerNorm(x))      8 heads × 40 dims
                      x = x + FeedForward(LayerNorm(x)) ]                     320 → 1280 → 320, GELU
             ─► LayerNorm ─► LM head Linear(320 → V)  (weights tied to the token embedding)
             ─► logits (B, 256, V) ─► cross-entropy with targets shifted by one
```

* **Multi-head self-attention**: one fused linear layer produces Q, K, V; reshaped to 8 heads;
  `softmax(QKᵀ / √40 + M) V`; heads concatenated; output projection; dropout on attention weights and output.
* **Causal masking**: `M` is a lower-triangular boolean buffer; scores of future positions are set to −∞ before
  the softmax, so position *t* only attends to positions ≤ *t*. Verified by a test that perturbs future
  characters and checks that earlier logits are bit-identical, and by checking the attention matrix is
  lower-triangular with rows summing to 1.
* **LayerNorm** written by hand (mean/variance over the feature dimension, learnable γ/β, computed in fp32);
  matches `F.layer_norm` to 1e-5 in the tests (used only as a reference there).
* **Residual connections** around both sub-layers (pre-LN layout).
* **Parameter count**: ~7.51 M total (exact value in `metrics_report.csv`; 7.40 M non-embedding).

## 3. Hyperparameters and justification

| Hyperparameter | Value | Justification |
|---|---|---|
| Context length `block_size` | 256 chars | ≈ 45–50 words, i.e. several TinyStories sentences — enough to keep names and the current event in view, while attention cost (∝ T²) stays cheap. |
| Layers / width / heads | 6 / 320 / 8 (head dim 40) | ~7.5 M parameters. 10 epochs × ~88 M training characters (exact count in `data_processed/main/meta.json`) ≈ 0.9 B training tokens ≈ 115 tokens per parameter, so the model is data-rich (low over-fitting risk) yet trains in well under an hour or two on one GPU. The TinyStories paper shows models of only a few million parameters already produce fluent stories. 8 heads let different heads specialise (e.g. previous character, word start, quote matching). |
| FFN expansion | 4× (1280), GELU | standard GPT ratio; GELU is the smooth activation used by GPT-2. |
| Layout | pre-LayerNorm | more stable gradients than post-LN (Xiong et al., 2020); trains reliably with a short warm-up. |
| Weight tying | on | input and output character representations share one matrix (Press & Wolf, 2017); small saving at char level, but a cleaner, more regularised model. |
| Dropout | 0.1 | mild regularisation; with ~115 tokens/parameter heavy dropout would only slow learning. |
| Initialisation | N(0, 0.02); residual output projections scaled by 1/√(2·n_layer) | GPT-2 scheme: keeps the residual-stream variance constant with depth, initial loss ≈ ln V. |
| Optimiser | AdamW, β = (0.9, 0.95), ε = 1e-8, weight decay 0.1 (matrices/embeddings only) | standard for Transformers; β₂ = 0.95 reacts faster to gradient-scale changes; no decay on biases/LayerNorm gains. |
| Peak LR / floor | 6e-4 → 6e-5 | 6e-4 is a well-tested peak for models of this size; the floor at 10% keeps learning in the last epochs. |
| **LR warm-up + schedule** | linear warm-up over the first 2% of steps (~1.1 K), then cosine decay | warm-up avoids large, noisy Adam updates while the second-moment estimates are still poor; cosine decay gives fast early progress and fine convergence at the end. Plot: `outputs/<run_id>/lr_schedule.png`. |
| Batch | 64 × 256 = 16,384 characters per step | good GPU utilisation for a 7.5 M model and smooth gradient estimates; ~5.4 K steps per epoch. |
| Gradient clipping | global L2 norm 1.0 | guards against loss spikes; pre-clip norms logged every step for the stability analysis. |
| Epochs | 10 | the brief's minimum; best checkpoint chosen by validation loss. |
| Precision | bf16 autocast on Ampere+ GPUs (fp16 + GradScaler on older GPUs) | ~2× faster, lower memory; attention softmax, LayerNorm statistics and the loss are computed in fp32. |
| Generation | greedy and temperature 0.8 sampling, 500 new chars, shared prompts | greedy shows the model's mode (and its repetition tendency); T = 0.8 trades a little diversity for coherence. |

## 4. Metrics

All metrics come from the best checkpoint, and are written by `src/evaluate.py` to `metrics_report.csv`.
Definitions are the team's shared ones in `task1_llm/shared_eval/metrics.py`:
perplexity = exp(CE), BPC = CE / ln 2, generalisation gap = val CE − train CE (train CE in eval mode, i.e. no dropout),
Distinct-n = unique / total word n-grams over all sampled continuations, repeated 4-gram rate = 1 − unique/total
word 4-grams within a continuation (averaged), gradient norm = pre-clip global L2 norm.

<!-- METRICS:START -->
_The table is generated automatically from `metrics_report.csv` when the full run finishes._
<!-- METRICS:END -->

## 5. Training curves and stability

![loss curves](outputs/RUN_ID/loss_curves.png)

_After the run (the image link is updated automatically): add 2–3 sentences on
what the curves show (how fast loss falls, where train and validation flatten, the size of the generalisation gap,
and whether there were spikes or NaNs — see `grad_norm.png` and the `loss_spikes` / `nonfinite_steps` metrics)._

## 6. Generated samples

See `outputs/<run_id>/samples.md` (10 shared prompts × greedy + 3 temperature samples).

## 7. How my model differs from my teammates'

| Member | Layers | d_model | Heads | Context | Params | Other differences |
|---|---|---|---|---|---|---|
| Shibin Thomas | 6 | 320 | 8 | 256 | 7.51 M | pre-LN, tied weights, cosine LR 6e-4, 2% warm-up, batch 64 |
| _teammate_ | | | | | | |

## 8. Hardware disclosure

Filled in automatically: the `Hardware` row of `metrics_report.csv` and the `hardware` section of the run manifest
(GPU model, GPU memory, CUDA / cuDNN versions, CPU model) record exactly where the model was trained.

## References
* Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.
* Eldan & Li, *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?*, 2023.
* Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2), 2019.
* Xiong et al., *On Layer Normalization in the Transformer Architecture*, ICML 2020.
* Press & Wolf, *Using the Output Embedding to Improve Language Models*, EACL 2017.
* Li et al., *A Diversity-Promoting Objective Function for Neural Conversation Models* (Distinct-n), NAACL 2016.
