# Task 1 results — Denisha Ketan Tank

## Model and training setup

This run trains a custom character-level GPT language model from scratch on TinyStories. It uses a 4-layer Transformer decoder with 4-head causal self-attention, `d_model=256`, head dimension 64, feed-forward dimension 1024, context length 128, learned token and position embeddings, pre-layer normalization, residual connections, GELU, dropout 0.1, AdamW, peak learning rate 0.0005, 500-step warm-up, cosine decay, gradient clipping at 1.0, and mixed precision on a Tesla T4.

The tokenizer is an explicit character vocabulary built from the selected stories. Training uses 100,000 stories and validation uses 10,000 stories with next-character cross-entropy.

## Final metrics

| Metric | Value |
|---|---:|
| Train cross-entropy | 0.712783 |
| Validation cross-entropy | 0.682218 |
| Perplexity | 1.978261 |
| Bits per character | 0.984233 |
| Generalization gap | -0.030565 |
| Top-1 next-character accuracy | 0.782190 |
| Final gradient norm | 0.231820 |
| Maximum epoch gradient norm | 0.287184 |
| Distinct-1 / 2 / 3 | 0.108025 / 0.458204 / 0.736025 |
| Repeated 4-gram rate | 0.152648 |
| Parameters | 3250688 |
| Training tokens/sec | 224657.931134 |
| Generation tokens/sec | 268.036336 |
| Peak GPU memory (MB) | 408.318848 |
| Training time (seconds) | 3995.287927 |
| Device | cuda |
| Vocabulary size | 114 |
| Gradient NaN count | 0 |

## Interpretation

The validation loss and perplexity measure next-character prediction quality; lower is better. Top-1 accuracy measures how often the most likely predicted character matches the target. The generalization gap is validation loss minus training loss. The diversity metrics measure generated-text variety, while repeated 4-gram rate measures local repetition.
