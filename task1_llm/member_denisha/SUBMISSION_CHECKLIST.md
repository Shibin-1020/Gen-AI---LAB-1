# Task 1 submission checklist — Denisha Ketan Tank

This folder contains the final Task 1 implementation and evidence from the completed GPU run.

## Requirement coverage

- Character-level tokenization with explicit `char_to_idx` and `idx_to_char`: implemented in `src/train.py`.
- Fixed-length autoregressive input-target sequences: implemented by `CharDataset` with `sequence_length: 128` and `stride: 128`.
- Independent split of 100,000 training stories and 10,000 validation stories: configured in `config.yaml`.
- GPT-style Transformer blocks from scratch: causal multi-head self-attention, layer normalization, feed-forward network, residual connections, learned token/position embeddings, and language-model head are implemented directly in `src/train.py`.
- No prebuilt Transformer or attention module: the model uses custom `CausalSelfAttention` and `TransformerBlock` classes.
- Cross-entropy training, warm-up, cosine scheduling, gradient clipping, and minimum 10 epochs: configured in `config.yaml` and recorded in the raw log.
- Text generation: temperature samples at 0.7, 0.9, and 1.1 are stored in `outputs/samples.json` and `outputs/sample.txt`.
- Three failure cases: documented in `failure_analysis.md` using the generated samples from this run.

## Final run summary

| Metric | Value |
|---|---:|
| Validation cross-entropy | 0.682218 |
| Validation top-1 character accuracy | 0.782190 |
| Perplexity | 1.978261 |
| Bits per character | 0.984233 |
| Parameters | 3,250,688 |
| Training time | 3,995.29 seconds |
| Training throughput | 224,657.93 tokens/sec |
| Generation throughput | 268.04 tokens/sec |
| Peak GPU memory | 408.32 MB |
| Gradient NaN count | 0 |
| Hardware | Tesla T4 |

## Evidence map

- `config.yaml`: reproducible hyperparameters.
- `src/train.py`: model, preprocessing, training, evaluation, generation, and metrics.
- `src/smoke_test.py`: fast CPU smoke test for forward/backward, causal masking, finite gradients, and generation.
- `Task1_GPT_Colab.ipynb`: portable VSCode/Colab execution notebook.
- `outputs/metrics.csv`: final metric table.
- `outputs/history.json`: all 10 epoch records.
- `outputs/loss_curves.png`: training/validation loss plot.
- `outputs/samples.json`, `outputs/sample.txt`: generated text samples.
- `results.md`: final written results summary.
- `failure_analysis.md`: three observed generation failures and testable improvements.
- `checkpoints/gpt_from_scratch.pt`: final model checkpoint for demo/reload.
- `checkpoints/gpt_from_scratch_latest.pt`: resumable checkpoint with optimizer/scheduler state.
- `../../reproducibility/raw_logs/task1_denisha.log`: unedited training evidence.
- `../../reproducibility/manifests/task1_denisha.json`: environment and checkpoint mapping.

## Reproduction

From the repository root:

```bash
python task1_llm/member_denisha/run_task1.py --smoke-only
python task1_llm/member_denisha/run_task1.py
```

The full command downloads the approved TinyStories train split if absent, resumes from the latest checkpoint, regenerates the report and metrics, writes a manifest, and validates the required artifacts.
