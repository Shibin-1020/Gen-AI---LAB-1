"""Fast CPU smoke test for Task 1's from-scratch GPT implementation.

This test uses synthetic character data, so it does not download TinyStories or
write training artifacts. It checks the model shape, causal mask, loss,
backpropagation, finite gradients, and autoregressive generation.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import yaml

from train import GPTFromScratch


def main() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    cfg = yaml.safe_load(config_path.read_text())
    tiny = dict(cfg)
    tiny.update({"embedding_dim": 32, "num_heads": 4, "num_layers": 2, "feedforward_dim": 64, "sequence_length": 16, "dropout": 0.0})
    vocab_size, batch_size, length = 17, 2, tiny["sequence_length"]
    model = GPTFromScratch(vocab_size, tiny).cpu()
    x = torch.randint(0, vocab_size, (batch_size, length))
    y = torch.randint(0, vocab_size, (batch_size, length))
    logits, loss = model(x, y)
    assert logits.shape == (batch_size, length, vocab_size)
    assert torch.isfinite(loss).item(), "non-finite cross-entropy loss"
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all().item() for g in gradients), "non-finite gradients"
    mask = model.blocks[0].attn.mask
    expected_mask = torch.tril(torch.ones_like(mask, dtype=torch.bool))
    assert mask.shape == (length, length) and bool(torch.equal(mask, expected_mask)), "invalid causal mask"
    assert not bool(torch.triu(mask, diagonal=1).any()), "causal mask exposes future tokens"
    with torch.no_grad():
        generated = model.generate(x[:, :3], max_new_tokens=5, temperature=0.9)
    assert generated.shape == (batch_size, 8)
    assert math.isfinite(float(generated.float().mean()))
    print("PASS: Task 1 smoke test; from-scratch GPT forward/backward, causal mask, and generation are valid.")


if __name__ == "__main__":
    main()
