"""Sanity tests for the from-scratch GPT. Run:  python task1_llm/shibin_thomas/tests/test_model.py
(also works with pytest). No data or GPU needed."""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from data_prep import EOS_ID, build_vocab, clean_story, decode, encode  # noqa: E402
from model import GPTCharLM, GPTConfig, LayerNorm  # noqa: E402
from train import lr_at  # noqa: E402

CFG = GPTConfig(vocab_size=50, block_size=32, n_layer=2, n_head=4, d_model=32, dropout=0.0)


def test_no_prebuilt_transformer_modules():
    src = (SRC / "model.py").read_text()
    code = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("#"))
    code = re.sub(r'""".*?"""', "", code, flags=re.S)          # ignore docstrings
    for banned in ["MultiheadAttention", "nn.Transformer", "scaled_dot_product_attention",
                   "nn.LayerNorm", "F.layer_norm", "flash_attn"]:
        assert banned not in code, f"prebuilt module used: {banned}"


def test_layernorm_matches_reference():
    torch.manual_seed(0)
    ln = LayerNorm(16)
    with torch.no_grad():
        ln.weight.uniform_(0.5, 1.5)
        ln.bias.uniform_(-0.5, 0.5)
    x = torch.randn(4, 7, 16)
    ref = F.layer_norm(x, (16,), ln.weight, ln.bias, eps=1e-5)    # reference only, not used in the model
    assert torch.allclose(ln(x), ref, atol=1e-5)


def test_shapes_and_initial_loss():
    torch.manual_seed(0)
    m = GPTCharLM(CFG)
    x = torch.randint(0, CFG.vocab_size, (3, 20))
    y = torch.randint(0, CFG.vocab_size, (3, 20))
    logits, loss = m(x, y)
    assert logits.shape == (3, 20, CFG.vocab_size)
    assert abs(loss.item() - math.log(CFG.vocab_size)) < 0.3     # ~uniform prediction at init


def test_causal_mask_blocks_future():
    """Changing token t must not change the logits at positions < t."""
    torch.manual_seed(0)
    m = GPTCharLM(CFG).eval()
    x = torch.randint(0, CFG.vocab_size, (1, 24))
    x2 = x.clone()
    x2[0, 15:] = (x2[0, 15:] + 7) % CFG.vocab_size
    l1, _ = m(x)
    l2, _ = m(x2)
    assert torch.allclose(l1[0, :15], l2[0, :15], atol=1e-6)
    assert not torch.allclose(l1[0, 15:], l2[0, 15:])


def test_attention_weights_are_lower_triangular():
    m = GPTCharLM(CFG).eval()
    attn = m.blocks[0].attn
    attn.store_attn = True
    m(torch.randint(0, CFG.vocab_size, (2, 10)))
    a = attn.last_attn                                             # (B, h, T, T)
    assert torch.all(a.triu(diagonal=1) == 0)
    assert torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1)), atol=1e-5)


def test_weight_tying_and_param_count():
    m = GPTCharLM(CFG)
    assert m.lm_head.weight.data_ptr() == m.tok_emb.weight.data_ptr()
    assert m.num_params() == sum(p.numel() for p in m.parameters())


def test_overfits_tiny_batch():
    torch.manual_seed(0)
    m = GPTCharLM(CFG)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    x = torch.randint(0, CFG.vocab_size, (4, 17))
    for _ in range(150):
        _, loss = m(x[:, :-1], x[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.3, loss.item()


def test_generate_greedy_and_sampling():
    m = GPTCharLM(CFG).eval()
    idx = torch.zeros(1, 3, dtype=torch.long) + 5
    assert m.generate(idx, 40, greedy=True).shape == (1, 43)       # longer than block_size: context cropping
    g = torch.Generator().manual_seed(0)
    assert m.generate(idx, 10, temperature=0.8, top_k=5, generator=g).shape == (1, 13)


def test_vocab_roundtrip_and_cleaning():
    text, status = clean_story("Lily’s  ball — “yay”…", min_chars=5)
    assert status == "ok" and text == "Lily's ball - \"yay\"..."
    assert clean_story("short", 50)[1] == "too_short"
    assert clean_story("x" * 60 + "中", 50)[1] == "non_ascii"
    c2i, i2c, _ = build_vocab(["hello world"])
    assert c2i["<eos>"] == EOS_ID and decode(encode("hello", c2i), i2c) == "hello"


def test_lr_schedule():
    lrs = [lr_at(s, 1000, 100, 1e-3, 1e-4, "cosine") for s in range(1000)]
    assert lrs[0] == 1e-5 and abs(lrs[99] - 1e-3) < 1e-12           # linear warm-up
    assert all(a >= b for a, b in zip(lrs[100:], lrs[101:]))        # monotone decay
    assert abs(lrs[-1] - 1e-4) < 1e-6


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} tests passed")
