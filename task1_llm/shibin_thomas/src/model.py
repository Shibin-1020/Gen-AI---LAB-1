"""Task 1.2 -- GPT-style character language model, implemented from scratch.

Only basic tensor building blocks are used (nn.Linear, nn.Embedding, nn.Dropout, GELU,
matmul, softmax). Layer normalisation, multi-head self-attention, causal masking,
the feed-forward network, residual connections and the transformer block are written
by hand -- no nn.MultiheadAttention, nn.Transformer*, nn.LayerNorm or
F.scaled_dot_product_attention.

Architecture (pre-LayerNorm GPT, as in GPT-2):
    idx -> token embedding + learned positional embedding -> dropout
        -> N x [ x = x + MHSA(LN(x));  x = x + FFN(LN(x)) ]
        -> final LN -> LM head (d_model -> vocab_size) -> logits
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 8
    d_model: int = 320
    ffn_mult: int = 4
    dropout: float = 0.1
    bias: bool = True
    tie_weights: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class LayerNorm(nn.Module):
    """y = (x - mean) / sqrt(var + eps) * gamma + beta, statistics over the last dim."""

    def __init__(self, dim: int, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.float()                                   # stable statistics under mixed precision
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        y = (x - mean) * torch.rsqrt(var + self.eps) * self.weight
        if self.bias is not None:
            y = y + self.bias
        return y.to(in_dtype)


class CausalSelfAttention(nn.Module):
    """Multi-head scaled dot-product self-attention with a causal (lower-triangular) mask."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0, "d_model must be divisible by n_head"
        self.n_head, self.head_dim = cfg.n_head, cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)   # fused Q, K, V projections
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)      # output projection W_O
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, cfg.block_size, cfg.block_size), persistent=False)
        self.last_attn: torch.Tensor | None = None       # filled only when store_attn=True
        self.store_attn = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)          # (B, h, T, T)
        scores = scores.masked_fill(~self.causal_mask[:, :, :T, :T], float("-inf"))  # no peeking ahead
        attn = F.softmax(scores.float(), dim=-1).to(q.dtype)
        if self.store_attn:
            self.last_attn = attn.detach()
        attn = self.attn_drop(attn)

        y = attn @ v                                                             # (B, h, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)                         # concat heads
        return self.resid_drop(self.proj(y))


class FeedForward(nn.Module):
    """Position-wise MLP: Linear(d, m*d) -> GELU -> Linear(m*d, d) -> dropout."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        hidden = cfg.ffn_mult * cfg.d_model
        self.fc1 = nn.Linear(cfg.d_model, hidden, bias=cfg.bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, cfg.d_model, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Pre-LN block with residual connections around attention and the FFN."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = LayerNorm(cfg.d_model, cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = LayerNorm(cfg.d_model, cfg.bias)
        self.ffn = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPTCharLM(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)      # learnable token embeddings
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)      # learnable positional embeddings
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = LayerNorm(cfg.d_model, cfg.bias)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)  # hidden -> vocab logits
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # GPT-2: scale residual-branch output projections by 1/sqrt(2 * n_layer)
        for name, p in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("ffn.fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())           # tied weights counted once
        if non_embedding:
            n -= self.pos_emb.weight.numel() + self.tok_emb.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None, greedy: bool = False, eos_id: int | None = None,
                 generator: torch.Generator | None = None) -> torch.Tensor:
        """Autoregressive decoding: greedy (argmax) or temperature (+ optional top-k) sampling."""
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.block_size:]                  # crop to the context window
            logits, _ = self(ctx)
            logits = logits[:, -1, :].float()
            if greedy:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / max(temperature, 1e-6)
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1, generator=generator)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and bool((nxt == eos_id).all()):
                break
        return idx
