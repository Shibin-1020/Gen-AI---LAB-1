"""Task 2.2 -- my three sentiment classifiers. All embeddings are learned from scratch
(nn.Embedding, random N(0, 1) init, <pad> row fixed at zero); no pretrained vectors or language models.

Every model maps token ids (B, T) + lengths (B,) to ONE logit per review; P(positive) = sigmoid(logit).

* MeanPoolMLP   (baseline)      bag-of-embeddings: masked mean over tokens -> MLP. Order-insensitive.
* TextCNN       (experiment 1)  1-D convolutions of width 3/4/5 over the embeddings + max-over-time
                                pooling (Kim, 2014): detects local n-gram phrases such as "not good".
* BiGRUAttention (experiment 2) 2-layer bidirectional GRU + additive attention pooling
                                (Bahdanau et al., 2015; Yang et al., 2016): reads the whole sequence in
                                order and learns which positions carry the sentiment.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def _mask(lengths: torch.Tensor, T: int) -> torch.Tensor:
    return torch.arange(T, device=lengths.device)[None, :] < lengths[:, None]       # (B, T) True = real token


class MeanPoolMLP(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 64, dropout: float = 0.3, **_):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x, lengths):
        m = _mask(lengths, x.size(1)).unsqueeze(-1).float()
        e = self.emb(x) * m
        pooled = e.sum(1) / lengths.clamp(min=1).unsqueeze(1).float()               # masked mean
        return self.fc2(self.drop(F.relu(self.fc1(self.drop(pooled))))).squeeze(-1)


class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, kernel_sizes=(3, 4, 5), num_filters: int = 128,
                 dropout: float = 0.5, **_):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, k, padding=k // 2) for k in kernel_sizes])
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), 1)

    def forward(self, x, lengths):
        T = x.size(1)
        valid = _mask(lengths, T)
        e = self.emb(x).transpose(1, 2)                                              # (B, E, T)
        feats = []
        for conv in self.convs:
            h = F.relu(conv(e))[:, :, :T]                                            # (B, F, T)
            h = h.masked_fill(~valid[:, None, :], float("-inf"))                     # ignore padding positions
            feats.append(h.max(dim=2).values)                                        # max-over-time
        return self.fc(self.drop(torch.cat(feats, dim=1))).squeeze(-1)


class BiGRUAttention(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 128, num_layers: int = 2,
                 attn_dim: int = 128, dropout: float = 0.3, **_):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.emb_drop = nn.Dropout(dropout)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.attn_proj = nn.Linear(2 * hidden_dim, attn_dim)
        self.attn_v = nn.Linear(attn_dim, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden_dim, 1)
        self.last_attention: torch.Tensor | None = None

    def forward(self, x, lengths):
        T = x.size(1)
        e = self.emb_drop(self.emb(x))
        packed = pack_padded_sequence(e, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        h, _ = pad_packed_sequence(out, batch_first=True, total_length=T)            # (B, T, 2H)
        scores = self.attn_v(torch.tanh(self.attn_proj(h))).squeeze(-1)             # (B, T)
        scores = scores.masked_fill(~_mask(lengths, T), float("-inf"))
        alpha = torch.softmax(scores, dim=1)
        self.last_attention = alpha.detach()
        context = (alpha.unsqueeze(-1) * h).sum(1)                                   # attention-weighted sum
        return self.fc(self.drop(context)).squeeze(-1)


MODELS = {"meanpool_mlp": MeanPoolMLP, "textcnn": TextCNN, "bigru_attn": BiGRUAttention}


def build_model(mcfg: dict, vocab_size: int) -> nn.Module:
    kwargs = {k: v for k, v in mcfg.items() if k != "type"}
    if "kernel_sizes" in kwargs:
        kwargs["kernel_sizes"] = tuple(kwargs["kernel_sizes"])
    return MODELS[mcfg["type"]](vocab_size=vocab_size, **kwargs)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
