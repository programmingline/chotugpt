"""ChotuGPT: the full GPT model, in PyTorch.

Lecture 21: the whole architecture in one file.

Why PyTorch now? Up to lecture 20 we wrote every piece in NumPy so nothing was hidden.
But training means backprop (lecture 13): the chain rule through every layer, by hand.
For a model with a million weights that is a lot of error-prone code. PyTorch records every
operation in the forward pass and runs the backward pass for us — `loss.backward()` —
so we keep exactly the same ideas and let the library do the calculus.

    token ids -> token embedding + position embedding
              -> n_layer x Block( x + attn(ln1(x)) ; x + mlp(ln2(x)) )
              -> final LayerNorm -> lm_head -> logits (a score for every token in the vocabulary)
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 512      # 256 bytes + 256 BPE merges (lecture 23)
    block_size: int = 256      # the longest context the model can see, in tokens
    n_layer: int = 4           # how many Blocks are stacked
    n_head: int = 4            # attention heads per Block
    n_embd: int = 128          # numbers per token (the width of the model)
    dropout: float = 0.0       # randomly zero some activations while training (0 = off)


class CausalSelfAttention(nn.Module):
    """Lecture 19's multi-head causal attention, now in PyTorch."""

    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head, self.d_head = cfg.n_head, cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)     # Wq, Wk, Wv in one matrix
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)        # Wo: mixes the heads back together
        self.drop = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)  # True = allowed to look

    def forward(self, x):                                    # x: (B, T, C)
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, d_head): every head works on its own slice
        q, k, v = [t.view(B, T, self.n_head, self.d_head).transpose(1, 2) for t in (q, k, v)]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)        # (B, n_head, T, T)
        scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))   # no peeking at the future
        weights = self.drop(F.softmax(scores, dim=-1))
        out = weights @ v                                                # (B, n_head, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, C)             # join the heads
        return self.drop(self.proj(out))


class MLP(nn.Module):
    """Lecture 20's feed-forward network: widen 4x, GELU, shrink back."""

    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Lecture 20's pre-norm transformer block."""

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))     # tokens talk to each other ...
        x = x + self.mlp(self.ln2(x))      # ... then each token thinks on its own
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)   # lecture 16: one row per token
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)   # lecture 17: one learned row per position
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight                 # weight tying: same table in and out
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Small random starting weights (GPT-2 style), zero biases."""
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        """idx: (B, T) token ids. Returns logits (B, T, vocab_size) and, if targets are given, the loss."""
        B, T = idx.shape
        assert T <= self.cfg.block_size, "sequence longer than block_size"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))      # (B, T, C)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))                       # (B, T, vocab_size)
        loss = None
        if targets is not None:                                   # lecture 14: cross-entropy
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())         # tied weights are counted once
