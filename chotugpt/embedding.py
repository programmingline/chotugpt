"""ChotuGPT embeddings.

Lecture 16: the token embedding table — a lookup table of learned vectors,
one row per token ID.
Lecture 17: positional encoding — sinusoidal positions (Vaswani et al., 2017)
and rotary position embedding, RoPE (Su et al., 2021).
"""
import numpy as np


class Embedding:
    """A table of vectors: row i is the vector for token ID i."""

    def __init__(self, vocab_size, dim, seed=16):
        rng = np.random.default_rng(seed)
        self.weight = rng.normal(0.0, 1.0, size=(vocab_size, dim))   # random at the start, learned in training

    def __call__(self, ids):
        return self.weight[np.asarray(ids)]                          # lookup = just pick rows


def positional_encoding(seq_len, dim, base=10000.0):
    """Sinusoidal positions: PE[pos, 2i] = sin(pos / base^(2i/dim)), PE[pos, 2i+1] = cos(same angle)."""
    pos = np.arange(seq_len)[:, None]                 # 0, 1, 2, ... as a column
    two_i = np.arange(0, dim, 2)[None, :]             # 0, 2, 4, ... one per pair of dimensions
    angle = pos / base ** (two_i / dim)               # each pair turns at its own speed
    pe = np.zeros((seq_len, dim))
    pe[:, 0::2] = np.sin(angle)                       # even dimensions: sine
    pe[:, 1::2] = np.cos(angle)                       # odd dimensions: cosine
    return pe


def rope(x, pos, base=10000.0):
    """Rotary position embedding: turn each pair (x[2i], x[2i+1]) by the angle pos * base^(-2i/dim)."""
    x = np.asarray(x, dtype=float)
    dim = x.shape[-1]
    angle = pos * base ** (-np.arange(0, dim, 2) / dim)
    cos, sin = np.cos(angle), np.sin(angle)
    a, b = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = a * cos - b * sin                # a 2D rotation for every pair
    out[..., 1::2] = a * sin + b * cos
    return out


if __name__ == "__main__":
    emb = Embedding(vocab_size=10, dim=4)
    print(emb.weight.shape)                           # (10, 4)
    print(positional_encoding(7, 4).shape)            # (7, 4)
    print(rope([1.0, 0.0], 1).round(2))               # [0.54 0.84]
