"""ChotuGPT embeddings.

Lecture 16: the token embedding table — a lookup table of learned vectors,
one row per token ID. (Lecture 17 adds positional encoding.)
"""
import numpy as np


class Embedding:
    """A table of vectors: row i is the vector for token ID i."""

    def __init__(self, vocab_size, dim, seed=16):
        rng = np.random.default_rng(seed)
        self.weight = rng.normal(0.0, 1.0, size=(vocab_size, dim))   # random at the start, learned in training

    def __call__(self, ids):
        return self.weight[np.asarray(ids)]                          # lookup = just pick rows


if __name__ == "__main__":
    emb = Embedding(vocab_size=10, dim=4)
    print(emb.weight.shape)          # (10, 4)
    print(emb([0, 3]).shape)         # (2, 4)
