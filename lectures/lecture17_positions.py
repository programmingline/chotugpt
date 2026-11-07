"""Lecture 17: positional encoding — sinusoidal positions and RoPE (toy sizes, for teaching)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from chotugpt.embedding import Embedding, positional_encoding, rope  # noqa: E402

emb = Embedding(vocab_size=10, dim=4)            # same toy table as lecture 16
ids = [0, 1, 2, 3, 4, 5, 6]                      # After my morning chai , I feel
pe = positional_encoding(len(ids), 4)
print(pe[:4].round(2))                           # positions 0 to 3

x = emb(ids) + pe                                # add position, number by number
print("shape:", x.shape)                         # (7, 4)

chai = 3
print("chai at 0:", (emb(chai) + pe[0]).round(2))   # "chai morning"
print("chai at 1:", (emb(chai) + pe[1]).round(2))   # "morning chai"

# RoPE: rotate a query and a key by their positions
q, k = np.array([0.9, 0.8]), np.array([0.8, 0.9])
for m, n in [(1, 0), (5, 4), (9, 8)]:
    print(m, n, round(rope(q, m) @ rope(k, n), 3))   # same distance -> same score
