"""Lecture 16: embeddings — token IDs pull rows out of a table (toy sizes, for teaching)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from chotugpt.embedding import Embedding  # noqa: E402

# a toy vocabulary of ten tokens (real tokenizers have tens of thousands)
vocab = ["After", " my", " morning", " chai", ",", " I", " feel", " tea", " fresh", " sleepy"]
stoi = {t: i for i, t in enumerate(vocab)}

tokens = ["After", " my", " morning", " chai", ",", " I", " feel"]
ids = [stoi[t] for t in tokens]
print("ids:", ids)                          # [0, 1, 2, 3, 4, 5, 6]

emb = Embedding(vocab_size=10, dim=4)       # a 10 x 4 table
x = emb(ids)                                # one row per token
print("shape:", x.shape)                    # (7, 4)
chai = stoi[" chai"]                        # 3
print("chai:", emb(chai).round(2))

one_hot = np.eye(10)[chai]                  # 1 at position 3, zeros elsewhere
print("same row:", np.allclose(one_hot @ emb.weight, emb(chai)))   # True
print("table size:", emb.weight.size)       # 10 x 4 = 40
print("GPT-2 small:", 50257 * 768)          # 38597376
