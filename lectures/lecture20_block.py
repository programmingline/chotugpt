"""Lecture 20: feed-forward (MLP), residual connections and LayerNorm — one transformer Block.

Illustrative numbers — for teaching only: the weights are random (training learns real ones in lecture 24).
This is our last NumPy lecture; from lecture 21 on we use PyTorch.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from chotugpt.block import Block, layer_norm  # noqa: E402
from chotugpt.embedding import Embedding, positional_encoding  # noqa: E402
from chotugpt.nn import gelu  # noqa: E402

np.set_printoptions(precision=2, suppress=True)
tokens = ["After", " my", " morning", " chai", ",", " I", " feel"]
ids = list(range(7))                                   # lecture 16's toy vocabulary: IDs 0..6

d_model, n_head = 8, 2
x = Embedding(vocab_size=10, dim=d_model)(ids) + positional_encoding(7, d_model)   # lectures 16-17
print("input x:", x.shape)                             # 7 tokens, 8 numbers each

# 1) LayerNorm: every token's vector gets mean 0 and standard deviation 1
chai = x[3]
print("chai before LayerNorm:", chai, "| mean %.2f std %.2f" % (chai.mean(), chai.std()))
n = layer_norm(x)
print("chai after  LayerNorm:", n[3], "| mean %.2f std %.2f" % (n[3].mean(), n[3].std()))
print("all tokens, std before:", x.std(axis=-1))
print("all tokens, std after: ", n.std(axis=-1))

# 2) GELU inside the MLP: smooth, lets small negatives through a little
print("gelu([-2, -0.5, 0, 0.5, 2]):", gelu(np.array([-2, -0.5, 0, 0.5, 2.0])))

# 3) one Block, step by step
block = Block(d_model, n_head)
a = block.attn(block.ln1(x))                           # what attention adds
h = x + a                                              # residual 1
hidden = block.mlp.fc(block.ln2(h))                    # the MLP widens 8 -> 32 ...
m = block.mlp(block.ln2(h))                            # ... and shrinks back to 8
out = h + m                                            # residual 2
print("ln1(x):", block.ln1(x).shape, "| attn:", a.shape, "| mlp hidden:", hidden.shape,
      "| mlp out:", m.shape, "| block out:", out.shape)
print("same as block(x):", np.allclose(out, block(x)))

# 4) the residual keeps the input: out = x + (attention's change) + (MLP's change)
print("out - x == attn + mlp:", np.allclose(out - x, a + m))
size = lambda v: np.linalg.norm(v, axis=-1)            # the length of each token's vector
print("length of x      :", size(x))
print("length of changes:", size(a + m))
cos = (x * out).sum(-1) / (size(x) * size(out))
print("cosine(x, out)   :", cos)                       # close to 1: the output still points like the input
print("changes / x (all tokens): %.0f%%" % (100 * np.linalg.norm(a + m) / np.linalg.norm(x)))

# 5) how many weights does one Block have?
counts = {"attn": 4 * d_model * d_model,
          "mlp": (d_model * 4 * d_model + 4 * d_model) + (4 * d_model * d_model + d_model),
          "layernorms": 2 * 2 * d_model}
print("weights:", counts, "total:", sum(counts.values()))
