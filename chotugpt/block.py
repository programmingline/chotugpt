"""ChotuGPT transformer block (NumPy) — the last NumPy piece before we switch to PyTorch.

Lecture 20: the feed-forward network (MLP), residual connections and LayerNorm,
put together into one pre-norm transformer Block:

    x = x + attn(ln1(x))     # tokens look at each other (lecture 19), then add back the input
    x = x + mlp(ln2(x))      # each token thinks on its own,             then add back again
"""
import numpy as np

from chotugpt.attention import CausalSelfAttention
from chotugpt.nn import Linear, gelu


def layer_norm(x, gamma=None, beta=None, eps=1e-5):
    """Rescale every token's vector to mean 0 and standard deviation 1,
    then (optionally) stretch by gamma and shift by beta — both learned in training."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    out = (x - mean) / np.sqrt(var + eps)        # eps: never divide by zero
    if gamma is not None:
        out = out * gamma
    if beta is not None:
        out = out + beta
    return out


class LayerNorm:
    """layer_norm with its own gamma (starts at 1) and beta (starts at 0)."""

    def __init__(self, dim):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)

    def __call__(self, x):
        return layer_norm(x, self.gamma, self.beta)


class MLP:
    """Feed-forward network: widen to 4x, GELU, shrink back. Applied to every token separately."""

    def __init__(self, d_model, rng=None):
        rng = rng or np.random.default_rng(20)
        self.fc = Linear(d_model, 4 * d_model, rng)      # e.g. 8 -> 32
        self.proj = Linear(4 * d_model, d_model, rng)    # 32 -> 8

    def __call__(self, x):
        return self.proj(gelu(self.fc(x)))


class Block:
    """One pre-norm transformer block: LayerNorm before each part, a residual around each part."""

    def __init__(self, d_model, n_head, seed=20):
        self.ln1 = LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, seed=seed)
        self.ln2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, np.random.default_rng(seed + 1))

    def __call__(self, x):                        # x: (T, d_model)
        x = x + self.attn(self.ln1(x))            # residual 1: keep x, add what attention found
        x = x + self.mlp(self.ln2(x))             # residual 2: keep that, add what the MLP found
        return x


if __name__ == "__main__":
    x = np.random.default_rng(0).normal(size=(7, 8))
    block = Block(d_model=8, n_head=2)
    print(block(x).shape)                         # (7, 8): same shape in, same shape out
    print(layer_norm(x).mean(axis=-1).round(6))   # every row now has mean 0
