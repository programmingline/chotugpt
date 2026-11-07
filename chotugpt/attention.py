"""ChotuGPT attention (NumPy).

Lecture 18: scaled dot-product attention, softmax(Q K^T / sqrt(d)) V.
Lecture 19: the causal mask (no peeking at future tokens) and multi-head attention
            in the CausalSelfAttention class.
"""
import numpy as np


def softmax(x, axis=-1):
    """Turn scores into weights that are positive and sum to 1 along `axis`."""
    e = np.exp(x - x.max(axis=axis, keepdims=True))   # subtract the max for numerical safety
    return e / e.sum(axis=axis, keepdims=True)


def causal_mask(T):
    """(T, T) lower triangle of True: token i may look at tokens 0..i, never later ones."""
    return np.tril(np.ones((T, T), dtype=bool))


def attention(Q, K, V, mask=None):
    """Every query is compared with every key; the weights mix the values.

    Q, K, V: arrays of shape (..., T, d). mask: (T, T) booleans, False = not allowed.
    Returns (out, weights) with shapes (..., T, d) and (..., T, T).
    """
    d = Q.shape[-1]
    scores = Q @ np.swapaxes(K, -1, -2) / np.sqrt(d)   # row = who is looking, column = who is looked at
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)        # the future gets minus infinity ...
    weights = softmax(scores, axis=-1)                  # ... so softmax gives it exactly 0
    return weights @ V, weights


class CausalSelfAttention:
    """Multi-head causal self-attention: heads run in parallel, then concatenate and project."""

    def __init__(self, d_model, n_head, seed=0):
        assert d_model % n_head == 0, "d_model must split evenly into heads"
        self.n_head, self.d_head = n_head, d_model // n_head
        rng = np.random.default_rng(seed)
        s = 1 / np.sqrt(d_model)
        self.Wq, self.Wk, self.Wv, self.Wo = [
            rng.normal(0, s, (d_model, d_model)) for _ in range(4)]

    def split_heads(self, x):                  # (T, d_model) -> (n_head, T, d_head)
        T = x.shape[0]
        return x.reshape(T, self.n_head, self.d_head).transpose(1, 0, 2)

    def __call__(self, x):                     # x: (T, d_model)
        T, d_model = x.shape
        q = self.split_heads(x @ self.Wq)      # (n_head, T, d_head)
        k = self.split_heads(x @ self.Wk)
        v = self.split_heads(x @ self.Wv)
        mask = causal_mask(T)
        out, self.weights = attention(q, k, v, mask)        # all heads at once
        out = out.transpose(1, 0, 2).reshape(T, d_model)    # join the heads
        return out @ self.Wo                                # output projection


if __name__ == "__main__":
    x = np.random.default_rng(0).normal(size=(5, 8))
    attn = CausalSelfAttention(d_model=8, n_head=2)
    y = attn(x)
    print(y.shape, attn.weights.shape)        # (5, 8) (2, 5, 5)
    print(attn.weights[0].round(2))           # zeros above the diagonal
