"""ChotuGPT attention (NumPy).

Lecture 18: scaled dot-product attention, softmax(Q K^T / sqrt(d)) V.
(Lecture 19 adds the causal mask and multi-head attention.)
"""
import numpy as np


def softmax(x, axis=-1):
    """Turn scores into weights that are positive and sum to 1 along `axis`."""
    e = np.exp(x - x.max(axis=axis, keepdims=True))   # subtract the max for numerical safety
    return e / e.sum(axis=axis, keepdims=True)


def attention(Q, K, V):
    """Every query is compared with every key; the weights mix the values.

    Q, K, V: arrays of shape (T, d). Returns (out, weights) with shapes (T, d) and (T, T).
    """
    d = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d)      # (T, T): row = who is looking, column = who is looked at
    weights = softmax(scores, axis=-1)  # each row sums to 1
    return weights @ V, weights


if __name__ == "__main__":
    Q = K = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    V = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    out, w = attention(Q, K, V)
    print(w.round(2))
    print(out.round(2))
