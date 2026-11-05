"""Lecture 8: scaled dot-product self-attention — softmax(Q K^T / sqrt(d)) V with three tiny tokens."""
import numpy as np

Q = K = np.array([[1, 0], [0, 1], [1, 1]])
V = np.array([[1, 0], [0, 2], [1, 1]])

scores = Q @ K.T / np.sqrt(2)
weights = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)
out = weights @ V
print(weights.round(2))   # last row: [0.25 0.25 0.5 ]
print(out.round(2))       # [[0.8 0.8] [0.6 1.2] [0.75 1.]]
