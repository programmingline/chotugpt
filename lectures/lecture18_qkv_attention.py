"""Lecture 18: self-attention by hand on our sentence — queries, keys, values, scaling, softmax, weighted sum.

Illustrative numbers — for teaching only. The vectors X stand in for the embeddings + positions
from lectures 16 and 17, and the weight matrices are made up (training learns real ones in Module 5).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from chotugpt.attention import attention, softmax  # noqa: E402

tokens = ["After", " my", " morning", " chai", ",", " I", " feel"]
X = np.array([[0.2, 0.1, 0.9, 0.0],   # After
              [0.0, 0.8, 0.1, 0.3],   # my
              [0.7, 0.1, 0.8, 0.2],   # morning
              [0.9, 0.2, 0.3, 0.6],   # chai
              [0.0, 0.0, 0.1, 0.1],   # ,
              [0.1, 0.9, 0.0, 0.4],   # I
              [0.6, 0.7, 0.1, 0.9]])  # feel

rng = np.random.default_rng(941)
Wq, Wk, Wv = [rng.uniform(-1, 1, (4, 4)).round(1) for _ in range(3)]

Q, K, V = X @ Wq, X @ Wk, X @ Wv     # (7, 4) each
scores = Q @ K.T / np.sqrt(4)        # (7, 7)
out, weights = attention(Q, K, V)

print("feel scores :", scores[6].round(2))
print("feel weights:", weights[6].round(2))
print("sum:", weights[6].sum().round(2), "| out shape:", out.shape)
print("feel new vector:", out[6].round(2))

# why divide by sqrt(d)? bigger scores make softmax spiky
print(softmax(np.array([1.0, 2.0, 3.0])).round(2))
print(softmax(np.array([8.0, 16.0, 24.0])).round(3))
