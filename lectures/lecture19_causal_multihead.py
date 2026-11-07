"""Lecture 19: the causal mask and multi-head attention on our sentence.

Illustrative numbers — for teaching only. X and the single-head weights are lecture 18's toy numbers;
the CausalSelfAttention weights are random (training learns real ones in Module 5).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from chotugpt.attention import CausalSelfAttention, attention, causal_mask  # noqa: E402

tokens = ["After", " my", " morning", " chai", ",", " I", " feel"]
X = np.array([[0.2, 0.1, 0.9, 0.0],   # After
              [0.0, 0.8, 0.1, 0.3],   # my
              [0.7, 0.1, 0.8, 0.2],   # morning
              [0.9, 0.2, 0.3, 0.6],   # chai
              [0.0, 0.0, 0.1, 0.1],   # ,
              [0.1, 0.9, 0.0, 0.4],   # I
              [0.6, 0.7, 0.1, 0.9]])  # feel

# 1) one head with lecture 18's weights, now with the causal mask
rng = np.random.default_rng(941)
Wq, Wk, Wv = [rng.uniform(-1, 1, (4, 4)).round(1) for _ in range(3)]
_, w = attention(X @ Wq, X @ Wk, X @ Wv, mask=causal_mask(7))
print("After weights:", w[0].round(2))
print("chai weights :", w[3].round(2))
print("feel weights :", w[6].round(2))

# 2) multi-head causal self-attention: 2 heads of 2 numbers each
attn = CausalSelfAttention(d_model=4, n_head=2, seed=19)
y = attn(X)
print("weights:", attn.weights.shape, "| out:", y.shape)
print("weight on the future:", np.triu(attn.weights, k=1).sum())
