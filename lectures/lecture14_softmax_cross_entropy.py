"""Lecture 14: softmax turns scores into probabilities; cross-entropy measures surprise."""
import numpy as np
words = ["fresh", "tired", "sleepy"]     # "After my morning chai, I feel ___"
scores = np.array([2.0, 1.0, 0.1])       # logits: raw scores from the model

def softmax(z):
    e = np.exp(z)                         # step 1: every score becomes positive
    return e / e.sum()                    # step 2: divide by the total -> adds up to 1

def cross_entropy(p, target):
    return -np.log(p[target])             # surprise: minus log of the correct word's probability

p = softmax(scores)
print(p.round(2))                         # [0.66 0.24 0.1 ]
print(cross_entropy(p, 0).round(2))  # fresh   -> 0.42
print(cross_entropy(p, 2).round(2))  # sleepy  -> 2.32

grad = p.copy(); grad[0] -= 1             # gradient w.r.t. the scores: p minus one-hot
print(grad.round(2))                      # [-0.34  0.24  0.1 ]
