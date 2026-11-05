"""Lecture 7: attention weights — softmax turns relevance scores into weights that sum to 1 (illustrative scores)."""
import numpy as np

words = ["After", "my", "morning", "chai", ",", "I", "feel"]
scores = np.array([0.2, 0.5, 1.8, 3.0, 0.1, 0.6, 1.2])

weights = np.exp(scores) / np.exp(scores).sum()
print(weights.round(2))   # [0.03 0.05 0.17 0.57 0.03 0.05 0.09]
for w, a in sorted(zip(words, weights), key=lambda x: -x[1])[:3]:
    print(f"{w:8} {a:.2f}")
