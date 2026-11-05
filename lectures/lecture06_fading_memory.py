"""Lecture 6: why simple RNNs forget — if each step keeps half of the past, old words fade exponentially."""
words = ["After", "my", "morning", "chai", ",", "I", "feel"]

for i, w in enumerate(words):
    steps = len(words) - 1 - i
    influence = 0.5 ** steps
    print(f"{w:8} {influence:.3f}")
# After    0.016 ... chai 0.125 ... feel 1.000
