"""Lecture 12: a neural network from scratch, in NumPy (toy numbers, for teaching).

Run from the repo root:  python lectures/lecture12_neural_network.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from chotugpt.nn import Linear, relu

x = np.array([0.9, 0.8, 0.1])              # chai vector
w, b = np.array([0.5, 0.4, -0.6]), 0.1     # one neuron
print("neuron:", round(x @ w + b, 2), relu(-2.0))

rng = np.random.default_rng(12)
layer1, layer2 = Linear(3, 4, rng), Linear(4, 2, rng)
h = relu(layer1(x))
print("hidden:", h.round(2))
print("scores:", layer2(h).round(2))       # fresh, sleepy
