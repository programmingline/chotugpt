"""Lecture 13: a single neuron learns a pattern with gradient descent and backprop (toy numbers, for teaching).

Run from the repo root:  python lectures/lecture13_gradient_descent.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from chotugpt.nn import Linear, mse_loss

X = np.array([[0.9, 0.8, 0.1],      # chai -> 1
              [0.8, 0.9, 0.3],      # tea  -> 1
              [0.1, 0.2, 0.95]])    # car  -> 0
y = np.array([[1.0], [1.0], [0.0]])

layer = Linear(3, 1, np.random.default_rng(13))
for step in range(101):
    pred = layer(X)                     # forward
    loss, grad = mse_loss(pred, y)      # loss + slope
    layer.backward(X, grad)             # chain rule
    layer.step(lr=0.1)                  # downhill
    if step % 20 == 0:
        print("step", step, "loss", round(loss, 4))
print("predictions:", layer(X).ravel().round(2))
