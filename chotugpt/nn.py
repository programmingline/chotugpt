"""ChotuGPT neural-network building blocks (NumPy only).

Lecture 12: a Linear layer and the ReLU / GELU activations.
"""
import numpy as np


class Linear:
    """A layer of neurons: y = x @ W + b."""

    def __init__(self, n_in, n_out, rng=None):
        rng = rng or np.random.default_rng(0)
        self.W = rng.normal(0.0, 1.0 / np.sqrt(n_in), size=(n_in, n_out))   # one column per neuron
        self.b = np.zeros(n_out)                                              # one bias per neuron

    def __call__(self, x):
        return x @ self.W + self.b


def relu(x):
    """Keep positive numbers, turn negative numbers into zero."""
    return np.maximum(0, x)


def gelu(x):
    """A smoother cousin of ReLU, used in GPT models (tanh approximation)."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


if __name__ == "__main__":
    print(relu(np.array([0.81, -2.0])))   # [0.81 0.  ]
