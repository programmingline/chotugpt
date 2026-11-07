"""ChotuGPT neural-network building blocks (NumPy only).

Lecture 12: a Linear layer and the ReLU / GELU activations.
Lecture 13: backward passes (the chain rule) and a gradient-descent step.
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

    def backward(self, x, grad_out):
        """Chain rule: from the loss's slope w.r.t. our output, get the slopes for W, b and our input."""
        self.dW = x.T @ grad_out
        self.db = grad_out.sum(axis=0)
        return grad_out @ self.W.T

    def step(self, lr):
        """Gradient descent: move every weight a little, opposite to its gradient."""
        self.W -= lr * self.dW
        self.b -= lr * self.db


def relu(x):
    """Keep positive numbers, turn negative numbers into zero."""
    return np.maximum(0, x)


def relu_backward(x, grad_out):
    """ReLU passes the gradient through where x was positive, and blocks it where x was negative."""
    return grad_out * (x > 0)


def gelu(x):
    """A smoother cousin of ReLU, used in GPT models (tanh approximation)."""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def mse_loss(pred, target):
    """Mean squared error, and its slope with respect to pred."""
    diff = pred - target
    return (diff ** 2).mean(), 2 * diff / diff.size


if __name__ == "__main__":
    print(relu(np.array([0.81, -2.0])))   # [0.81 0.  ]
