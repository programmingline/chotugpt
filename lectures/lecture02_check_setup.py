"""Lecture 2: check your setup. If this prints two version numbers and no error, you are ready."""
import numpy as np
import torch

print(np.__version__)
print(torch.__version__)

# NumPy crash course from the lecture
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a.shape)                      # (2, 3)
print(a * 2)                        # every number doubles
print(a @ np.array([1, 0, 1]))      # [ 4 10]
