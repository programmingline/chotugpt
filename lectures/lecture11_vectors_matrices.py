"""Lecture 11: vectors, matrices and dot products in NumPy (toy word vectors from lecture 5, for teaching)."""
import numpy as np

chai = np.array([0.9, 0.8, 0.1])
tea = np.array([0.8, 0.9, 0.3])
car = np.array([0.1, 0.2, 0.95])

print(round(chai @ tea, 3), round(chai @ car, 3))   # dot products

W = np.array([[1, 1, 0],     # row 1: "hot drink?"
              [0, 0, 1]])    # row 2: "vehicle?"
print(W @ chai, W @ car)     # each output = one dot product

words = np.stack([chai, tea, car])   # a 3 x 3 matrix
print(words.shape)
print((words @ words.T).round(3))    # every pair at once
