"""Lecture 5: word vectors — similar words get similar numbers (toy values, for teaching)."""
import numpy as np


def cosine(a, b):
    return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))


chai = np.array([0.9, 0.8, 0.1])
tea = np.array([0.8, 0.9, 0.3])
coffee = np.array([0.6, 0.9, 0.5])
car = np.array([0.1, 0.2, 0.95])

print("chai · tea    ", round(cosine(chai, tea), 2))     # 0.98
print("chai · coffee ", round(cosine(chai, coffee), 2))  # 0.91
print("chai · car    ", round(cosine(chai, car), 2))     # 0.29

# king - man + woman ≈ queen  (2D toy: [royalty, maleness])
king, man, woman, queen = (np.array(v) for v in ([0.95, 0.9], [0.1, 0.9], [0.1, 0.1], [0.95, 0.1]))
print(king - man + woman, "≈ queen", queen)            # [0.95 0.1 ]
