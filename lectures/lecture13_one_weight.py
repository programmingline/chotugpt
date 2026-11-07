"""Lecture 13: one weight, one gradient-descent step, by hand (toy numbers, for teaching)."""
w = 0.2
pred = w * 0.9                      # input: chai = 0.9
loss = (pred - 1) ** 2              # target: 1 (fresh); squared error
slope = 2 * (pred - 1) * 0.9        # derivative of the loss with respect to w
print("pred", round(pred, 2), "loss", round(loss, 2), "slope", round(slope, 2))
for step in range(4):
    w = w - 0.5 * slope             # gradient descent, learning rate 0.5
    pred = w * 0.9
    slope = 2 * (pred - 1) * 0.9
    print("w", round(w, 2), "loss", round((pred - 1) ** 2, 4))
