import numpy as np

X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 5, 4, 5])

m1 = np.sum(X * Y) / np.sum(X * X)
y1 = m1 * X
mse1 = np.mean((Y - y1) ** 2)

m2 = np.sum((X - X.mean()) * (Y - Y.mean())) / np.sum((X - X.mean()) ** 2)
b2 = Y.mean() - m2 * X.mean()
y2 = m2 * X + b2
mse2 = np.mean((Y - y2) ** 2)

print("Without Bias")
print("Slope:", m1)
print("MSE:", mse1)

print("\nWith Bias")
print("Slope:", m2)
print("Bias:", b2)
print("MSE:", mse2)
