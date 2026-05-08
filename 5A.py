import matplotlib.pyplot as plt

X = [1, 2, 3, 4, 5]
Y = [2, 4, 5, 4, 5]

lr = 0.01
epochs = 100

m1 = 0

loss1 = []

for _ in range(epochs):

    dm = 0

    for i in range(len(X)):
        y_pred = m1 * X[i]
        dm += (y_pred - Y[i]) * X[i]

    m1 = m1 - lr * (2/len(X)) * dm

    error = 0

    for i in range(len(X)):
        error += (Y[i] - m1 * X[i]) ** 2

    loss1.append(error / len(X))

m2 = 0
b2 = 0

loss2 = []

for _ in range(epochs):

    dm = 0
    db = 0

    for i in range(len(X)):
        y_pred = m2 * X[i] + b2

        dm += (y_pred - Y[i]) * X[i]
        db += (y_pred - Y[i])

    m2 = m2 - lr * (2/len(X)) * dm
    b2 = b2 - lr * (2/len(X)) * db

    error = 0

    for i in range(len(X)):
        error += (Y[i] - (m2 * X[i] + b2)) ** 2

    loss2.append(error / len(X))

plt.plot(loss1, label="Without Bias")
plt.plot(loss2, label="With Bias")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Gradient Descent Convergence")

plt.legend()
plt.show()

print("Without Bias Weight:", m1)

print("\nWith Bias Weight:", m2)
print("With Bias Bias:", b2)
