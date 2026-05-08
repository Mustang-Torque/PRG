import matplotlib.pyplot as plt

X = [1, 2, 3, 4, 5]
Y = [2, 4, 5, 4, 5]

n = len(X)

sum_x = sum_y = 0

for i in range(n):
    sum_x += X[i]
    sum_y += Y[i]

mx = sum_x / n
my = sum_y / n

num = den = 0

for i in range(n):
    num += (X[i] - mx) * (Y[i] - my)
    den += (X[i] - mx) ** 2

m = num / den
b = my - m * mx

pred = []

for i in range(n):
    pred.append(m * X[i] + b)

mse = mae = 0

for i in range(n):
    mse += (Y[i] - pred[i]) ** 2
    mae += abs(Y[i] - pred[i])

mse /= n
mae /= n

print("Slope:", m)
print("Bias:", b)

print("\nPredicted Values:", pred)

print("\nMean Squared Error:", mse)
print("Mean Absolute Error:", mae)

plt.scatter(X, Y)

plt.plot(X, pred, label="Regression Line")

plt.xlabel("X")
plt.ylabel("Y")

plt.title("Linear Regression with Bias")

plt.legend()
plt.show()
