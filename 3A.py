import matplotlib.pyplot as plt

X = [1, 2, 3, 4, 5]
Y = [2, 4, 5, 4, 5]

n = len(X)

xy = xx = 0

for i in range(n):
    xy += X[i] * Y[i]
    xx += X[i] * X[i]

m1 = xy / xx

y1 = []

for i in range(n):
    y1.append(m1 * X[i])

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

m2 = num / den
b2 = my - m2 * mx

y2 = []

for i in range(n):
    y2.append(m2 * X[i] + b2)

plt.scatter(X, Y)

plt.plot(X, y1, label="Without Bias")
plt.plot(X, y2, label="With Bias")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression")

plt.legend()
plt.show()

print("Without Bias Slope:", m1)
print("With Bias Slope:", m2)
print("Bias Value:", b2)
