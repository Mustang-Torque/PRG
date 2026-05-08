X = [1, 2, 3, 4, 5]
Y = [2, 4, 5, 4, 5]

n = len(X)

xy = xx = 0

for i in range(n):
    xy += X[i] * Y[i]
    xx += X[i] * X[i]

m1 = xy / xx

y1 = []
mse1 = 0

for i in range(n):
    pred = m1 * X[i]
    y1.append(pred)
    mse1 += (Y[i] - pred) ** 2

mse1 /= n

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
mse2 = 0

for i in range(n):
    pred = m2 * X[i] + b2
    y2.append(pred)
    mse2 += (Y[i] - pred) ** 2

mse2 /= n

print("Without Bias")
print("Slope:", m1)
print("MSE:", mse1)

print("\nWith Bias")
print("Slope:", m2)
print("Bias:", b2)
print("MSE:", mse2)
