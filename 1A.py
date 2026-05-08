X = [1, 2, 3, 4, 5]
Y = [2, 4, 5, 4, 5]

n = len(X)

sum_x = 0
sum_y = 0

for i in range(n):
    sum_x += X[i]
    sum_y += Y[i]

mean_x = sum_x / n
mean_y = sum_y / n

numerator = 0
denominator = 0

for i in range(n):
    numerator += (X[i] - mean_x) * (Y[i] - mean_y)
    denominator += (X[i] - mean_x) ** 2

m = numerator / denominator
b = mean_y - (m * mean_x)

y_pred = []

for i in range(n):
    prediction = m * X[i] + b
    y_pred.append(prediction)

mse_sum = 0

for i in range(n):
    mse_sum += (Y[i] - y_pred[i]) ** 2

mse = mse_sum / n

print("Slope (m):", m)
print("Bias (b):", b)
print("Predicted Values:", y_pred)
print("Mean Squared Error:", mse)
