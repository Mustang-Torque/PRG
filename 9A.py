import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

cols = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "class"
]

df = pd.read_csv(url, names=cols)

df = df[df["class"] != "Iris-virginica"]

df["class"] = df["class"].map({
    "Iris-setosa": 0,
    "Iris-versicolor": 1
})

X = df.iloc[:, [0, 1]].values
Y = df.iloc[:, -1].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

lr = 0.1
epochs = 10

w1 = [0, 0]

for _ in range(epochs):

    for i in range(len(X)):

        s = w1[0] * X[i][0] + w1[1] * X[i][1]

        pred = 1 if s >= 0 else 0

        error = Y[i] - pred

        w1[0] += lr * error * X[i][0]
        w1[1] += lr * error * X[i][1]

w2 = [0, 0]
b2 = 0

for _ in range(epochs):

    for i in range(len(X)):

        s = w2[0] * X[i][0] + w2[1] * X[i][1] + b2

        pred = 1 if s >= 0 else 0

        error = Y[i] - pred

        w2[0] += lr * error * X[i][0]
        w2[1] += lr * error * X[i][1]

        b2 += lr * error

x1 = [min(X[:,0]), max(X[:,0])]

y1 = [
    -(w1[0] * x + 0) / w1[1]
    for x in x1
]

y2 = [
    -(w2[0] * x + b2) / w2[1]
    for x in x1
]

for i in range(len(X)):

    if Y[i] == 0:
        plt.scatter(X[i][0], X[i][1], marker='o')

    else:
        plt.scatter(X[i][0], X[i][1], marker='x')

plt.plot(x1, y1, label="Without Bias")
plt.plot(x1, y2, label="With Bias")

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")

plt.title("Decision Boundary Comparison")

plt.legend()
plt.show()
