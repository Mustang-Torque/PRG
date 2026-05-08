import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

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

X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42
)

lr = 0.1
epochs = 10

w1 = [0] * len(X_train[0])

for _ in range(epochs):

    for i in range(len(X_train)):

        s = 0

        for j in range(len(w1)):
            s += w1[j] * X_train[i][j]

        pred = 1 if s >= 0 else 0

        error = Y_train[i] - pred

        for j in range(len(w1)):
            w1[j] += lr * error * X_train[i][j]

pred1 = []

for i in range(len(X_test)):

    s = 0

    for j in range(len(w1)):
        s += w1[j] * X_test[i][j]

    pred1.append(1 if s >= 0 else 0)

acc1 = accuracy_score(Y_test, pred1)

w2 = [0] * len(X_train[0])
b2 = 0

for _ in range(epochs):

    for i in range(len(X_train)):

        s = b2

        for j in range(len(w2)):
            s += w2[j] * X_train[i][j]

        pred = 1 if s >= 0 else 0

        error = Y_train[i] - pred

        for j in range(len(w2)):
            w2[j] += lr * error * X_train[i][j]

        b2 += lr * error

pred2 = []

for i in range(len(X_test)):

    s = b2

    for j in range(len(w2)):
        s += w2[j] * X_test[i][j]

    pred2.append(1 if s >= 0 else 0)

acc2 = accuracy_score(Y_test, pred2)

print("Without Bias Accuracy:", acc1)

print("\nWith Bias Accuracy:", acc2)
print("Bias Value:", b2)
