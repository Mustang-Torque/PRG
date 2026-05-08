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

weights = [0] * len(X_train[0])
bias = 0

lr = 0.1
epochs = 10

for _ in range(epochs):

    for i in range(len(X_train)):

        s = bias

        for j in range(len(weights)):
            s += weights[j] * X_train[i][j]

        pred = 1 if s >= 0 else 0

        error = Y_train[i] - pred

        for j in range(len(weights)):
            weights[j] += lr * error * X_train[i][j]

        bias += lr * error

predictions = []

for i in range(len(X_test)):

    s = bias

    for j in range(len(weights)):
        s += weights[j] * X_test[i][j]

    pred = 1 if s >= 0 else 0

    predictions.append(pred)

acc = accuracy_score(Y_test, predictions)

print("Weights:", weights)
print("Bias:", bias)
print("Accuracy:", acc)
