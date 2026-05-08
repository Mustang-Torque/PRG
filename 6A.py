import pandas as pd
from sklearn.model_selection import train_test_split
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

X = df.iloc[:, :-1]
Y = df.iloc[:, -1]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42
)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

print("\nFirst 5 Scaled Samples\n")
print(X_train[:5])

print("\nFirst 5 Labels\n")
print(Y_train[:5].values)
