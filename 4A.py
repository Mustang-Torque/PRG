import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

data = {
    "X1": [10, 20, 30, 40, 50],
    "X2": [15, 25, 35, 45, 55],
    "Y":  [20, 35, 45, 55, 65]
}

df = pd.DataFrame(data)

print("Original Data\n")
print(df)

corr = df.corr()

print("\nCorrelation Matrix\n")
print(corr)

X = df[["X1", "X2"]]
Y = df["Y"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nScaled Features\n")
print(X_scaled)

model = LinearRegression()
model.fit(X_scaled, Y)

pred = model.predict(X_scaled)

print("\nPredicted Values\n")
print(pred)

print("\nCoefficients:", model.coef_)
print("Intercept:", model.intercept_)
