import yfinance as yf

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

#Adatletöltés
df = yf.download("EURUSD=X", period="1y", interval="1d")[["Close"]].dropna()

#Hozam
df["Return"] = df["Close"].pct_change()
df["Prev_Return"] = df["Return"].shift(1)
df = df.dropna()

#Szétválasztás
split = int(len(df) * 0.8)
X = df[["Prev_Return"]].values
y = df["Return"].values

#Skálázás
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test = X_scaled[:split], X_scaled[split:]
y_train, y_test = y[:split], y[split:]

#MLP modell
mlp = MLPRegressor(hidden_layer_sizes=(8, 4), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)

#Előrejelzés
y_pred = mlp.predict(X_test)
mse_nn = mean_squared_error(y_test, y_pred)

print(f"MLP modell MSE: {mse_nn:.6f}")

