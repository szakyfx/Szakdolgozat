import yfinance as yf
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

#Adatletöltés
df = yf.download("EURUSD=X", period="1y", interval="1d")[["Close"]].dropna()

# Egyszerű hozamok számítása
df["Return"] = df["Close"].pct_change()
df["Prev_Return"] = df["Return"].shift(1)
df = df.dropna()

# Szét választás
split = int(len(df) * 0.8)
train, test = df.iloc[:split], df.iloc[split:]

# Modell tanítása
model = DecisionTreeRegressor(max_depth=5, random_state=42)
model.fit(train[["Prev_Return"]], train["Return"])

# Előrejelzés
pred = model.predict(test[["Prev_Return"]])
mse = mean_squared_error(test["Return"], pred)

print(f"Modell MSE: {mse:.6f}")

