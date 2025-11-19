
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



REFERENCE_DATE = "2025-07-01"   # vagy None

# három panel hossza
PANELS = [
    ("10 éves", dict(years=10)),
    ("1 éves",  dict(years=1)),
    ("Féléves", dict(months=6)),
]


#adat & oszlop-normalizálás

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in df.columns]
    if "adj close" in df.columns and "close" not in df.columns:
        df["close"] = df["adj close"]
    return df


def fetch_prices_range(ticker: str, start: str, end: str) -> pd.DataFrame:

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError(f"Nincs adat: {ticker} ({start} → {end})")
    df = _normalize_columns(df)
    df = df.dropna(subset=["close"]).copy()
    return df


def fetch_prices_relative(ticker: str, until: str | None, years: int = 0, months: int = 0, days: int = 0) -> tuple[pd.DataFrame, str, str]:

    if until is None:
        end_dt = pd.Timestamp.today(tz="UTC").normalize()
    else:
        end_dt = pd.to_datetime(until).tz_localize("UTC") if pd.to_datetime(until).tzinfo is None else pd.to_datetime(until).tz_convert("UTC")
        end_dt = end_dt.normalize()

    start_dt = end_dt - pd.DateOffset(years=years, months=months, days=days)
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str   = end_dt.strftime("%Y-%m-%d")

    df = fetch_prices_range(ticker, start=start_str, end=end_str)
    return df, start_str, end_str


#trend, sávok, reziduum

def compute_trend_and_bands(df: pd.DataFrame):

    y = df["close"].values.reshape(-1, 1)
    x = np.arange(len(df)).reshape(-1, 1)
    model = LinearRegression().fit(x, y)
    trend = model.predict(x).ravel()
    resid = y.ravel() - trend
    std = float(np.std(resid, ddof=0))

    out = df.copy()
    out["Trend"] = trend
    # ±1σ sávok
    out["+1σ"] = trend + 1 * std
    out["-1σ"] = trend - 1 * std
    # ±2σ sávok
    out["+2σ"] = trend + 2 * std
    out["-2σ"] = trend - 2 * std
    # extrém detektáláshoz
    out["_resid"] = resid
    out["_sigma"] = std
    return out


#extrém csúcs/mélypont detektálás

def detect_extrema(df: pd.DataFrame, k_sigma: float = 1.5, neighborhood: int = 2):

    close = df["close"].values
    resid = df["_resid"].values
    sigma = float(df["_sigma"].iloc[0])  # konstans a sorozatra

    n = len(df)
    peaks = np.zeros(n, dtype=bool)
    troughs = np.zeros(n, dtype=bool)

    for i in range(n):
        lo = max(0, i - neighborhood)
        hi = min(n, i + neighborhood + 1)
        neighborhood_vals = close[lo:hi]

        is_peak_local   = close[i] == neighborhood_vals.max() and np.sum(neighborhood_vals == close[i]) == 1
        is_trough_local = close[i] == neighborhood_vals.min() and np.sum(neighborhood_vals == close[i]) == 1

        if is_peak_local and (resid[i] > +k_sigma * sigma):
            peaks[i] = True
        if is_trough_local and (resid[i] < -k_sigma * sigma):
            troughs[i] = True

    return pd.Series(peaks, index=df.index, name="is_peak"), pd.Series(troughs, index=df.index, name="is_trough")


#volatilitás-rezsimek

def compute_volatility_regimes(df: pd.DataFrame, vol_window: int = 30):
    vola = df["close"].rolling(vol_window).std(ddof=0)
    thr_low, thr_high = vola.quantile(0.4), vola.quantile(0.6)
    regime = []
    state = 0  # 0 = calm, 1 = volatile
    for v in vola:
        if state == 0 and v > thr_high:
            state = 1
        elif state == 1 and v < thr_low:
            state = 0
        regime.append(state)
    out = df.copy()
    out["Regime"] = np.where(np.array(regime) == 1, "volatile", "calm")
    return out


# rajz

def plot_single_panel(df: pd.DataFrame, title: str, ticker: str, k_sigma: float, start_str: str, end_str: str):
    plt.figure(figsize=(16, 8))

    # rezsim háttér
    for regime, color in [("volatile", "#fabcbc"), ("calm", "#b2fab4")]:
        mask = df["Regime"] == regime
        plt.fill_between(df.index, df["close"].min(), df["close"].max(),
                         where=mask, color=color, alpha=0.35, label=f"{regime} rezsim")

    # ár, trend, sávok
    plt.plot(df.index, df["close"], label="Ár", color="black", linewidth=1.3)
    plt.plot(df.index, df["Trend"], label="Trend", color="green", linewidth=1.6)

    # ±1σ: halványabb szaggatott
    plt.plot(df.index, df["+1σ"], "--", color="gray",  linewidth=1.2, label="+1σ")
    plt.plot(df.index, df["-1σ"], "--", color="gray",  linewidth=1.2, label="-1σ")

    # ±2σ: erősebb szaggatott
    plt.plot(df.index, df["+2σ"], "--", color="red",   linewidth=1.2, label="+2σ")
    plt.plot(df.index, df["-2σ"], "--", color="orange", linewidth=1.2, label="-2σ")

    # csúcspontok/mélypontok
    peaks_mask, troughs_mask = detect_extrema(df, k_sigma=k_sigma, neighborhood=2)
    if peaks_mask.any():
        plt.scatter(df.index[peaks_mask], df["close"][peaks_mask],
                    s=55, color="#d90000", edgecolor="black", zorder=5, label=f"Csúcsok (>{k_sigma}σ)")
    if troughs_mask.any():
        plt.scatter(df.index[troughs_mask], df["close"][troughs_mask],
                    s=55, color="#6a0dad", edgecolor="black", zorder=5, label=f"Mélypontok (<-{k_sigma}σ)")

    plt.title(f"{ticker} – {title} | {start_str} → {end_str}  (±1σ, ±2σ; k={k_sigma}σ)", fontsize=14)
    plt.xlabel("Dátum")
    plt.ylabel("Árfolyam")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()


#fő futtatás

if __name__ == "__main__":
    TICKER = "EURUSD=X"
    K_SIGMA = 1.5

    for label, span in PANELS:
        df, start_str, end_str = fetch_prices_relative(
            TICKER,
            until=REFERENCE_DATE,
            years=span.get("years", 0),
            months=span.get("months", 0),
            days=span.get("days", 0)
        )
        df = compute_trend_and_bands(df)
        df = compute_volatility_regimes(df)
        plot_single_panel(df, label, TICKER, k_sigma=K_SIGMA, start_str=start_str, end_str=end_str)
