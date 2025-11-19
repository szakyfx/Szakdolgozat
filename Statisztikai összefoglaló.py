#EVT
evt_threshold = to_scalar(returns.quantile(EVT_QUANTILE))
evt_excess = (returns[returns > evt_threshold] - evt_threshold).dropna()
evt_value = np.nan
evt_std = to_scalar(evt_excess.std(ddof=1)) if len(evt_excess) else 0.0

if len(evt_excess) >= 3 and evt_std > 0.0:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            # Fittelés (helyparaméter fixálva 0-ra)
            _ = genpareto.fit(evt_excess.to_numpy(), floc=0)
            # EVT-küszöb árbázison (aktuális záró * (1 + küszöb-hozam))
            last_price = to_scalar(data["Close"].iloc[-1])
            evt_value = last_price * (1.0 + evt_threshold)
        except Exception:
            evt_value = np.nan

#MARKOV-SWITCHING
markov_msg = None
markov_last_prob = None
markov_llf = None
try:
    r = returns.dropna()
    if len(r) >= 200:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ms_model = MarkovRegression(r, k_regimes=2, trend='c', switching_variance=True)
            ms_res = ms_model.fit(disp=False)
        # magasabb varianciájú állapot simított valószínűsége
        markov_last_prob = float(ms_res.smoothed_marginal_probabilities[1].iloc[-1])
        markov_llf = float(ms_res.llf)
    else:
        markov_msg = f"Kevés adat – Markov kihagyva (n={len(r)} < 200)"
except Exception as e:
    markov_msg = f"Markov hiba: {e}"

#EREDMÉNYEK
print("\n📊 10 éves elemzési összefoglaló")
print(f"Vizsgált eszköz: {TICKER}")
print("────────────────────────────────────────────")
print(f"Felső tartomány valószínűsége: {p_upper:.3f} %")
print(f"Alsó tartomány valószínűsége: {p_lower:.3f} %")
print(f"Felső átlagos eltérés: {mean_upper_excess:.4f}")
print(f"Felső maximális túllépés: {max_upper_excess:.4f}")
print(f"Alsó átlagos eltérés: {mean_lower_excess:.4f}")
print(f"Alsó maximális túllépés: {max_lower_excess:.4f}")
print(f"Szélső érték (EVT-küszöb): {evt_value:.3f}")
print(f"Aszimmetria (ferdeség): {asymmetry:.3f}")
print(f"Value at Risk (VaR 5%): {var_value:.4f}")
# ---- Markov eredmények kiírása (csak ez az új sorozat) ----
if markov_last_prob is not None:
    print(f"Markov-switching (k=2): magas vol. rezsim valószínűsége (utolsó nap) = {markov_last_prob:.2%} | LLF = {markov_llf:.2f}")
elif markov_msg:
    print(markov_msg)
print("────────────────────────────────────────────")
print("ARIMA modell (1,1,1) – időbeli dinamika előrejelzés")
print(f"AIC érték: {aic_val:.2f}")
print(f"BIC érték: {bic_val:.2f}")
print(f"Reziduum szórása: {resid_std:.5f}")
print(f"Paraméterek (φ, θ, μ): {[round(x, 4) for x in params_list]}")
print(f"30 napos előrejelzett ár: {pred_last:.3f}")
print(f"Várható elmozdulás 30 nap alatt: {forecast_change:.2f} %")
print(f"95%-os konfidencia intervallum: [{ci_low:.3f}, {ci_high:.3f}]")
print("────────────────────────────────────────────")