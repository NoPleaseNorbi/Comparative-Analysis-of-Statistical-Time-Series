import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess


def rets(ticker):
    px = yf.Ticker(ticker).history(start="2015-01-01", end="2024-12-31",
                                   auto_adjust=True)["Close"]
    return np.log(px).diff().dropna()


rng = np.random.default_rng(7)
ar1 = ArmaProcess(ar=[1, -0.7], ma=[1]).generate_sample(400, distrvs=rng.standard_normal)

series = [("Simulated AR(1), $\\phi_1 = 0.7$", ar1, (-0.15, 0.8)),
          ("AAPL log returns", rets("AAPL"), (-0.08, 0.08)),
          ("S&P 500 log returns", rets("^GSPC"), (-0.08, 0.08))]

fig, axes = plt.subplots(3, 2, figsize=(6.5, 6.5))
for (name, x, ylim), row in zip(series, axes):
    plot_acf(x, lags=18, zero=False, ax=row[0], title=f"{name}: ACF")
    plot_pacf(x, lags=18, zero=False, ax=row[1], title=f"{name}: PACF")
    for ax in row:
        ax.set_ylim(*ylim)
        ax.set_xlabel("Lag $k$")
fig.tight_layout()
fig.savefig("fig_ar.pdf")

for name, x, _ in series[1:]:
    x = np.asarray(x)
    print(f"{name}: n = {len(x)}, rho_1 = {np.corrcoef(x[:-1], x[1:])[0, 1]:+.4f}")