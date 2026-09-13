import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

px = yf.Ticker("AAPL").history(start="2015-01-01", end="2024-12-31",
                               auto_adjust=True)["Close"]
px.to_csv("aapl.csv")          # keep the file so results stay reproducible
py = yf.Ticker("^GSPC").history(start="2015-01-01", end="2024-12-31",
                               auto_adjust=True)["Close"]
py.to_csv("sp500.csv")  # keep the file so results stay reproducible


r = np.log(px).diff().dropna()
mu, sd = r.mean(), r.std()

h = np.arange(1, 61)
last = np.log(px.iloc[-1])
mid = np.exp(last + mu * h)
lo = np.exp(last + mu * h - 1.96 * sd * np.sqrt(h))
hi = np.exp(last + mu * h + 1.96 * sd * np.sqrt(h))

hist = px.iloc[-250:].values
x = len(hist) - 1 + h

fig, ax = plt.subplots(figsize=(6.5, 3))
ax.plot(hist, label="AAPL close")
ax.fill_between(x, lo, hi, alpha=0.2, label="95% interval")
ax.plot(x, mid, label="Random-walk forecast")
ax.set_xlabel("Trading day $t$")
ax.set_ylabel("Price (USD)")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig("fig_rw_forecast.pdf")

print(f"drift mu = {100*mu:.4f} %/day, sd = {100*sd:.3f} %/day, n = {len(r)}")