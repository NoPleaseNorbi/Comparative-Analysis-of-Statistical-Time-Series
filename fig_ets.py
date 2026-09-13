"""Section 2.7 - Exponential smoothing. Figure plus every number quoted."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ALPHAS = [0.05, 0.2, 0.6]


def ses(x, alpha):
    """Simple exponential smoothing: l_t = alpha x_t + (1-alpha) l_{t-1}."""
    l = np.empty(len(x))
    l[0] = x[0]
    for t in range(1, len(x)):
        l[t] = alpha * x[t] + (1 - alpha) * l[t - 1]
    return l


px = pd.read_csv("aapl.csv", index_col=0)["Close"].values
window = px[-250:]

fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7),
                         gridspec_kw={"width_ratios": [1, 1.3], "wspace": 0.3})

k = np.arange(0, 26)
for a in ALPHAS:
    axes[0].plot(k, a * (1 - a) ** k, "o-", ms=3, label=f"$\\alpha={a}$")
axes[0].set_title("Weight given to a day $k$ days old", fontsize=9.5)
axes[0].set_xlabel("Age $k$ of the observation")
axes[0].set_ylabel("Weight")
axes[0].legend(fontsize=8)

axes[1].plot(window, color="0.6", lw=0.9, label="AAPL close")
for a in [0.05, 0.6]:
    axes[1].plot(ses(window, a), lw=1.4, label=f"level, $\\alpha={a}$")
axes[1].set_title("The same smoothing applied to price", fontsize=9.5)
axes[1].set_xlabel("Trading day")
axes[1].set_ylabel("Price (USD)")
axes[1].legend(fontsize=8, loc="upper left")

fig.subplots_adjust(bottom=0.2)
fig.savefig("fig_ets.pdf", bbox_inches="tight")

print("alpha  weight on today  half-life (days)  weight still on day 25")
for a in ALPHAS:
    half = np.log(0.5) / np.log(1 - a)
    print(f" {a:<5} {a:>14.2f} {half:>17.1f} {a*(1-a)**25:>22.5f}")

# how well does the smoothed level predict tomorrow, versus the naive forecast?
print("\none-step-ahead RMSE on the full AAPL price series:")
print(f"  naive (yesterday's price)      {np.sqrt(np.mean(np.diff(px)**2)):.3f}")
for a in ALPHAS:
    l = ses(px, a)
    print(f"  exponential smoothing a={a:<5}  {np.sqrt(np.mean((px[1:] - l[:-1])**2)):.3f}")