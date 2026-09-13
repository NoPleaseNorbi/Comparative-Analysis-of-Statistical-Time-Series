"""Section 2.6 - SARIMA. Figure plus every number quoted in the text."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]


def load(name):
    px = pd.read_csv(name, index_col=0)["Close"]
    px.index = pd.to_datetime(px.index, utc=True)
    return np.log(px).diff().dropna() * 100


def weekday_stats(r):
    g = r.groupby(r.index.dayofweek)
    m = np.array([g.get_group(d).mean() for d in range(5)])
    se = np.array([g.get_group(d).std() / np.sqrt(len(g.get_group(d))) for d in range(5)])
    n = len(r)
    ssb = sum(len(g.get_group(d)) * (g.get_group(d).mean() - r.mean()) ** 2 for d in range(5))
    ssw = sum(((g.get_group(d) - g.get_group(d).mean()) ** 2).sum() for d in range(5))
    F = (ssb / 4) / (ssw / (n - 5))
    return m, se, F, n


series = {"AAPL": load("aapl.csv"), "S&P 500": load("sp500.csv")}

# simulated series with a genuine weekly cycle, for contrast
rng = np.random.default_rng(12)
t = np.arange(260)
seasonal = 1.5 * np.sin(2 * np.pi * t / 5) + 0.4 * rng.standard_normal(260)

fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.9),
                         gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.32})

axes[0].plot(t[:60], seasonal[:60])
axes[0].axhline(0, color="0.6", lw=0.8)
axes[0].set_title("Simulated series with a five-day cycle", fontsize=9.5)
axes[0].set_xlabel("Trading day $t$")
axes[0].set_ylabel("$X_t$")

off = {"AAPL": -0.12, "S&P 500": 0.12}
for name, r in series.items():
    m, se, F, n = weekday_stats(r)
    axes[1].errorbar(np.arange(5) + off[name], m, yerr=2 * se, fmt="o",
                     ms=4, capsize=3, label=name)
    print(f"{name}: F(4,{n-5}) = {F:.3f}")
    for d, lbl in enumerate(DAYS):
        print(f"   {lbl}: {m[d]:+.4f}% +/- {2*se[d]:.4f}  (t = {m[d]/se[d]:+.2f})")
axes[1].axhline(0, color="0.6", lw=0.8)
axes[1].set_xticks(range(5))
axes[1].set_xticklabels(DAYS)
axes[1].set_title("Mean return by weekday ($\\pm 2$ s.e.)", fontsize=9.5)
axes[1].set_ylabel("Mean return (%)")
axes[1].legend(loc="lower left", fontsize=8)

fig.subplots_adjust(bottom=0.2, top=0.88)
fig.savefig("fig_sarima.pdf", bbox_inches="tight")
print("\n5% critical value for F(4, ~2500) is approximately 2.37")