import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

n = 25
shock = np.zeros(n)
shock[5] = 1.0                                  # one unit shock on day 5

ar = np.zeros(n)                                # AR(1), phi = 0.7
for t in range(1, n):
    ar[t] = 0.7 * ar[t - 1] + shock[t]

ma = shock + 0.8 * np.r_[0, shock[:-1]]         # MA(1), theta = 0.8

fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.4), sharey=True)
for ax, y, title in ((axes[0], ar, "AR(1), $\\phi_1 = 0.7$"),
                     (axes[1], ma, "MA(1), $\\theta_1 = 0.8$")):
    ax.axhline(0, color="0.6", lw=0.8)
    lag = np.arange(n) - 5
    ax.vlines(lag, 0, y, lw=1.6)
    ax.plot(lag, y, "o", ms=3.5)
    ax.set_xlim(-2, 16)
    ax.set_title(f"{title}: effect of one shock", fontsize=9.5)
    ax.set_xlabel("Days since the shock arrived")
axes[0].set_ylabel("Effect on $X_t$")

fig.tight_layout()
fig.savefig("fig_ma.pdf")