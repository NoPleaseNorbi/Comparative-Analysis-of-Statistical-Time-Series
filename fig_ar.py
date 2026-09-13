import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

rng = np.random.default_rng(3)
e = rng.standard_normal(400)                 # same shocks for every panel

fig, axes = plt.subplots(2, 2, figsize=(6.5, 3.8), sharex=True)
for phi, ax in zip([0.3, 0.7, 0.95, 1.0], axes.ravel()):
    x = np.zeros(400)
    for t in range(1, 400):
        x[t] = phi * x[t - 1] + e[t]
    ax.axhline(0, color="0.6", lw=0.8)
    ax.plot(x)
    ax.set_title(f"$\\phi_1 = {phi}$" + (" (random walk)" if phi == 1.0 else ""),
                 fontsize=9.5)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$X_t$")

fig.tight_layout()
fig.savefig("fig_ar.pdf")