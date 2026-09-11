import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(11)

fig, axes = plt.subplots(2, 2, figsize=(6.5, 3.6), sharex=True)
for i, ax in enumerate(axes.ravel()):
    price = 100 + np.cumsum(rng.choice([-1, 1], size=500))
    ax.plot(price)
    ax.set_title(f"Path {i + 1}")
    ax.set_xlabel("Trading day $t$")
    ax.set_ylabel("Price")

fig.tight_layout()
fig.savefig("fig_rw_illusion.pdf")