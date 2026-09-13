"""Section 2.9 - ARCH / GARCH. Figure plus every number quoted.
GARCH(1,1) fitted by maximum likelihood with scipy; no external econometrics
package required."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def load(name):
    px = pd.read_csv(name, index_col=0)["Close"]
    px.index = pd.to_datetime(px.index, utc=True)
    return np.log(px).diff().dropna() * 100


def garch11_fit(r):
    """Maximise the Gaussian log-likelihood of
    sigma2_t = omega + alpha e_{t-1}^2 + beta sigma2_{t-1}."""
    e = r - r.mean()
    v0 = e.var()

    def sigma2(p):
        omega, alpha, beta = p
        s = np.empty(len(e))
        s[0] = v0
        for t in range(1, len(e)):
            s[t] = omega + alpha * e[t - 1] ** 2 + beta * s[t - 1]
        return s

    def negll(p):
        if p[0] <= 0 or p[1] < 0 or p[2] < 0 or p[1] + p[2] >= 0.999:
            return 1e10
        s = sigma2(p)
        return 0.5 * np.sum(np.log(s) + e ** 2 / s)

    best = minimize(negll, [v0 * 0.05, 0.08, 0.90], method="Nelder-Mead",
                    options={"maxiter": 4000, "xatol": 1e-8, "fatol": 1e-8})
    return best.x, np.sqrt(sigma2(best.x))


def persistence_memory(r):
    """How strongly the size of a move predicts the size of later moves."""
    a = np.abs(r - r.mean())
    a = a - a.mean()
    d = np.dot(a, a)
    return np.array([np.dot(a[:-k], a[k:]) / d for k in (1, 5, 20, 60)])


for name, f in [("AAPL", "aapl.csv"), ("S&P 500", "sp500.csv")]:
    r = load(f)
    (omega, alpha, beta), sig = garch11_fit(r.values)
    uncond = np.sqrt(omega / (1 - alpha - beta))
    mem = persistence_memory(r.values)
    print(f"\n{name}")
    print(f"  omega {omega:.4f}   alpha {alpha:.4f}   beta {beta:.4f}"
          f"   alpha+beta {alpha+beta:.4f}")
    print(f"  implied long-run daily volatility {uncond:.3f}%"
          f"   (sample {r.std():.3f}%)")
    print(f"  half-life of a volatility shock {np.log(0.5)/np.log(alpha+beta):.1f} days")
    print(f"  fitted sigma_t ranges {sig.min():.2f}% to {sig.max():.2f}%")
    print(f"  correlation of |r_t| with |r_(t-k)|, k=1,5,20,60: "
          + ", ".join(f"{v:+.3f}" for v in mem))
    if name == "AAPL":
        r_aapl, sig_aapl = r, sig

fig, ax = plt.subplots(figsize=(6.5, 2.8))
x = np.arange(len(r_aapl))
ax.plot(x, r_aapl.values, lw=0.5, color="0.55", label="AAPL daily return")
ax.plot(x, 2 * sig_aapl, lw=1.2, color="C1", label="$\\pm 2\\hat\\sigma_t$ (GARCH)")
ax.plot(x, -2 * sig_aapl, lw=1.2, color="C1")
ax.axhline(0, color="0.3", lw=0.7)
ax.set_xlabel("Trading day $t$")
ax.set_ylabel("Return (%)")
ax.set_title("Returns and the fitted conditional volatility", fontsize=9.5)
ax.legend(fontsize=8, loc="lower left")
fig.subplots_adjust(bottom=0.2)
fig.savefig("fig_garch.pdf", bbox_inches="tight")