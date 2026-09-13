"""Section 2.8 - ARFIMA / fractional differencing.
Figure plus every number quoted in the text."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LAGS = 12
TAU = 1e-4                      # weight cutoff for the fixed-width window


def weights(d, tau=TAU, max_k=5000):
    """Coefficients of (1-B)^d: w_0 = 1, w_k = w_{k-1} (k-1-d)/k.
    Truncated once |w_k| falls below tau."""
    w = [1.0]
    for k in range(1, max_k):
        nxt = w[-1] * (k - 1 - d) / k
        if abs(nxt) < tau:
            break
        w.append(nxt)
    return np.array(w)


def fracdiff(x, d, tau=TAU):
    """Fractionally difference x by order d, using only past values."""
    w = weights(d, tau)
    L = len(w)
    out = np.full(len(x), np.nan)
    for t in range(L - 1, len(x)):
        out[t] = np.dot(w, x[t::-1][:L])
    return out[~np.isnan(out)]


def adf(y, lags=LAGS):
    y = np.asarray(y, float)
    dy = np.diff(y)
    n = len(dy) - lags
    Y = dy[lags:]
    X = np.column_stack([np.ones(n), y[lags:-1]]
                        + [dy[lags - i:-i] for i in range(1, lags + 1)])
    b = np.linalg.lstsq(X, Y, rcond=None)[0]
    r = Y - X @ b
    s2 = r @ r / (n - X.shape[1])
    return b[1] / np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])


def rho1(x):
    x = np.asarray(x, float) - np.mean(x)
    return np.dot(x[:-1], x[1:]) / np.dot(x, x)


THRESH = -2.83                  # simulated in Section 2.5
grid = np.round(np.arange(0.0, 1.01, 0.05), 2)
results = {}

for name, f in [("AAPL", "aapl.csv"), ("S&P 500", "sp500.csv")]:
    logp = np.log(pd.read_csv(f, index_col=0)["Close"].values)
    stats = np.array([adf(fracdiff(logp, d)) for d in grid])
    dmin = grid[np.argmax(stats < THRESH)]
    series = fracdiff(logp, dmin)
    results[name] = (stats, dmin, series)
    print(f"\n{name}: minimum d passing ADF = {dmin:.2f}"
          f"  (statistic {adf(series):+.2f})")
    print(f"  weights kept at d={dmin:.2f}: {len(weights(dmin))}"
          f"   at d=1.00: {len(weights(1.0))}")
    print(f"  lag-1 autocorrelation   d={dmin:.2f}: {rho1(series):+.4f}"
          f"    d=1.00: {rho1(fracdiff(logp, 1.0)):+.4f}")
    print(f"  correlation with the log price   d={dmin:.2f}: "
          f"{np.corrcoef(series, logp[-len(series):])[0,1]:+.3f}"
          f"    d=1.00: {np.corrcoef(fracdiff(logp,1.0), logp[-len(fracdiff(logp,1.0)):])[0,1]:+.3f}")

fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8),
                         gridspec_kw={"width_ratios": [1, 1.2], "wspace": 0.3})

for d in [0.2, 0.45, 1.0]:
    w = np.abs(weights(d))
    axes[0].semilogy(np.arange(len(w)), w, "o-", ms=2.5, lw=1,
                     label=f"$d={d}$ ({len(w)} terms)")
axes[0].set_xlim(-2, 260)
axes[0].set_title("Size of the weight on a day $k$ back", fontsize=9.5)
axes[0].set_xlabel("Lag $k$")
axes[0].set_ylabel("$|\\pi_k|$ (log scale)")
axes[0].legend(fontsize=7.5)

for name in results:
    axes[1].plot(grid, results[name][0], "o-", ms=3, label=name)
axes[1].axhline(THRESH, color="0.4", ls="--", lw=1)
axes[1].text(0.02, THRESH - 1.2, "5% threshold", fontsize=7.5, color="0.3")
axes[1].set_title("Stationarity against differencing order", fontsize=9.5)
axes[1].set_xlabel("Differencing order $d$")
axes[1].set_ylabel("ADF statistic")
axes[1].legend(fontsize=8, loc="lower left")

fig.subplots_adjust(bottom=0.2)
fig.savefig("fig_arfima.pdf", bbox_inches="tight")