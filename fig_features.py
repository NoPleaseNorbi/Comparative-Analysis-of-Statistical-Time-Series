"""Section 3 - what the models receive as input.
The same price series at three differencing orders."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TAU = 1e-4


def weights(d, tau=TAU, max_k=5000):
    w = [1.0]
    for k in range(1, max_k):
        nxt = w[-1] * (k - 1 - d) / k
        if abs(nxt) < tau:
            break
        w.append(nxt)
    return np.array(w)


def fracdiff(x, d, tau=TAU):
    w = weights(d, tau)
    L = len(w)
    out = np.full(len(x), np.nan)
    for t in range(L - 1, len(x)):
        out[t] = np.dot(w, x[t::-1][:L])
    return out


def adf(y, lags=12):
    y = np.asarray(y, float)
    y = y[~np.isnan(y)]
    dy = np.diff(y)
    n = len(dy) - lags
    Y = dy[lags:]
    X = np.column_stack([np.ones(n), y[lags:-1]]
                        + [dy[lags - i:-i] for i in range(1, lags + 1)])
    b = np.linalg.lstsq(X, Y, rcond=None)[0]
    r = Y - X @ b
    s2 = r @ r / (n - X.shape[1])
    return b[1] / np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])


logp = np.log(pd.read_csv("aapl.csv", index_col=0)["Close"].values)
d_star = 0.45

panels = [("$d = 0$: the log price itself", logp, "0.35"),
          (f"$d = {d_star}$: fractionally differenced", fracdiff(logp, d_star), "C1"),
          ("$d = 1$: log returns", np.r_[np.nan, np.diff(logp)], "C0")]

fig, axes = plt.subplots(3, 1, figsize=(6.5, 4.6), sharex=True)
for (title, y, c), ax in zip(panels, axes):
    ax.plot(y, lw=0.6, color=c)
    ax.set_title(f"{title}   (ADF ${adf(y):+.2f}$)", fontsize=9.5)
    ax.set_ylabel("value")
axes[-1].set_xlabel("Trading day $t$")
fig.subplots_adjust(hspace=0.55, bottom=0.12)
fig.savefig("fig_features.pdf", bbox_inches="tight")

print("series fed to the models (AAPL):")
for title, y, _ in panels:
    v = y[~np.isnan(y)]
    lag1 = np.corrcoef(v[:-1], v[1:])[0, 1]
    print(f"  {title:45s} ADF {adf(y):+7.2f}   lag-1 corr {lag1:+.3f}"
          f"   usable obs {len(v)}")