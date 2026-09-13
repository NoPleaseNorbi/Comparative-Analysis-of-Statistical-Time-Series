"""Section 2.5 - ARIMA. Generates the figure and every number quoted in the text.
No value in the section is taken from a table or from memory."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent
LAGS = 12
SEED = 0


def adf(y, lags=LAGS):
    """Augmented Dickey-Fuller statistic: beta / se(beta) from
    dX_t = alpha + beta X_{t-1} + sum gamma_i dX_{t-i} + eps_t."""
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


def critical_values(n, reps=20000, seed=SEED):
    """Dickey-Fuller critical values by simulation: the statistic's
    distribution under a true random walk of length n."""
    rng = np.random.default_rng(seed)
    s = np.array([adf(np.cumsum(rng.standard_normal(n))) for _ in range(reps)])
    return np.percentile(s, [1, 5, 10])


def ar1(x):
    """Least-squares AR(1) coefficient with its t-statistic."""
    y, lag = np.asarray(x[1:], float), np.asarray(x[:-1], float)
    X = np.column_stack([np.ones(len(lag)), lag])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ b
    se = np.sqrt(r.var(ddof=2) * np.linalg.inv(X.T @ X)[1, 1])
    return b[1], b[1] / se


def load(name):
    px = pd.read_csv(HERE / name, index_col=0)["Close"]
    px.index = pd.to_datetime(px.index, utc=True)
    return px


series = {"AAPL": load("aapl.csv"), "S&P 500": load("sp500.csv")}
logp = {k: np.log(v.values) for k, v in series.items()}
ret = {k: pd.Series(np.diff(v) * 100, index=series[k].index[1:])
       for k, v in logp.items()}

# ---- figure -------------------------------------------------------------
t_lvl, t_dif = adf(logp["AAPL"]), adf(ret["AAPL"].values)
fig, axes = plt.subplots(2, 1, figsize=(6.5, 4))
axes[0].plot(logp["AAPL"])
axes[0].set_title(f"AAPL log price $\\ln X_t$   (ADF $t = {t_lvl:+.2f}$)", fontsize=9.5)
axes[0].set_ylabel("$\\ln X_t$")
axes[1].axhline(0, color="0.6", lw=0.8)
axes[1].plot(ret["AAPL"].values)
axes[1].set_title(f"First difference $\\Delta \\ln X_t$   (ADF $t = {t_dif:+.2f}$)", fontsize=9.5)
axes[1].set_ylabel("Return (\\%)")
axes[1].set_xlabel("Trading day $t$")
fig.tight_layout()
fig.savefig(HERE / "fig_arima.pdf")

# ---- every number quoted in Section 2.5 ---------------------------------
print("--- ADF statistics ---")
for k in series:
    print(f"  {k:8s} level {adf(logp[k]):+8.3f}   differenced {adf(ret[k].values):+8.3f}")

c1, c5, c10 = critical_values(len(logp["AAPL"]))
print(f"\n--- simulated DF critical values (n={len(logp['AAPL'])}, {LAGS} lags) ---")
print(f"  1% {c1:.2f}   5% {c5:.2f}   10% {c10:.2f}")

print("\n--- AR(1) on returns ---")
for k in series:
    r = ret[k]
    ex = r[~((r.index >= "2020-02-01") & (r.index <= "2020-04-30"))]
    print(f"  {k:8s} full {ar1(r.values)[0]:+.3f} (t={ar1(r.values)[1]:+.2f})"
          f"   excl 2020 {ar1(ex.values)[0]:+.3f} (t={ar1(ex.values)[1]:+.2f})")

print("\n--- March 2020 ---")
r = ret["AAPL"]
w = r[(r.index >= "2020-03-01") & (r.index <= "2020-03-31")]
big = w.reindex(w.abs().sort_values(ascending=False).index[:3]).sort_index()
print("  three largest AAPL moves in March 2020:",
      ", ".join(f"{d.date()} {v:+.1f}%" for d, v in big.items()))
print(f"  March 2020 sd {w.std():.2f}% vs full sample {r.std():.2f}% "
      f"(ratio {w.std()/r.std():.1f})")