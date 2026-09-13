"""Unified walk-forward comparison: statistical models and machine learning
models, same protocol, same test days, one-step-ahead next-day log return."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

L, REFIT, TRAIN0, TAU, D = 10, 250, 1200, 1e-4, 0.5


def weights(d, tau=TAU):
    w = [1.0]
    for k in range(1, 5000):
        nxt = w[-1] * (k - 1 - d) / k
        if abs(nxt) < tau:
            break
        w.append(nxt)
    return np.array(w)


def fracdiff(x, d):
    w = weights(d)
    out = np.full(len(x), np.nan)
    for t in range(len(w) - 1, len(x)):
        out[t] = np.dot(w, x[t::-1][:len(w)])
    return out


def ols(X, y):
    A = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(A, y, rcond=None)[0]


def arma11_fit(r):
    """Conditional least squares for r_t = c + phi r_{t-1} + theta e_{t-1} + e_t."""
    def sse(p):
        c, phi, th = p
        e = np.zeros(len(r))
        for t in range(1, len(r)):
            e[t] = r[t] - c - phi * r[t - 1] - th * e[t - 1]
        return np.sum(e[1:] ** 2)
    best = minimize(sse, [r.mean(), 0.0, 0.0], method="Nelder-Mead",
                    options={"maxiter": 2000, "fatol": 1e-8})
    c, phi, th = best.x
    e = np.zeros(len(r))
    for t in range(1, len(r)):
        e[t] = r[t] - c - phi * r[t - 1] - th * e[t - 1]
    return c, phi, th, e[-1]


px = pd.read_csv("aapl.csv", index_col=0)["Close"].values
logp = np.log(px)
ret = np.r_[np.nan, np.diff(logp)] * 100
frac = fracdiff(logp, D)

rows, y_all, idx_all = [], [], []
for t in range(L, len(ret) - 1):
    if np.isnan(ret[t - L + 1:t + 1]).any() or np.isnan(frac[t - L + 1:t + 1]).any():
        continue
    rows.append((ret[t - L + 1:t + 1], frac[t - L + 1:t + 1]))
    y_all.append(ret[t + 1]); idx_all.append(t + 1)
Xr = np.array([a for a, _ in rows]); Xf = np.array([b for _, b in rows])
y = np.array(y_all); idx = np.array(idx_all)
print(f"aligned sample {len(y)} rows, test days {len(y) - TRAIN0}")

ML = {"Linear regression": LinearRegression(), "Ridge regression": Ridge(alpha=10.0),
      "Lasso regression": Lasso(alpha=0.01, max_iter=5000),
      "SVR (RBF)": SVR(kernel="rbf", C=1.0, epsilon=0.05),
      "Random forest": RandomForestRegressor(n_estimators=200, max_depth=6,
                                             random_state=0, n_jobs=-1)}
preds = {k: [] for k in ["Random walk", "AR(1)", "AR(5)", "ARMA(1,1)",
                         "ARFIMA-type AR(5)", "Markov switching"] + list(ML)}
truth = []

for start in range(TRAIN0, len(y), REFIT):
    stop = min(start + REFIT, len(y))
    truth.append(y[start:stop])
    tr_r, tr_f, tr_y = Xr[:start], Xf[:start], y[:start]

    preds["Random walk"].append(np.full(stop - start, tr_y.mean()))
    for name, k, src in [("AR(1)", 1, Xr), ("AR(5)", 5, Xr),
                         ("ARFIMA-type AR(5)", 5, Xf)]:
        b = ols(src[:start, -k:], tr_y)
        preds[name].append(b[0] + src[start:stop, -k:] @ b[1:])

    c, phi, th, _ = arma11_fit(tr_y)
    e = 0.0
    out = []
    for t in range(start, stop):
        out.append(c + phi * y[t - 1] + th * e)
        e = y[t] - out[-1]
    preds["ARMA(1,1)"].append(np.array(out))

    # two-regime mean, weighted by how turbulent the recent window looks
    lo, hi = tr_y[np.abs(tr_y) < np.percentile(np.abs(tr_y), 75)], tr_y
    mu_lo, mu_hi = lo.mean(), hi[np.abs(hi) >= np.percentile(np.abs(hi), 75)].mean()
    vol = pd.Series(np.abs(y)).rolling(20).mean().values
    thr = np.nanpercentile(vol[:start], 75)
    w = (vol[start:stop] > thr).astype(float)
    preds["Markov switching"].append(w * mu_hi + (1 - w) * mu_lo)

    sc = StandardScaler().fit(tr_r)
    for name, proto in ML.items():
        m = clone(proto).fit(sc.transform(tr_r), tr_y)
        preds[name].append(m.predict(sc.transform(Xr[start:stop])))

truth = np.concatenate(truth)
bench = np.sqrt(np.mean((truth - np.concatenate(preds["Random walk"])) ** 2))
print(f"\n{'model':22s} {'RMSE':>8s} {'vs RW':>8s} {'dir.acc':>8s} {'corr':>7s}")
final = {}
for name, chunks in preds.items():
    p = np.concatenate(chunks)
    rmse = np.sqrt(np.mean((truth - p) ** 2))
    da = np.mean(np.sign(p) == np.sign(truth)) * 100
    cr = np.corrcoef(p, truth)[0, 1] if p.std() > 0 else 0.0
    final[name] = p
    print(f"{name:22s} {rmse:8.4f} {100*(rmse/bench-1):+7.2f}% "
          f"{da:7.1f}% {cr:+7.3f}")

# zoomed view: 60 days, returns, so the forecasts are visible
n = len(truth)
s = slice(n - 60, n)
fig, ax = plt.subplots(figsize=(6.5, 2.8))
ax.axhline(0, color="0.7", lw=0.8)
ax.bar(np.arange(60), truth[s], color="0.8", label="actual return")
for name in ["AR(1)", "Random forest", "SVR (RBF)"]:
    ax.plot(np.arange(60), final[name][s], lw=1.2, label=name)
ax.set_xlabel("Trading day (last 60 of the test period)")
ax.set_ylabel("Return (%)")
ax.legend(fontsize=7.5, ncol=4, loc="upper center")
fig.subplots_adjust(bottom=0.2)
fig.savefig("fig_zoom.pdf", bbox_inches="tight")
print(f"\nactual return range over those 60 days: "
      f"{truth[s].min():+.2f}% to {truth[s].max():+.2f}%")
for name in ["AR(1)", "Random forest", "SVR (RBF)"]:
    print(f"  {name:16s} forecast range {final[name][s].min():+.2f}% "
          f"to {final[name][s].max():+.2f}%")