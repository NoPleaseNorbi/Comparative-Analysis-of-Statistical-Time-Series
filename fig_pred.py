"""Per-model prediction figures: one-step-ahead forecast against the real
AAPL price, plus the same forecast seen as returns."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

L, REFIT, TRAIN0, SHOW = 10, 250, 1200, 250

MODELS = {
    "linear": ("Linear regression", LinearRegression()),
    "ridge":  ("Ridge regression", Ridge(alpha=10.0)),
    "lasso":  ("Lasso regression", Lasso(alpha=0.01, max_iter=5000)),
    "svr":    ("Support vector regression", SVR(kernel="rbf", C=1.0, epsilon=0.05)),
    "rf":     ("Random forest", RandomForestRegressor(n_estimators=200, max_depth=6,
                                                      random_state=0, n_jobs=-1)),
}

px = pd.read_csv("aapl.csv", index_col=0)["Close"].values
ret = np.r_[np.nan, np.diff(np.log(px))] * 100

X, y, idx = [], [], []
for t in range(L, len(ret) - 1):
    win = ret[t - L + 1:t + 1]
    if np.isnan(win).any():
        continue
    X.append(win); y.append(ret[t + 1]); idx.append(t + 1)
X, y, idx = np.array(X), np.array(y), np.array(idx)

for key, (label, proto) in MODELS.items():
    preds, truth, where = [], [], []
    for start in range(TRAIN0, len(X), REFIT):
        stop = min(start + REFIT, len(X))
        sc = StandardScaler().fit(X[:start])
        from sklearn.base import clone
        m = clone(proto).fit(sc.transform(X[:start]), y[:start])
        preds.append(m.predict(sc.transform(X[start:stop])))
        truth.append(y[start:stop]); where.append(idx[start:stop])
    pred, true, where = (np.concatenate(preds), np.concatenate(truth),
                         np.concatenate(where))

    rmse = np.sqrt(np.mean((true - pred) ** 2))
    da = np.mean(np.sign(pred) == np.sign(true)) * 100
    corr = np.corrcoef(pred, true)[0, 1]
    # one-step-ahead price forecast: yesterday's real price carried forward
    pred_px = px[where - 1] * np.exp(pred / 100)

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.5),
                             gridspec_kw={"width_ratios": [1.7, 1], "wspace": 0.3})
    s = slice(-SHOW, None)
    axes[0].plot(px[where][s], color="0.45", lw=1.0, label="actual price")
    axes[0].plot(pred_px[s], color="C1", lw=1.0, label="predicted")
    axes[0].set_title(f"{label}: last {SHOW} test days", fontsize=9.5)
    axes[0].set_xlabel("Trading day"); axes[0].set_ylabel("Price (USD)")
    axes[0].legend(fontsize=7.5, loc="upper left")

    axes[1].axhline(0, color="0.7", lw=0.7); axes[1].axvline(0, color="0.7", lw=0.7)
    axes[1].plot(true, pred, "o", ms=2, alpha=0.35, color="C0")
    axes[1].set_title(f"correlation ${corr:+.3f}$", fontsize=9.5)
    axes[1].set_xlabel("actual return (%)"); axes[1].set_ylabel("predicted (%)")

    fig.subplots_adjust(bottom=0.22)
    fig.savefig(f"fig_pred_{key}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"{label:28s} RMSE {rmse:.4f}  dir.acc {da:5.1f}%  corr {corr:+.3f}")

print(f"\nbenchmark (random walk): RMSE "
      f"{np.sqrt(np.mean((y[TRAIN0:] - y[:TRAIN0].mean())**2)):.4f}")