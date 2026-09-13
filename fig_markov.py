"""Section 2.10 - Markov switching. Two-regime model fitted by maximum
likelihood using the Hamilton filter."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def load(name):
    px = pd.read_csv(name, index_col=0)["Close"]
    px.index = pd.to_datetime(px.index, utc=True)
    return np.log(px).diff().dropna() * 100


def unpack(theta):
    p11, p22 = 1 / (1 + np.exp(-theta[0])), 1 / (1 + np.exp(-theta[1]))
    mu = theta[2:4]
    sig = np.exp(theta[4:6])
    return p11, p22, mu, sig


def hamilton_filter(r, theta):
    """Returns (negative log-likelihood, filtered P(state = 1))."""
    p11, p22, mu, sig = unpack(theta)
    P = np.array([[p11, 1 - p11], [1 - p22, p22]])
    pi = np.array([(1 - p22) / (2 - p11 - p22), (1 - p11) / (2 - p11 - p22)])
    ll, filt = 0.0, np.empty((len(r), 2))
    for t, x in enumerate(r):
        dens = np.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
        joint = pi * dens
        lik = joint.sum()
        if lik <= 0:
            return 1e10, filt
        ll += np.log(lik)
        pi_post = joint / lik
        filt[t] = pi_post
        pi = P.T @ pi_post
    return -ll, filt


r = load("aapl.csv")
x = r.values
start = np.array([2.5, 2.5, 0.1, -0.1, np.log(1.0), np.log(3.0)])
best = minimize(lambda th: hamilton_filter(x, th)[0], start,
                method="Nelder-Mead",
                options={"maxiter": 20000, "maxfev": 20000, "fatol": 1e-6})
p11, p22, mu, sig = unpack(best.x)
_, filt = hamilton_filter(x, best.x)

hi = int(np.argmax(sig))                       # index of the turbulent regime
prob_hi = filt[:, hi]

print(f"regime 1: mean {mu[0]:+.3f}%  sd {sig[0]:.3f}%")
print(f"regime 2: mean {mu[1]:+.3f}%  sd {sig[1]:.3f}%")
print(f"stay probabilities: p11 {p11:.4f}   p22 {p22:.4f}")
print(f"expected duration: regime 1 {1/(1-p11):.1f} days,"
      f" regime 2 {1/(1-p22):.1f} days")
print(f"days assigned to the turbulent regime (P>0.5): "
      f"{(prob_hi > 0.5).sum()} of {len(x)} ({100*(prob_hi>0.5).mean():.1f}%)")
covid = (r.index >= "2020-02-20") & (r.index <= "2020-04-30")
print(f"mean turbulent probability, 20 Feb - 30 Apr 2020: {prob_hi[covid].mean():.2f}")
print(f"mean turbulent probability, rest of sample:       {prob_hi[~covid].mean():.2f}")

fig, axes = plt.subplots(2, 1, figsize=(6.5, 3.6), sharex=True,
                         gridspec_kw={"height_ratios": [1.2, 1], "hspace": 0.35})
axes[0].axhline(0, color="0.3", lw=0.7)
axes[0].plot(x, lw=0.5, color="0.55")
axes[0].set_ylabel("Return (%)")
axes[0].set_title("AAPL daily returns", fontsize=9.5)

axes[1].fill_between(np.arange(len(x)), 0, prob_hi, color="C1", alpha=0.35, lw=0)
axes[1].plot(prob_hi, lw=0.7, color="C1")
axes[1].set_ylim(0, 1.02)
axes[1].set_ylabel("$P(\\mathrm{turbulent})$")
axes[1].set_xlabel("Trading day $t$")
axes[1].set_title("Filtered probability of the turbulent regime", fontsize=9.5)

fig.subplots_adjust(bottom=0.15)
fig.savefig("fig_markov.pdf", bbox_inches="tight")