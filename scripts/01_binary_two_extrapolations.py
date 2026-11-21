"""Illustration 1: how weak overlap turns imputation and re-weighting into
extrapolation (binary treatment, one covariate, target = ATT).

Design: X | D=0 ~ N(0,1), X | D=1 ~ N(delta,1). The true propensity is
logistic-linear, so the propensity model is CORRECTLY specified; overlap is
technically satisfied for every delta (Gaussian tails), but it weakens fast.
Theory: the ATT odds weights w(x) = e(x)/(1-e(x)) * P(D=0)/P(D=1) have
chi-square divergence chi2 = exp(delta^2) - 1, so the effective number of
controls is n0 / (1 + chi2) = n0 * exp(-delta^2).

Outcome: Y(0) = m0(X) + eps with m0 convex; Y(1) = Y(0) + 1 (ATT = 1).
Estimators of the ATT:
  * OLS imputation (fit Y ~ 1 + X on controls, impute Y(0) for treated)
  * IPW / odds weighting with estimated logistic propensity (Hajek)
  * 1-nearest-neighbour matching with replacement
Each is a linear smoother in the control outcomes: tau_hat = mean(Y_treated)
- sum_j w_j Y_j. The weights reveal the two faces of extrapolation:
regression puts NEGATIVE weight on some controls (it extrapolates a line),
IPW puts HUGE weight on a few controls (it extrapolates their outcomes).
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from common import (style, save, write_text, ess, implied_ols_weights,
                    BLUE, ORANGE, AQUA, INK2)

style()
rng = np.random.default_rng(1)
n0, n1, R = 500, 200, 300
m0 = lambda x: x + 0.3 * x ** 2


def draw(delta, rng):
    x0 = rng.standard_normal(n0)
    x1 = rng.standard_normal(n1) + delta
    y0 = m0(x0) + rng.standard_normal(n0)
    y1 = m0(x1) + 1 + rng.standard_normal(n1)
    return x0, x1, y0, y1


def weights(x0, x1):
    # OLS imputation weights
    Phi0 = np.column_stack([np.ones(n0), x0])
    w_ols = implied_ols_weights(Phi0, np.array([1.0, x1.mean()]))
    # estimated-propensity odds weights (Hajek normalised)
    X = np.concatenate([x0, x1])[:, None]
    D = np.r_[np.zeros(n0), np.ones(n1)]
    e = LogisticRegression(C=1e6).fit(X, D).predict_proba(x0[:, None])[:, 1]
    w_ipw = e / (1 - e)
    w_ipw /= w_ipw.sum()
    # 1-NN matching with replacement: weight = reuse count / n1
    idx = np.abs(x1[:, None] - x0[None, :]).argmin(1)
    w_nn = np.bincount(idx, minlength=n0) / n1
    nn_d = np.abs(x1 - x0[idx]).mean()
    return w_ols, w_ipw, w_nn, nn_d


deltas = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
rows = []
for d in deltas:
    est = {"OLS imputation": [], "IPW (odds)": [], "1-NN matching": []}
    diag = {"ess_ipw": [], "maxw_ipw": [], "neg_ols": [], "nn_d": [],
            "out_range": []}
    for r in range(R):
        x0, x1, y0, y1 = draw(d, rng)
        w_ols, w_ipw, w_nn, nn_d = weights(x0, x1)
        for name, w in zip(est, (w_ols, w_ipw, w_nn)):
            est[name].append(y1.mean() - w @ y0)
        diag["ess_ipw"].append(ess(w_ipw) / n0)
        diag["maxw_ipw"].append(w_ipw.max())
        diag["neg_ols"].append(-w_ols[w_ols < 0].sum())
        diag["nn_d"].append(nn_d)
        diag["out_range"].append(np.mean(x1 > x0.max()))
    row = {"delta": d, "ess_theory": np.exp(-d ** 2)}
    for k, v in est.items():
        v = np.array(v)
        row[k + " bias"] = v.mean() - 1
        row[k + " rmse"] = np.sqrt(np.mean((v - 1) ** 2))
    for k, v in diag.items():
        row[k] = float(np.mean(v))
    rows.append(row)

# ---- table ----
cols = list(rows[0].keys())
lines = ["# Illustration 1: ATT with weakening overlap (R=%d reps, n0=%d, n1=%d)"
         % (R, n0, n1), "\t".join(cols)]
for row in rows:
    lines.append("\t".join(f"{row[c]:.4f}" for c in cols))
write_text("01_binary_sweep.tsv", "\n".join(lines) + "\n")

# ---- figure A: what the weights look like at delta = 2 ----
x0, x1, y0, y1 = draw(2.0, np.random.default_rng(7))
w_ols, w_ipw, w_nn, _ = weights(x0, x1)
o = np.argsort(x0)
fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.2))
grid = np.linspace(-4, 6, 400)
phi = lambda z, mu: np.exp(-(z - mu) ** 2 / 2) / np.sqrt(2 * np.pi)
ax[0].fill_between(grid, phi(grid, 0), color=BLUE, alpha=0.25, lw=0)
ax[0].plot(grid, phi(grid, 0), color=BLUE, label="controls  X|D=0")
ax[0].fill_between(grid, phi(grid, 2), color=ORANGE, alpha=0.25, lw=0)
ax[0].plot(grid, phi(grid, 2), color=ORANGE, label="treated  X|D=1")
ax[0].set_title("(a) covariate densities, delta = 2")
ax[0].set_xlabel("X"); ax[0].legend(loc="upper left")
ax[1].axhline(0, color=INK2, lw=0.8)
ax[1].plot(x0[o], w_ols[o] * n0, color=BLUE, label="OLS imputation")
ax[1].plot(x0[o], w_ipw[o] * n0, color=ORANGE, label="IPW (odds)")
ax[1].set_ylim(-6, 40)
ax[1].set_title("(b) implied weight on each control (x n0)")
ax[1].set_xlabel("control's X"); ax[1].legend(loc="upper left")
ax[1].annotate("negative weights:\nregression extrapolates a line",
               xy=(-2.0, -3.5), xytext=(-3.2, 12), fontsize=8, color=INK2,
               arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
xs = np.array([r["delta"] for r in rows])
ax[2].plot(xs, [r["ess_ipw"] for r in rows], "o-", color=ORANGE, ms=4,
           label="IPW weights, simulated")
ax[2].plot(xs, [r["ess_theory"] for r in rows], "--", color=INK2, lw=1.2,
           label="theory  exp(-delta^2)")
ax[2].set_yscale("log"); ax[2].set_xlabel("delta (mean shift in SDs)")
ax[2].set_title("(c) effective share of controls used, ESS / n0")
ax[2].legend(loc="lower left")
fig.tight_layout(); save(fig, "01_weights.png")

# ---- figure B: estimator error and diagnostics vs delta ----
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
for name, c in zip(["OLS imputation", "IPW (odds)", "1-NN matching"],
                   [BLUE, ORANGE, AQUA]):
    ax[0].plot(xs, [r[name + " rmse"] for r in rows], "o-", color=c, ms=4,
               label=name)
ax[0].set_yscale("log"); ax[0].set_xlabel("delta")
ax[0].set_title("(a) RMSE of the ATT estimate (truth = 1)")
ax[0].legend(loc="upper left")
ax[1].plot(xs, [r["neg_ols"] for r in rows], "o-", color=BLUE, ms=4,
           label="negative weight mass, OLS")
ax[1].plot(xs, [r["nn_d"] for r in rows], "o-", color=AQUA, ms=4,
           label="mean NN distance (Lipschitz bias bound / L)")
ax[1].plot(xs, [r["out_range"] for r in rows], "o-", color=INK2, ms=4,
           label="share of treated beyond max control X")
ax[1].set_xlabel("delta"); ax[1].set_title("(b) extrapolation diagnostics")
ax[1].legend(loc="upper left")
fig.tight_layout(); save(fig, "01_sweep.png")
