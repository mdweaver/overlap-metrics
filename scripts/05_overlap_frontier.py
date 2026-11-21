"""Illustration 5: the bias-variance overlap frontier (candidate metric).

For a target population P* (here: the treated, so the estimand is the ATT)
and donor units (controls), any estimator that is linear in the donors'
outcomes, tau_hat = mean(Y_treated) - sum_j w_j Y_j, has conditional error
    sum_j w_j m0(X_j) - E_{P*} m0(X)   (bias)   +   sum_j w_j eps_j   (noise).
If m0 is L-Lipschitz, the worst-case bias is L * W1(P*, Q_w), where
Q_w = sum_j w_j delta_{X_j} (Kantorovich-Rubinstein duality; for signed
weights the KR norm). The noise variance is sigma^2 * sum_j w_j^2 =
sigma^2 / ESS(w).

The FRONTIER is  B(k) = min { W1(P*, Q_w) : w >= 0, sum w = 1, ESS(w) >= k }
(computed with the sufficient cap w_j <= 1/k). It answers: "to keep at least
k effective donors, how much extrapolation per unit of outcome roughness must
I accept?" Its left end (k = 1) is the irreducible extrapolation
E_{P*}[ dist(X, donor sample) ]; its right end (k = n0) is W1 between the raw
groups.

Three 2-D scenarios (covariates in control-SD units):
  A  "shift": X0 ~ N(0, I), X1 ~ N((1, 0), I)        -> needs re-weighting
  B  "gap":   X0 ~ U([-2,1] x [-2,2]), X1 ~ U([0,2] x [-2,2])
             -> half the treated lie beyond the control support
  C  "hole":  X0 on annulus 1<=|x|<=2, X1 half in the hole, half on annulus
Estimators placed on the same axes: OLS imputation (signed implied weights),
logistic IPW (odds weights; quadratic terms), 1-NN matching.
Check: realised bias of each estimator for the 1-Lipschitz m0(x) = |x|_2
never exceeds its W1 / KR bound.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.spatial.distance import cdist
from common import (style, save, write_text, ess, frontier, kr_bias,
                    implied_ols_weights, BLUE, ORANGE, AQUA, INK2)

style()
rng = np.random.default_rng(5)
n0, n1 = 400, 150


def annulus(n, r0, r1, rng):
    r = np.sqrt(rng.uniform(r0 ** 2, r1 ** 2, n))
    a = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([r * np.cos(a), r * np.sin(a)])


scen = {
    "A shift": (rng.standard_normal((n0, 2)),
                rng.standard_normal((n1, 2)) + [1, 0]),
    "B gap": (np.column_stack([rng.uniform(-2, 1, n0), rng.uniform(-2, 2, n0)]),
              np.column_stack([rng.uniform(0, 2, n1), rng.uniform(-2, 2, n1)])),
    "C hole": (annulus(n0, 1, 2, rng),
               np.vstack([annulus(n1 // 2, 0, 0.7, rng),
                          annulus(n1 - n1 // 2, 1, 2, rng)])),
}
k_grid = [1, 1.5, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 300, 400]
m0 = lambda X: np.linalg.norm(X, axis=1)  # 1-Lipschitz outcome surface

lines = ["# Illustration 5: overlap frontier. W1 = worst-case bias per unit L"]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
for ax, (name, (X0, X1)) in zip(axes, scen.items()):
    W, E, _ = frontier(X1, X0, k_grid)
    # null calibration: the same floor when the target IS a control draw
    # (leave-one-out NN distance among controls); excess = floor - null
    D00 = cdist(X0, X0); np.fill_diagonal(D00, np.inf)
    null = D00.min(1).mean()
    target_mean = m0(X1).mean()
    # estimators
    pts = {}
    Phi0 = np.column_stack([np.ones(n0), X0])
    w = implied_ols_weights(Phi0, np.r_[1.0, X1.mean(0)])
    pts["OLS imputation"] = (1 / np.sum(w ** 2), kr_bias(X1, X0, w), w)
    poly = PolynomialFeatures(2, include_bias=False)
    Z = poly.fit_transform(np.vstack([X0, X1]))
    D = np.r_[np.zeros(n0), np.ones(n1)]
    e = LogisticRegression(C=1e4, max_iter=5000).fit(Z, D).predict_proba(
        poly.transform(X0))[:, 1]
    w = e / (1 - e); w /= w.sum()
    pts["IPW (logit, quadratic)"] = (ess(w), kr_bias(X1, X0, w), w)
    idx = cdist(X1, X0).argmin(1)
    w = np.bincount(idx, minlength=n0) / n1
    pts["1-NN matching"] = (ess(w), kr_bias(X1, X0, w), w)

    ax.plot(E, W, "-", color=INK2, lw=1.6, label="frontier (best possible)")
    ax.fill_between(E, W, 0, color=INK2, alpha=0.06, lw=0)
    for (lab, (es, b, w)), c in zip(pts.items(), [BLUE, ORANGE, AQUA]):
        ax.plot(es, b, "o", ms=8, color=c, mec="white", mew=1.5, label=lab)
    ax.set_xscale("log")
    ax.set_xticks([25, 50, 100, 200, 400]); ax.set_xticklabels(["25", "50", "100", "200", "400"])
    ax.minorticks_off(); ax.set_xlabel("effective number of controls (ESS)")
    ax.axhline(null, color=INK2, lw=0.8, ls=":")
    ax.text(E[-1], null, "null floor", fontsize=7, color=INK2, ha="right",
            va="bottom")
    ax.set_title(f"{name}: floor {W[0]:.2f} (null {null:.2f}), raw W1 {W[-1]:.2f}")
    lines.append(f"\n## {name}  (null floor = {null:.4f})\nk_cap\tESS\tW1")
    lines += [f"{k}\t{es:.1f}\t{wv:.4f}" for k, es, wv in zip(k_grid, E, W)]
    lines.append("estimator\tESS\tworst_case_bias_per_L\t"
                 "realised_bias_for_m0=|x|\tneg_weight_mass")
    for lab, (es, b, w) in pts.items():
        realised = target_mean - w @ m0(X0)
        lines.append(f"{lab}\t{es:.1f}\t{b:.4f}\t{realised:.4f}\t"
                     f"{-w[w < 0].sum():.3f}")
axes[0].set_ylabel("worst-case bias per unit Lipschitz L\n(W1, covariate-SD units)")
axes[0].legend(fontsize=7.5, loc="upper right")
fig.tight_layout(); save(fig, "05_frontier.png")
write_text("05_frontier.tsv", "\n".join(lines) + "\n")
print("\n".join(lines))
