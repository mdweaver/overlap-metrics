"""Illustration 8: worst-case bias of ANY outcome-linear estimator, split into
interpolation and extrapolation, for imputation and weighting estimators,
including energy balancing weights (Huling & Mak 2024).

For an estimator mu0_hat = sum_j w_j Y_j of the ATT counterfactual mean, the
worst-case conditional bias over 1-Lipschitz m0 is the Kantorovich-Rubinstein
norm of P*_n - Q_w (= W1 for non-negative weights). The optimal transport plan
attributes that cost to individual target units, so the bound splits into
  * interpolation : target units within c of the comparison data
                    (c = 3 x typical control spacing)
  * extrapolation : target units farther than c
  * leverage      : cost carried by negative weights (signed estimators only;
                    the price of "levering" a fitted surface beyond the data)
Average-case bias under a GP prior (RBF, length-scale 1) is the kernel MMD.

Energy balancing weights minimise the energy distance between the treated and
the weighted controls over the simplex. Energy distance is a squared MMD for
the distance-induced kernel (Sejdinovic et al. 2013), so energy balancing is a
kernel balancing method with no variance penalty (in Kallus's terms,
GOM(simplex, energy-kernel RKHS, lambda = 0)), and its worst-case bias over
any other class can be reported exactly like any other weights.
"""
import os, sys, io, contextlib, importlib.util
import numpy as np
import ot
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ess, implied_ols_weights, write_text

spec = importlib.util.spec_from_file_location(
    "f5", os.path.join(os.path.dirname(os.path.abspath(__file__)), "05_overlap_frontier.py"))
import matplotlib; matplotlib.use("Agg")
with contextlib.redirect_stdout(io.StringIO()):
    f5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(f5)


def kr_decomposed(target, donors, w, supported):
    """KR norm of P*_n - Q_w with per-source attribution."""
    n1 = len(target)
    wp, wm = np.clip(w, 0, None), np.clip(-w, 0, None)
    src = np.vstack([target, donors])
    a = np.concatenate([np.full(n1, 1.0 / n1), wm])
    mass = a.sum()
    ka, kb = a > 0, wp > 0
    M = cdist(src[ka], donors[kb])
    G = ot.emd(a[ka] / mass, wp[kb] / wp.sum(), M, numItermax=3_000_000) * mass
    row_cost = (G * M).sum(1)
    idx = np.where(ka)[0]
    is_target = idx < n1
    sup = np.zeros(len(idx), bool); sup[is_target] = supported[idx[is_target]]
    interp = row_cost[is_target & sup].sum()
    extrap = row_cost[is_target & ~sup].sum()
    lever = row_cost[~is_target].sum()
    return interp + extrap + lever, interp, extrap, lever


def energy_weights(X1, X0, iters=4000):
    """Exponentiated-gradient minimisation of the energy distance over the
    simplex (convex: -D00 is conditionally PSD for Euclidean distance)."""
    n0, n1 = len(X0), len(X1)
    D10, D00 = cdist(X1, X0), cdist(X0, X0)
    lin = 2 * D10.mean(0)                     # gradient of 2 a'D10 w
    w = np.full(n0, 1.0 / n0)
    for t in range(iters):
        g = lin - 2 * D00 @ w
        w = w * np.exp(-0.5 / np.sqrt(t + 1) * (g - g.min()))
        w /= w.sum()
    return w


def mmd_rbf(X1, X0, w, ell=1.0):
    k = lambda A, B: np.exp(-cdist(A, B, "sqeuclidean") / (2 * ell ** 2))
    a = np.full(len(X1), 1 / len(X1))
    return np.sqrt(max(a @ k(X1, X1) @ a - 2 * a @ k(X1, X0) @ w + w @ k(X0, X0) @ w, 0))


lines = ["# Illustration 8: worst-case Lipschitz bias split into interpolation / extrapolation / leverage",
         "# (per unit L, covariate-SD units); GP-RMS = average-case bias under RBF GP prior (amplitude 1, length-scale 1)",
         "scenario\testimator\tESS\tworst_case\tinterpolation\textrapolation\tleverage\tGP_RMS\tneg_mass\trealised_bias_m0=|x|"]
m0 = lambda X: np.linalg.norm(X, axis=1)
for name, (X0, X1) in f5.scen.items():
    n0, n1 = len(X0), len(X1)
    D00 = cdist(X0, X0); np.fill_diagonal(D00, np.inf)
    s0 = D00.min(1).mean()
    supported = cdist(X1, X0).min(1) <= 3 * s0
    W = {}
    W["OLS imputation (linear)"] = implied_ols_weights(np.column_stack([np.ones(n0), X0]),
                                                       np.r_[1.0, X1.mean(0)])
    P = PolynomialFeatures(2)
    W["OLS imputation (quadratic)"] = implied_ols_weights(P.fit_transform(X0),
                                                          P.transform(X1).mean(0))
    poly = PolynomialFeatures(2, include_bias=False)
    Z = poly.fit_transform(np.vstack([X0, X1]))
    e = LogisticRegression(C=1e4, max_iter=5000).fit(Z, np.r_[np.zeros(n0), np.ones(n1)]).predict_proba(
        poly.transform(X0))[:, 1]
    w = e / (1 - e); W["IPW (logit, quadratic)"] = w / w.sum()
    W["1-NN matching"] = np.bincount(cdist(X1, X0).argmin(1), minlength=n0) / n1
    W["energy balancing"] = energy_weights(X1, X0)
    lines.append(f"## {name}: share of treated beyond 3x spacing = {1 - supported.mean():.2f}")
    for lab, w in W.items():
        tot, i_, x_, l_ = kr_decomposed(X1, X0, w, supported)
        realised = m0(X1).mean() - w @ m0(X0)
        lines.append(f"{name}\t{lab}\t{1/np.sum(w**2):.1f}\t{tot:.3f}\t{i_:.3f}\t{x_:.3f}\t{l_:.3f}\t"
                     f"{mmd_rbf(X1, X0, w):.3f}\t{-w[w < 0].sum():.3f}\t{realised:.3f}")
write_text("08_bias_decomposition.tsv", "\n".join(lines) + "\n")
print("\n".join(lines))
