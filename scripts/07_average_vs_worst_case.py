"""Illustration 7: average-case (Gaussian-process prior) bias vs worst-case
Lipschitz bias for the weights of Illustration 5. Under f ~ GP(0, k), the RMS
bias of weights w is exactly the kernel MMD between the target and Q_w."""
import sys; import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, importlib.util
from scipy.spatial.distance import cdist
spec = importlib.util.spec_from_file_location("f5", os.path.join(os.path.dirname(os.path.abspath(__file__)), "05_overlap_frontier.py"))
import matplotlib; matplotlib.use("Agg")
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    f5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(f5)
from common import implied_ols_weights, kr_bias, ess
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
rng = np.random.default_rng(0)
ell = 1.0
k = lambda A, B: np.exp(-cdist(A, B, "sqeuclidean") / (2 * ell**2))
for name, (X0, X1) in f5.scen.items():
    n0, n1 = len(X0), len(X1)
    a = np.full(n1, 1 / n1)
    W = {}
    Phi0 = np.column_stack([np.ones(n0), X0]); W["OLS"] = implied_ols_weights(Phi0, np.r_[1.0, X1.mean(0)])
    poly = PolynomialFeatures(2, include_bias=False); Z = poly.fit_transform(np.vstack([X0, X1]))
    e = LogisticRegression(C=1e4, max_iter=5000).fit(Z, np.r_[np.zeros(n0), np.ones(n1)]).predict_proba(poly.transform(X0))[:, 1]
    w = e / (1 - e); W["IPW"] = w / w.sum()
    W["1-NN"] = np.bincount(cdist(X1, X0).argmin(1), minlength=n0) / n1
    # supported vs unsupported target units: NN distance > 3x null floor
    D00 = cdist(X0, X0); np.fill_diagonal(D00, np.inf); null = D00.min(1).mean()
    unsup = cdist(X1, X0).min(1) > 3 * null
    # GP draws
    Xall = np.vstack([X1, X0]); Kall = k(Xall, Xall) + 1e-8 * np.eye(n0 + n1)
    Lc = np.linalg.cholesky(Kall); F = Lc @ rng.standard_normal((n0 + n1, 2000))
    print(f"\n{name}: unsupported share of target = {unsup.mean():.2f}")
    print("est    worst-case/L   GP RMS bias (theory)  GP RMS (sim)")
    for lab, w in W.items():
        mmd2 = a @ k(X1, X1) @ a - 2 * a @ k(X1, X0) @ w + w @ k(X0, X0) @ w
        bias = a @ F[:n1] - w @ F[n1:]
        print(f"{lab:5s}  {kr_bias(X1, X0, w):10.3f}   {np.sqrt(mmd2):12.3f}   {np.sqrt(np.mean(bias**2)):12.3f}")
