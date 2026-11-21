"""Shared helpers for the overlap / extrapolation illustrations.

Everything here is deliberately small and dependency-light (numpy, scipy,
scikit-learn, POT). Nothing here is meant to be a package; it supports the
numerical illustrations in docs/investigation.md.
"""
from __future__ import annotations

import os
import numpy as np
import ot  # Python Optimal Transport
from scipy.optimize import linprog
from scipy.spatial.distance import cdist

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "outputs")
os.makedirs(OUT, exist_ok=True)

# --- figure style (validated reference palette, light mode, first 3 slots) ---
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
VIOLET, RED = "#6250d6", "#e34948"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "axes.edgecolor": INK2,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "lines.linewidth": 2.0,
        "font.size": 9.5, "axes.titlesize": 10, "legend.frameon": False,
        "figure.dpi": 110,
    })


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print("wrote", os.path.relpath(path, os.path.dirname(HERE)))


def write_text(name, text):
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        f.write(text)
    print("wrote", os.path.relpath(path, os.path.dirname(HERE)))


# --- weights and effective sample size --------------------------------------
def ess(w):
    """Kish effective sample size of (non-negative) weights."""
    w = np.asarray(w, float)
    return w.sum() ** 2 / np.sum(w ** 2)


def implied_ols_weights(Phi_src, phi_target_mean, ridge=0.0):
    """Weights w (len n_src) such that the regression-imputation estimate of
    E_target[m(X)] equals sum_j w_j Y_j, when m is fit by (ridge) least squares
    of Y on features Phi in the source sample. Weights sum to one if Phi has an
    intercept column, and they exactly reproduce the target feature means."""
    A = Phi_src.T @ Phi_src + ridge * np.eye(Phi_src.shape[1])
    return Phi_src @ np.linalg.solve(A, phi_target_mean)


# --- convex hull membership (King & Zeng) ------------------------------------
def in_hull(points, X):
    """For each row of `points`, is it in the convex hull of rows of X?
    (An LP that hits the 10 s time limit is counted as "outside"; this only
    happens far from the hull in high dimension.)
    Solved as an LP feasibility problem: lambda >= 0, sum lambda = 1,
    X' lambda = point."""
    n, p = X.shape
    A_eq = np.vstack([X.T, np.ones((1, n))])
    out = np.zeros(len(points), bool)
    for i, z in enumerate(points):
        b_eq = np.append(z, 1.0)
        res = linprog(np.zeros(n), A_eq=A_eq, b_eq=b_eq, bounds=(0, None),
                      method="highs", options={"time_limit": 10.0})
        out[i] = res.status == 0
    return out


# --- random Tukey depth (Cuesta-Albertos & Nieto-Reyes 2008) -----------------
def random_tukey_depth(points, X, n_dir=500, rng=None):
    rng = np.random.default_rng(rng)
    U = rng.standard_normal((n_dir, X.shape[1]))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    PX = X @ U.T             # n x n_dir
    Pz = points @ U.T        # m x n_dir
    depth = np.empty(len(points))
    for i in range(len(points)):
        below = (PX <= Pz[i]).mean(0)
        above = (PX >= Pz[i]).mean(0)
        depth[i] = np.minimum(below, above).min()
    return depth


# --- distance-to-support and the overlap frontier ----------------------------
def nn_dist(points, X):
    return cdist(points, X).min(1)


def frontier(target, donors, k_grid, cost=None):
    """Bias-variance overlap frontier.

    For each k in k_grid, solve
        min_w  W1( P_target , sum_j w_j delta_{X_j} )
        s.t.   w_j >= 0, sum_j w_j = 1, w_j <= 1/k
    The cap w_j <= 1/k guarantees ESS(w) >= k. By Kantorovich-Rubinstein
    duality the objective is the worst-case conditional bias of the weighted
    estimator over all 1-Lipschitz outcome functions.

    Solved exactly as a balanced transport problem with a zero-cost dummy
    target that absorbs the spare donor capacity.
    Returns arrays (W1, ESS, weights_list)."""
    n1, n0 = len(target), len(donors)
    M = cdist(target, donors) if cost is None else cost
    res_w, res_ess, ws = [], [], []
    for k in k_grid:
        cap = 1.0 / k
        spare = n0 * cap - 1.0
        a = np.append(np.full(n1, 1.0 / n1), spare)
        b = np.full(n0, cap)
        Mx = np.vstack([M, np.zeros((1, n0))])
        G = ot.emd(a, b, Mx, numItermax=2_000_000)
        Greal = G[:n1]
        w = Greal.sum(0)
        res_w.append(float((Greal * M).sum()))
        res_ess.append(ess(w))
        ws.append(w)
    return np.array(res_w), np.array(res_ess), ws


def kr_bias(target, donors, w):
    """Worst-case bias over 1-Lipschitz functions for a SIGNED weight vector
    w summing to one (e.g. regression implied weights): the
    Kantorovich-Rubinstein norm of P_target - sum_j w_j delta_{X_j}.
    Computed as a transport problem from (P_target + w^-) to w^+."""
    wp, wm = np.clip(w, 0, None), np.clip(-w, 0, None)
    n1 = len(target)
    src = np.vstack([target, donors])
    a = np.concatenate([np.full(n1, 1.0 / n1), wm])
    b = wp
    mass = a.sum()
    keep_a, keep_b = a > 0, b > 0
    M = cdist(src[keep_a], donors[keep_b])
    G = ot.emd(a[keep_a] / mass, b[keep_b] / b.sum(), M, numItermax=2_000_000)
    return float((G * M).sum() * mass)
