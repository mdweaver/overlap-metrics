"""Illustration 3: overlap for a continuous treatment (dose-response).

Target: the average dose-response mu(t) = E[Y(t)] = E_X[ m(t, X) ].
Positivity for mu(t) requires f(t | x) > 0 (ideally >= c > 0) for P_X-almost
every x: the vertical line {t} x supp(X) must lie inside the joint support of
(T, X). The marginal support of T being wide says nothing about this.

Two designs with the same Var(T | X):
  * "band":     T = X + U,  U ~ Uniform(-1, 1)   -> f(t|x) = 1/2 on |t-x|<=1,
                 0 outside. For EVERY t, units with |X - t| > 1 have no
                 support: strict positivity fails at every dose.
  * "gaussian": T = X + V,  V ~ N(0, 1/3)        -> f(t|x) > 0 everywhere but
                 tiny far from the diagonal (weak overlap).
X ~ N(0,1). Outcome m(t, x) = sin(1.5 t) + x + 0.5 t x, so mu(t) = sin(1.5 t).

Estimators of mu(t):
  * linear outcome model  Y ~ T + X  (imputation; misspecified)
  * cubic outcome model in (T, X) with interactions (imputation; flexible)
  * kernel-smoothed Hajek IPW with the TRUE generalised propensity density
    (weights K_h(T_i - t) / f(T_i | X_i))
Diagnostics, per target dose t:
  * non-overlap share  pi(t) = P_X( f(t|X) < eps )   (eps = 0.05)
  * kernel ESS of the IPW weights at t
  * Lipschitz extrapolation floor: mean over units of the distance from
    (X_i, t) to the nearest observed (X_j, T_j), cost |dx| + lambda |dt|.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.spatial.distance import cdist
from common import (style, save, write_text, ess, frontier,
                    BLUE, ORANGE, AQUA, INK2, GRID)

style()
rng = np.random.default_rng(3)
s_g = np.sqrt(1 / 3)
m = lambda t, x: np.sin(1.5 * t) + x + 0.5 * t * x
mu = lambda t: np.sin(1.5 * t)
tgrid = np.linspace(-2.5, 2.5, 21)
eps, h = 0.05, 0.25


def draw(n, design, rng):
    x = rng.standard_normal(n)
    if design == "band":
        t = x + rng.uniform(-1, 1, n)
        f = np.where(np.abs(t - x) <= 1, 0.5, 0.0)
    else:
        t = x + s_g * rng.standard_normal(n)
        f = norm.pdf(t, loc=x, scale=s_g)
    y = m(t, x) + 0.5 * rng.standard_normal(n)
    return x, t, f, y


def gps(t, x, design):
    if design == "band":
        return np.where(np.abs(t - x) <= 1, 0.5, 0.0)
    return norm.pdf(t, loc=x, scale=s_g)


def estimates(x, t, f, y):
    n = len(x)
    # linear
    Z = np.column_stack([np.ones(n), t, x])
    b = np.linalg.lstsq(Z, y, rcond=None)[0]
    lin = np.array([b[0] + b[1] * s + b[2] * x.mean() for s in tgrid])
    # cubic with interactions
    def feats(tt, xx):
        return np.column_stack([tt ** a * xx ** c for a in range(4)
                                for c in range(4) if a + c <= 3])
    bc = np.linalg.lstsq(feats(t, x), y, rcond=None)[0]
    cub = np.array([(feats(np.full(n, s), x) @ bc).mean() for s in tgrid])
    # kernel Hajek IPW with true GPS
    ipw, kess = [], []
    for s in tgrid:
        k = norm.pdf((t - s) / h)
        w = k / f
        ipw.append(np.sum(w * y) / np.sum(w) if w.sum() > 0 else np.nan)
        kess.append(ess(w))
    return lin, cub, np.array(ipw), np.array(kess)


R, n = 200, 1000
summary = {}
for design in ["band", "gaussian"]:
    L, C, I, K = [], [], [], []
    for r in range(R):
        x, t, f, y = draw(n, design, rng)
        lin, cub, ipw, kess = estimates(x, t, f, y)
        L.append(lin); C.append(cub); I.append(ipw); K.append(kess)
    L, C, I, K = map(np.array, (L, C, I, K))
    xbig = rng.standard_normal(200_000)
    pi = np.array([np.mean(gps(np.full_like(xbig, s), xbig, design) < eps)
                   for s in tgrid])
    # Lipschitz extrapolation floor (lambda = 1) on one sample of n = 600
    x, t, f, y = draw(600, design, np.random.default_rng(30))
    donors = np.column_stack([x, t])
    floor1, floor30 = [], []
    for s in tgrid:
        target = np.column_stack([x, np.full_like(x, s)])
        M = cdist(target, donors, "cityblock")
        W, E, _ = frontier(target, donors, [1, 30], cost=M)
        floor1.append(W[0]); floor30.append(W[1])
    summary[design] = dict(
        bias_lin=np.nanmean(L, 0) - mu(tgrid),
        bias_cub=np.nanmean(C, 0) - mu(tgrid),
        bias_ipw=np.nanmean(I, 0) - mu(tgrid),
        sd_ipw=np.nanstd(I, 0), sd_cub=np.nanstd(C, 0),
        kess=K.mean(0), pi=pi, floor1=np.array(floor1),
        floor30=np.array(floor30))

lines = ["design\tt\tmu\tpi_nonoverlap\tkernel_ess\tbias_linear\tbias_cubic\t"
         "sd_cubic\tbias_ipw\tsd_ipw\tLip_floor_ess1\tLip_floor_ess30"]
for design, S in summary.items():
    for i, s in enumerate(tgrid):
        lines.append(f"{design}\t{s:.2f}\t{mu(s):.3f}\t{S['pi'][i]:.3f}\t"
                     f"{S['kess'][i]:.1f}\t{S['bias_lin'][i]:.3f}\t"
                     f"{S['bias_cub'][i]:.3f}\t{S['sd_cub'][i]:.3f}\t"
                     f"{S['bias_ipw'][i]:.3f}\t{S['sd_ipw'][i]:.3f}\t"
                     f"{S['floor1'][i]:.3f}\t{S['floor30'][i]:.3f}")
# modified treatment policy T -> T + delta: share of units pushed out of
# support (band design) -- a target that needs only LOCAL support.
lines.append("\n# shift policy T -> T + delta in the band design: share of "
             "units whose shifted dose leaves the support = delta / 2")
for d in [0.1, 0.25, 0.5, 1.0]:
    x, t, f, y = draw(200_000, "band", rng)
    lines.append(f"delta={d}\tshare_out={np.mean(np.abs(t + d - x) > 1):.3f}")
write_text("03_continuous.tsv", "\n".join(lines) + "\n")

# ---- figure ----
fig, ax = plt.subplots(1, 4, figsize=(14, 3.3))
x, t, f, y = draw(1500, "band", np.random.default_rng(5))
ax[0].scatter(x, t, s=4, color=BLUE, alpha=0.35, lw=0)
xs = np.linspace(-3.2, 3.2, 10)
ax[0].plot(xs, xs + 1, color=INK2, lw=0.8); ax[0].plot(xs, xs - 1, color=INK2, lw=0.8)
ax[0].axhline(1.5, color=ORANGE, lw=1.2)
ax[0].plot([0.5, 2.5], [1.5, 1.5], color=ORANGE, lw=4)
ax[0].text(-3.1, 1.75, "target dose t = 1.5:\nonly X in [0.5, 2.5] supported",
           fontsize=7.5, color=INK2)
ax[0].set_xlabel("X"); ax[0].set_ylabel("T")
ax[0].set_title("(a) band design: joint support of (X, T)")
for design, ls, lab in [("band", "-", "band (bounded noise)"),
                        ("gaussian", "--", "gaussian (unbounded)")]:
    S = summary[design]
    ax[1].plot(tgrid, S["pi"], ls, color=ORANGE, label=lab)
    ax[3].plot(tgrid, S["floor1"], ls, color=AQUA, label=lab + ", ESS>=1")
    ax[3].plot(tgrid, S["floor30"], ls, color=INK2, lw=1.2,
               label=lab + ", ESS>=30")
S = summary["band"]
ax[2].axhline(0, color=INK2, lw=0.8)
ax[2].plot(tgrid, S["bias_lin"], color=BLUE, label="linear imputation")
ax[2].plot(tgrid, S["bias_cub"], color=AQUA, label="cubic imputation")
ax[2].plot(tgrid, S["bias_ipw"], color=ORANGE, label="kernel IPW, true GPS")
ax[1].set_title("(b) share of population with f(t|x) < 0.05")
ax[1].set_xlabel("target dose t"); ax[1].legend(fontsize=7.5)
ax[2].set_title("(c) bias of mu(t) estimates, band design")
ax[2].set_xlabel("target dose t"); ax[2].legend(fontsize=7.5)
ax[3].set_title("(d) Lipschitz extrapolation floor (per unit L)")
ax[3].set_xlabel("target dose t"); ax[3].legend(fontsize=7)
fig.tight_layout(); save(fig, "03_continuous.png")
