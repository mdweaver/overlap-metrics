"""Illustration 6: the hybrid analysis, i.e. split the target into supported /
unsupported units, report results as CURVES over the support cutoff and the
assumed outcome roughness, and estimate the roughness from comparison-arm
outcomes.

Estimand: ATT. Covariates in R^2 (comparable units).
  controls  X ~ U([-2, 1] x [-2, 2])   n0 = 400
  treated   X ~ U([ 0, 2] x [-2, 2])   n1 = 200
  -> treated units with x1 > 1 have no comparable controls.
Effect tau(x) = 0.25 + 0.25 x1 (heterogeneous, so trimming changes the estimand).
Two untreated-outcome surfaces that are IDENTICAL where controls exist
(x1 <= 1), so the control data cannot tell them apart:
  "continues":  m0(x) = x1 + 0.5 x2                      (gradient norm 1.12)
  "bends":      m0(x) = x1 + 0.5 x2 + 1.5 max(x1-1, 0)^2  (up to 4.03 in gap)
Noise sd 0.5.

Steps
 1. Support split. d_i = distance from treated unit i to its nearest control.
    Cutoff c = m * s0, where s0 = mean control-to-control NN distance (the
    typical spacing of the comparison data). Supported S(c) = {d_i <= c}.
 2. Comparison-arm outcome model: GP regression (ARD RBF + noise) fitted on
    CONTROL outcomes only. Never touches treated outcomes.
 3. Empirical roughness: gradient norm of the fitted GP mean over the control
    support (median, 90th percentile, max), with a bootstrap over controls.
 4. Results as curves:
    (a) EXCLUDE: ATT on the supported subpopulation S(c), with share excluded.
    (b) KEEP + BOUND: full-population ATT. Supported units are imputed by the
        GP mean (data speak there). Unsupported units get the sharp Lipschitz
        identified set for m0(x): [max_j m_j - L d(x,X_j), min_j m_j + L d(x,X_j)]
        (McShane-Whitney extension of the fitted control values), i.e. worst
        case only where the data cannot speak. Shown vs. L for several cutoffs.
    (c) Map over (cutoff, L) of the 95% lower bound of the full ATT, with the
        breakdown curve (lower bound = 0) and the empirical roughness.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from common import style, save, write_text, BLUE, ORANGE, AQUA, INK2, GRID

style()
rng = np.random.default_rng(6)
n0, n1, sd = 400, 200, 0.5
X0 = np.column_stack([rng.uniform(-2, 1, n0), rng.uniform(-2, 2, n0)])
X1 = np.column_stack([rng.uniform(0, 2, n1), rng.uniform(-2, 2, n1)])
tau = lambda X: 0.25 + 0.25 * X[:, 0]
surfaces = {
    "continues": lambda X: X[:, 0] + 0.5 * X[:, 1],
    "bends": lambda X: X[:, 0] + 0.5 * X[:, 1] + 1.5 * np.clip(X[:, 0] - 1, 0, None) ** 2,
}
true_grad = {
    "continues": lambda X: np.full(len(X), np.hypot(1, 0.5)),
    "bends": lambda X: np.hypot(1 + 3 * np.clip(X[:, 0] - 1, 0, None), 0.5),
}
e0, e1 = sd * rng.standard_normal(n0), sd * rng.standard_normal(n1)
Y0 = surfaces["continues"](X0) + e0          # identical under both surfaces
Y1 = {k: f(X1) + tau(X1) + e1 for k, f in surfaces.items()}
att_true = tau(X1).mean()

# ---------- 1. support split ----------
D10 = cdist(X1, X0)
d = D10.min(1)
D00 = cdist(X0, X0); np.fill_diagonal(D00, np.inf)
s0 = D00.min(1).mean()


# ---------- 2. GP on control outcomes ----------
def fit_gp(X, y, kernel=None, restarts=3):
    k = kernel or (C(1.0) * RBF([1.0, 1.0], (1e-2, 1e2)) + WhiteKernel(0.1))
    gp = GaussianProcessRegressor(k, normalize_y=False,
                                  n_restarts_optimizer=restarts,
                                  optimizer="fmin_l_bfgs_b" if restarts >= 0 else None,
                                  random_state=0)
    ybar = y.mean()
    gp.fit(X, y - ybar)
    return gp, ybar


def gp_mean_grad(gp, Xtr, Z):
    """Posterior mean gradient for an ARD RBF kernel at points Z."""
    k1 = gp.kernel_.k1                       # C * RBF
    ell = np.atleast_1d(k1.k2.length_scale)
    Kz = k1(Z, Xtr)                          # m x n
    diff = (Z[:, None, :] - Xtr[None, :, :]) / ell ** 2
    return -(Kz[:, :, None] * diff * gp.alpha_[None, :, None]).sum(1)


gp, ybar = fit_gp(X0, Y0)
m_ctrl = gp.predict(X0) + ybar
m_trt, sd_trt = gp.predict(X1, return_std=True); m_trt += ybar
gnorm = np.linalg.norm(gp_mean_grad(gp, X0, X0), axis=1)
# The control data REFUTE any L below the Lipschitz constant of the fitted
# control values: the Lipschitz identified set is then empty.
Dpos = D00.copy(); Dpos[~np.isfinite(Dpos)] = np.nan
L_refute = np.nanmax(np.abs(m_ctrl[:, None] - m_ctrl[None, :]) / Dpos)
L_med, L_90, L_max = np.median(gnorm), np.quantile(gnorm, 0.9), gnorm.max()

# ---------- 3. bootstrap the roughness estimates ----------
boot = []
for b in range(40):
    idx = rng.integers(0, n0, n0)
    gb, _ = fit_gp(X0[idx], Y0[idx], kernel=gp.kernel_, restarts=0)
    g = np.linalg.norm(gp_mean_grad(gb, X0[idx], X0), axis=1)
    boot.append([np.median(g), np.quantile(g, 0.9), g.max()])
boot = np.array(boot)
ci = np.quantile(boot, [0.05, 0.95], axis=0)
amp = np.sqrt(gp.kernel_.k1.k1.constant_value)
ells = gp.kernel_.k1.k2.length_scale
noise_sd = np.sqrt(gp.kernel_.k2.noise_level)


# ---------- 4. curves ----------
def lipschitz_set(L, idx):
    """Sharp Lipschitz bounds for m0 at treated units idx, from fitted
    control values (McShane-Whitney upper/lower extensions)."""
    Dsub = D10[idx]
    lo = (m_ctrl[None, :] - L * Dsub).max(1)
    hi = (m_ctrl[None, :] + L * Dsub).min(1)
    return lo, hi


mults = np.array([1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 16])
Lgrid = np.linspace(0.25, 6, 47)
res = {k: {} for k in surfaces}
for k in surfaces:
    tau_i = Y1[k] - m_trt                      # imputation where supported
    for m in mults:
        S = d <= m * s0
        U = ~S
        # (a) exclude
        att_S = tau_i[S].mean()
        se_S = tau_i[S].std(ddof=1) / np.sqrt(S.sum())
        # (b) keep + bound
        se_full = tau_i[S].std(ddof=1) / np.sqrt(n1)
        lo_b, hi_b = [], []
        for L in Lgrid:
            if L < L_refute:
                lo_b.append(np.nan); hi_b.append(np.nan); continue
            if U.any():
                mlo, mhi = lipschitz_set(L, np.where(U)[0])
                mhi = np.maximum(mhi, mlo)    # guard: L below fitted roughness
                part = tau_i[S].sum()
                lo_b.append((part + (Y1[k][U] - mhi).sum()) / n1 - 1.96 * se_full)
                hi_b.append((part + (Y1[k][U] - mlo).sum()) / n1 + 1.96 * se_full)
            else:
                lo_b.append(tau_i.mean() - 1.96 * se_full)
                hi_b.append(tau_i.mean() + 1.96 * se_full)
        res[k][m] = dict(share_out=U.mean(), att_S=att_S, se_S=se_S,
                         truth_S=tau(X1[S]).mean(), lo=np.array(lo_b),
                         hi=np.array(hi_b))

# breakdown roughness: smallest L at which the 95% lower bound reaches 0
def breakdown(k, m):
    lo = res[k][m]["lo"]
    j = np.where(lo <= 0)[0]
    return Lgrid[j[0]] if len(j) else np.inf

# ---------- text output ----------
lines = [
    "# Illustration 6: hybrid support split + sensitivity curves + empirical roughness",
    f"true full ATT = {att_true:.3f};  typical control spacing s0 = {s0:.3f}",
    f"GP on controls: amplitude {amp:.2f}, length-scales {np.round(ells, 2).tolist()}, noise sd {noise_sd:.2f}",
    "empirical roughness = gradient norm of GP mean over control support (true value 1.12 in the supported region)",
    f"  median {L_med:.2f} [90% boot {ci[0,0]:.2f}, {ci[1,0]:.2f}]",
    f"  90th pct {L_90:.2f} [{ci[0,1]:.2f}, {ci[1,1]:.2f}]",
    f"  max {L_max:.2f} [{ci[0,2]:.2f}, {ci[1,2]:.2f}]",
    f"  smallest L consistent with the fitted control values (data refute any smaller L): {L_refute:.2f}",
    "  true max gradient in the unsupported gap: continues 1.12, bends 4.03",
    "",
    "surface\tcutoff_mult\tcutoff\tshare_excluded\tATT_supported\tSE\ttrue_ATT_supported\t"
    "full_ATT_95lo@Lhat\tfull_ATT_95hi@Lhat\tfull_ATT_95lo@2xLhat\tfull_ATT_95hi@2xLhat\tbreakdown_L",
]
L_hat = max(L_90, L_refute)   # report at the smallest unrefuted estimate
j1 = np.searchsorted(Lgrid, L_hat); j2 = np.searchsorted(Lgrid, 2 * L_hat)
for k in surfaces:
    for m in mults:
        r = res[k][m]
        lines.append(f"{k}\t{m}\t{m*s0:.3f}\t{r['share_out']:.3f}\t{r['att_S']:.3f}\t"
                     f"{r['se_S']:.3f}\t{r['truth_S']:.3f}\t{r['lo'][j1]:.3f}\t"
                     f"{r['hi'][j1]:.3f}\t{r['lo'][j2]:.3f}\t{r['hi'][j2]:.3f}\t"
                     f"{breakdown(k, m):.2f}")
write_text("06_hybrid.tsv", "\n".join(lines) + "\n")
print("\n".join(lines[:9]))

# ---------- figure ----------
fig = plt.figure(figsize=(13, 7.2))
gs = fig.add_gridspec(2, 3)
# (a) support split at m = 3
ax = fig.add_subplot(gs[0, 0])
S3 = d <= 3 * s0
ax.scatter(*X0.T, s=5, color=BLUE, alpha=0.5, lw=0, label="controls")
ax.scatter(*X1[S3].T, s=14, color=AQUA, lw=0, label="treated, supported")
ax.scatter(*X1[~S3].T, s=14, color=ORANGE, lw=0, label="treated, unsupported")
ax.axvline(1, color=INK2, lw=0.8, ls=":")
ax.set_title(f"(a) support split at cutoff = 3 x spacing ({3*s0:.2f})")
ax.legend(fontsize=7, loc="lower left", bbox_to_anchor=(0, -0.38), ncol=3)
ax.set_aspect("equal")
# (b) exclude: share excluded vs cutoff
ax = fig.add_subplot(gs[0, 1])
so = [res["continues"][m]["share_out"] for m in mults]
ax.plot(mults, so, "o-", color=INK2, ms=4)
ax.set_xscale("log"); ax.set_xticks([1, 2, 4, 8, 16]); ax.set_xticklabels(["1", "2", "4", "8", "16"])
ax.minorticks_off()
ax.set_xlabel("support cutoff (multiples of typical control spacing)")
ax.set_title("(b) share of treated excluded as unsupported")
# (c) exclude: trimmed ATT vs cutoff
ax = fig.add_subplot(gs[0, 2])
for k, c, mk in [("continues", BLUE, "o"), ("bends", ORANGE, "s")]:
    a = np.array([res[k][m]["att_S"] for m in mults])
    s = np.array([res[k][m]["se_S"] for m in mults])
    ax.errorbar(mults * (1.03 if k == "bends" else 0.97), a, 1.96 * s, fmt=mk + "-",
                color=c, ms=4, lw=1.4, capsize=2, label=f"estimate, m0 {k}")
ax.plot(mults, [res["continues"][m]["truth_S"] for m in mults], color=INK2, lw=1.2,
        ls="--", label="true ATT of the retained units")
ax.axhline(att_true, color=INK2, lw=0.8, ls=":")
ax.text(16, att_true, "full-population ATT", fontsize=7, color=INK2, ha="right", va="bottom")
ax.set_xscale("log"); ax.set_xticks([1, 2, 4, 8, 16]); ax.set_xticklabels(["1", "2", "4", "8", "16"])
ax.minorticks_off()
ax.set_xlabel("support cutoff (multiples of typical control spacing)")
ax.set_title("(c) EXCLUDE: ATT of supported treated (95% CI)")
ax.legend(fontsize=7, loc="upper left")
# (d),(e) keep + bound vs L, per surface
for col, k in [(0, "continues"), (1, "bends")]:
    ax = fig.add_subplot(gs[1, col])
    ax.axvspan(0, L_refute, color=INK2, alpha=0.10, lw=0)
    ax.text(L_refute / 2, 2.9, "refuted\nby control\ndata", fontsize=7, color=INK2, ha="center", va="top")
    ax.axvspan(ci[0, 1], ci[1, 1], color=AQUA, alpha=0.15, lw=0)
    ax.axvline(L_90, color=AQUA, lw=1.2)
    ax.text(ci[1, 1], 2.0, " estimated\n roughness\n (90th pct)", fontsize=7, color=INK2, va="top")
    for m, c in [(1.5, ORANGE), (3, BLUE), (8, INK2)]:
        r = res[k][m]
        ax.fill_between(Lgrid, r["lo"], r["hi"], color=c, alpha=0.12, lw=0)
        ax.plot(Lgrid, r["lo"], color=c, lw=1.4,
                label=f"cutoff {m:g} x spacing ({100*r['share_out']:.0f}% unsupported)")
        ax.plot(Lgrid, r["hi"], color=c, lw=1.4)
    ax.axhline(att_true, color=INK2, lw=1, ls="--")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_ylim(-2.5, 3)
    ax.set_xlabel("assumed roughness L (outcome per covariate unit)")
    ax.set_xlim(0, 6)
    ax.set_title(f"({'d' if col == 0 else 'e'}) KEEP + BOUND, full ATT; m0 {k} (true ATT dashed)")
    if col == 0:
        ax.legend(fontsize=7, loc="lower left")
# (f) map of 95% lower bound over (cutoff, L), surface "bends"
ax = fig.add_subplot(gs[1, 2])
k = "bends"
Z = np.ma.masked_invalid(np.array([res[k][m]["lo"] for m in mults]))
im = ax.contourf(Lgrid, mults, Z, levels=np.linspace(-3, 1, 17), cmap="Blues")
cs = ax.contour(Lgrid, mults, Z, levels=[0], colors=[ORANGE], linewidths=2)
ax.clabel(cs, fmt={0: "lower bound = 0"}, fontsize=7)
ax.axvline(L_90, color=AQUA, lw=1.2)
ax.set_yscale("log"); ax.set_yticks([1, 2, 4, 8, 16]); ax.set_yticklabels(["1", "2", "4", "8", "16"])
ax.minorticks_off()
ax.axvspan(0, L_refute, color=INK2, alpha=0.10, lw=0)
ax.set_xlim(0, 6)
ax.set_xlabel("assumed roughness L"); ax.set_ylabel("support cutoff (x spacing)")
ax.set_title("(f) 95% lower bound of full ATT, m0 bends")
fig.colorbar(im, ax=ax, fraction=0.05)
fig.tight_layout(); save(fig, "06_hybrid.png")
