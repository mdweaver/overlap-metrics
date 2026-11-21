"""Illustration 2: overlap diagnostics as the number of covariates p grows.

Two designs, both with X | D=0 ~ N(0, I_p):
  * "no confounding":    X | D=1 ~ N(0, I_p)            (identical groups)
  * "mild per-covariate": X | D=1 ~ N(0.25 * 1_p, I_p)   (each covariate's
     standardised difference is only 0.25, well inside usual balance rules)

Diagnostics for the ATT target (treated units relative to controls):
  * share of treated points inside the controls' convex hull (King & Zeng)
  * mean random Tukey depth of treated points w.r.t. controls (0 = outside hull)
  * ESS / n0 of the ORACLE ATT weights (theory exp(-p * 0.25^2))
  * share of ORACLE propensity scores outside [0.1, 0.9]  (Crump et al. rule)
  * ratio of mean treated-to-control NN distance to the control-to-control
    NN distance (a scale-free "distance to data" measure)

The point (D'Amour et al. 2021; Balestriero et al. 2021): hull membership
collapses with p even when the two groups are IDENTICAL, so it conflates
"high dimension" with "extrapolation". Density-ratio measures (ESS, propensity
tails) behave correctly in the identical case, but with a fixed per-covariate
imbalance they deteriorate exponentially in p.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from common import (style, save, write_text, ess, in_hull, random_tukey_depth,
                    BLUE, ORANGE, AQUA, INK2)

style()
rng = np.random.default_rng(2)
n0, n1_hull, n1 = 500, 60, 500
ps = [1, 2, 5, 10, 20, 50, 100]
lines = ["design\tp\thull_share\tmean_depth\tess_share\tess_theory\tps_out_0.1_0.9\tnn_ratio"]
res = {}
for design, delta in [("identical", 0.0), ("shift0.25", 0.25)]:
    out = []
    for p in ps:
        X0 = rng.standard_normal((n0, p))
        X1 = rng.standard_normal((n1, p)) + delta
        hull = in_hull(X1[:n1_hull], X0).mean()
        depth = random_tukey_depth(X1[:n1_hull], X0, rng=rng).mean()
        # oracle log-odds of D=1 vs D=0 given x (equal group sizes)
        lo = delta * X0.sum(1) - p * delta ** 2 / 2
        w = np.exp(lo)  # ATT odds weights on controls
        ess_share = ess(w) / n0
        lo1 = delta * X1.sum(1) - p * delta ** 2 / 2
        e_all = 1 / (1 + np.exp(-np.r_[lo, lo1]))
        ps_out = np.mean((e_all < 0.1) | (e_all > 0.9))
        d10 = cdist(X1, X0).min(1).mean()
        D00 = cdist(X0, X0); np.fill_diagonal(D00, np.inf)
        d00 = D00.min(1).mean()
        row = (hull, depth, ess_share, np.exp(-p * delta ** 2), ps_out,
               d10 / d00)
        out.append(row)
        lines.append(f"{design}\t{p}\t" + "\t".join(f"{v:.4f}" for v in row))
        print(design, p, np.round(row, 3))
    res[design] = np.array(out)
write_text("02_high_dimension.tsv", "\n".join(lines) + "\n")

fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.2))
for design, ls, lab in [("identical", "--", "identical groups"),
                        ("shift0.25", "-", "shift 0.25 SD per covariate")]:
    r = res[design]
    ax[0].plot(ps, r[:, 0], ls, marker="o", ms=4, color=BLUE,
               label="in convex hull, " + lab)
    ax[0].plot(ps, r[:, 1] * 2, ls, marker="s", ms=4, color=AQUA,
               label="Tukey depth x2, " + lab)
    ax[1].plot(ps, r[:, 2], ls, marker="o", ms=4, color=ORANGE,
               label="ESS / n0, " + lab)
    ax[1].plot(ps, 1 - r[:, 4], ls, marker="s", ms=4, color=INK2,
               label="share of e(x) in [0.1,0.9], " + lab)
    ax[2].plot(ps, r[:, 5], ls, marker="o", ms=4, color=BLUE,
               label=lab)
for a in ax:
    a.set_xscale("log"); a.set_xlabel("number of covariates p")
ax[0].set_title("(a) geometric: hull and depth")
ax[1].set_title("(b) density ratio: ESS, propensity tails")
ax[2].set_title("(c) NN distance, treated vs. control-control")
ax[1].set_yscale("log"); ax[1].set_ylim(1e-3, 1.5)
ax[0].legend(fontsize=7, loc="upper right")
ax[1].legend(fontsize=7.5, loc="lower left")
ax[2].legend(fontsize=7.5, loc="lower right")
fig.tight_layout(); save(fig, "02_high_dimension.png")
