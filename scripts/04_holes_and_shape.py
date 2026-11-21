"""Illustration 4: a hole in the support. Convex-hull and depth diagnostics
miss it; local (union-of-balls / nearest-neighbour / conformal) diagnostics
and persistent homology find it.

Controls: uniform on the annulus 1 <= |x| <= 2 in R^2 (n0 = 400).
Treated:  half uniform in the disc |x| <= 0.7 (inside the hole, NO control
          support), half uniform on the annulus (n1 = 100).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from ripser import ripser
from common import (style, save, write_text, in_hull, random_tukey_depth,
                    nn_dist, BLUE, ORANGE, AQUA, INK2)

style()
rng = np.random.default_rng(4)


def annulus(n, r0, r1, rng):
    r = np.sqrt(rng.uniform(r0 ** 2, r1 ** 2, n))
    a = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([r * np.cos(a), r * np.sin(a)])


X0 = annulus(400, 1, 2, rng)
X1 = np.vstack([annulus(50, 0, 0.7, rng), annulus(50, 1, 2, rng)])
in_hole = np.r_[np.ones(50, bool), np.zeros(50, bool)]

hull = in_hull(X1, X0)
depth = random_tukey_depth(X1, X0, rng=rng)
d = nn_dist(X1, X0)

# split-conformal outlier p-values (Bates et al. 2023 style), score = distance
# to 5th nearest control in a training split; calibrate on held-out controls
perm = rng.permutation(len(X0)); tr, ca = X0[perm[:250]], X0[perm[250:]]
k = 5
score = lambda Z: np.sort(cdist(Z, tr), 1)[:, k - 1]
s_cal, s_test = score(ca), score(X1)
pval = (1 + (s_cal[None, :] >= s_test[:, None]).sum(1)) / (len(ca) + 1)
# Benjamini-Hochberg at q = 0.1
q, m_ = 0.1, len(pval)
o = np.argsort(pval); thresh = q * np.arange(1, m_ + 1) / m_
passed = np.where(pval[o] <= thresh)[0]
flag = np.zeros(m_, bool)
if len(passed):
    flag[o[: passed.max() + 1]] = True

# persistent homology of the control sample (Vietoris-Rips, H0 and H1)
dgms = ripser(X0, maxdim=1)["dgms"]
h1 = dgms[1]; pers = h1[:, 1] - h1[:, 0]
top = np.sort(pers)[::-1][:3]

txt = f"""# Illustration 4: hole in the support (n0=400 controls on annulus, 100 treated)
                                    in hole   on annulus
share inside control convex hull    {hull[in_hole].mean():.2f}      {hull[~in_hole].mean():.2f}
mean random Tukey depth             {depth[in_hole].mean():.3f}     {depth[~in_hole].mean():.3f}
mean NN distance to controls        {d[in_hole].mean():.3f}     {d[~in_hole].mean():.3f}
share flagged by conformal BH q=0.1 {flag[in_hole].mean():.2f}      {flag[~in_hole].mean():.2f}
Persistent homology of controls: top-3 H1 persistences = {np.round(top, 3).tolist()}
(one long-lived loop = the hole; the rest are sampling noise)
"""
write_text("04_holes.txt", txt); print(txt)

fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.5))
hv = ConvexHull(X0).vertices
ax[0].fill(X0[hv, 0], X0[hv, 1], color=INK2, alpha=0.07, lw=0)
ax[0].plot(np.r_[X0[hv, 0], X0[hv[0], 0]], np.r_[X0[hv, 1], X0[hv[0], 1]],
           color=INK2, lw=0.8, label="control convex hull")
ax[0].scatter(*X0.T, s=6, color=BLUE, alpha=0.6, lw=0, label="controls")
ax[0].scatter(*X1[flag].T, s=22, color=ORANGE, lw=0,
              label="treated, flagged (conformal BH)")
ax[0].scatter(*X1[~flag].T, s=22, facecolor="none", edgecolor=ORANGE,
              lw=1, label="treated, not flagged")
ax[0].set_aspect("equal"); ax[0].set_title("(a) the convex hull does not see the hole")
ax[0].legend(fontsize=7, loc="lower left", bbox_to_anchor=(0, -0.42), ncol=2)
lim = max(dgms[1][:, 1].max(), 0.8)
ax[1].plot([0, lim], [0, lim], color=INK2, lw=0.8)
ax[1].scatter(*dgms[0][:-1].T, s=10, color=BLUE, label="H0 (components)")
ax[1].scatter(*h1.T, s=14, color=ORANGE, label="H1 (loops / holes)")
ax[1].set_xlabel("birth scale (Rips edge length)"); ax[1].set_ylabel("death scale")
ax[1].set_title("(b) persistence diagram of controls"); ax[1].legend(fontsize=7.5)
r = np.linspace(0, 1.2, 200)
ax[2].plot(r, [(d[in_hole] > x).mean() for x in r], color=ORANGE,
           label="treated in hole")
ax[2].plot(r, [(d[~in_hole] > x).mean() for x in r], color=AQUA,
           label="treated on annulus")
ax[2].set_xlabel("radius r"); ax[2].set_ylabel("share farther than r")
ax[2].set_title("(c) coverage curve C(r) = P(dist to controls > r)")
ax[2].legend(fontsize=7.5)
fig.tight_layout(); save(fig, "04_holes.png")
