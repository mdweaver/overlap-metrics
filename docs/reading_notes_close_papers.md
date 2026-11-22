# Reading notes: the three closest papers

*Read in full (main text plus appendices and supplements) on 22 November 2026.*

| Paper | Version read | Length | What was read |
|---|---|---|---|
| Armstrong & Kolesár | arXiv:1712.04594v5, 18 January 2021 (the *Econometrica* 2021 version) | 56 pp. | Main text; Appendices A–B; Supplemental Appendices C–D |
| Kallus | *JMLR* 21(62), 2020, published PDF | 54 pp. | Main text; Appendices A–D |
| Ma, Huang & Namkoong | arXiv:2511.22003v2, 18 April 2026 | 39 pp. | Main text; Appendices A–D |

For Ma, Huang & Namkoong, note that v2 adds Yian Huang as an author; v1 listed Ma and Namkoong only.

Page or section references below are to those versions. "Plain language" lines are mine.

---

## 1. Armstrong & Kolesár (2021), "Finite-sample optimal estimation and inference on average treatment effects under unconfoundedness"

**Setting.**
- **Conditioning:** inference conditions on the realised covariates and treatments (fixed design).
- **Estimand:** the conditional ATE, \(Lf=\frac1n\sum_i[f(x_i,1)-f(x_i,0)]\), or the CATT. Appendix A.1 allows any known weights \(w_i\) on units. A PATE/PATT version is in §3.2.
- **Outcome model:** \(Y_i=f(x_i,d_i)+u_i\), with normal errors and known variance for the finite-sample results. Both assumptions are relaxed for feasible, asymptotically valid inference (§3–4).

**What is assumed about the outcome surface.**
- \(f\) lies in a researcher-specified class \(\mathcal F\) that is **convex and centrosymmetric**.
- The leading case is Lipschitz, \(|f(x,d)-f(\tilde x,d)|\le C\lVert x-\tilde x\rVert_X\), with the same \(C\) for both arms.
- *Not* linear in observables: this is a nonparametric class.
- The authors point out (§2.4, p. 11) that a pure Lipschitz class *rules out the linear model* unless the coefficients are bounded.
- Appendix A.2 therefore offers \(\mathcal F_{z,\mathrm{Lip}}(C)\): an **unrestricted linear part** in chosen covariates \(z\), plus a Lipschitz residual orthogonal to \(z\) (eq. 28). With \(z=x\) and \(C=0\) this is exactly the linear model, so \(C\) measures "departure from linearity".
- The optimal estimator in that class becomes **regression-adjusted matching** (Theorem A.5).
- Monotonicity restrictions are discussed and allowed when centrosymmetry is dropped.

**What is linear.**
- The *estimator* is linear in outcomes, \(\hat L_k=\sum_i k(x_i,d_i)Y_i\), with weights depending on the whole design \(\{x_i,d_i\}\), possibly nonlinearly.
- The class covers series, kernel and matching estimators (§2.2).
- Restricting to linear (affine) estimators loses little:
  - minimax RMSE among all estimators is at least 89.4% of the best linear estimator's (Donoho 1994; Theorem A.2), so non-linear estimators can cut worst-case RMSE by at most 10.6%;
  - the fixed-length CI is at least 71.7% efficient relative to CIs that adapt to a smooth function (Theorem A.3), and typically above 97% in their application.

**Core results.**
1. **Worst-case bias of any given linear estimator** (Theorem 2.1). Under Lipschitz, the worst-case bias is a finite linear program over the \(2n\) values \(f(x_i,d)\), subject to pairwise Lipschitz constraints (eq. 14). This uses Beliakov's (2006) extension result (Lemma A.1). Only treated–control pairs involving units used as matches need constraining. A CI is \(\hat L\pm\mathrm{cv}_\alpha(\overline{\rm bias}/\mathrm{sd})\,\mathrm{sd}\), where \(\mathrm{cv}_\alpha(b)\) is the \(1-\alpha\) quantile of \(|N(b,1)|\) (eq. 10).
2. **Optimal weights** (Theorem 2.2) solve a convex program indexed by \(\delta\). Varying \(\delta\) traces the **optimal bias–variance frontier** (§2.3, following Donoho 1994 and Low 1995). Choose \(\delta\) for RMSE or CI length. A piecewise-linear path algorithm (Appendix A.3) makes the cost depend on \(n\), not on \(p\).
3. **Matching structure.** The optimal estimator imputes the counterfactual by a weighted average of opposite-arm units chosen through an "effective distance" that penalises reuse (pp. 10–11). For large \(C\) it is **1-NN matching** (Theorem 2.3). As \(C\to 0\) it becomes the raw difference in means.
4. **The metric must match the class.** The matching metric should be the norm defining the Lipschitz class. Supplement C shows 74–80% efficiency for the "wrong" (inverse-SD Euclidean) metric.
5. **No data-driven choice of \(C\).** Adaptation bounds (§2.5, Theorem A.3) imply that \(C\) and the norm cannot be chosen from the data to tighten CIs without undercoverage somewhere in \(\mathcal F\). The authors recommend **reporting results for a range of \(C\)** as a sensitivity analysis (footnote 3; Remark 3.1). Rescaling makes the whole path in \(C\) free to compute.
6. **Impossibility of \(\sqrt n\)-inference** (Theorem 4.1). With \(\gamma\)-Hölder smoothness, excess CI length is at least of order \(n^{-\gamma/p}\). For Lipschitz (\(\gamma=1\)), \(\sqrt n\)-inference is possible only if \(p\le 2\). This holds *even with perfect overlap*: interpolation bias between nearby points dominates when \(p>2\).
7. **Validity without overlap** (Theorem 4.2, Lemma 4.1). The feasible CIs are asymptotically valid under bounded support, *without any overlap condition*, including set identification under complete non-overlap. The key is that optimal weights satisfy a Lindeberg condition (eq. 22): no unit gets too much weight.
8. **Matching estimators** with fixed \(M\) can violate Lindeberg under poor overlap (§4.3), so their normal approximation fails. They suggest computing \(\mathrm{Lind}(k)\) as a diagnostic, with a cutoff of 0.075 following Noack & Rothe.

**Application.** NSW treated men (n = 185) against PSID comparisons (n = 2,490), CATT.
- *Norm:* a weighted \(\ell_1\) norm with **interpretable per-covariate bounds** on partial effects (Table 1). For example, race and marriage gaps are at most $2,500, about 25% at $10,000 earnings; lagged earnings at most one-for-one.
- *Results:*
  - At \(C=1\): estimate 0.96, worst-case bias 1.63, robust SE 1.01, CI (−2.33, 4.25).
  - Bias is a large share of CI width.
  - Results are plotted across \(C\in[0,2]\) (Figure 1).
  - The difference from the experimental benchmark (1.79) is well within the worst-case bias.

**What it does not do.**
- It does not split the sample into overlap and non-overlap regions; one worst-case bound covers everything.
- Treatment is binary only.
- There is no diagnostic of *where* the bias comes from.
- \(C\) is not calibrated from data, deliberately.

**Software.** R package `ATEHonest` (github.com/kolesarm/ATEHonest).

*Plain language:* fix in advance how fast the expected outcome can change with each covariate. The method then finds the weighted comparison that is most precise given the worst bias that assumption allows, and widens the CI by exactly that bias. It works with or without overlap. Poor overlap just shows up as a large bias term.

---

## 2. Kallus (2020), "Generalized optimal matching methods for causal inference"

**Setting.**
- **Estimand:** the sample ATT (SATT). Extensions to the ATE, ATE on a target population and continuous treatments are in companion papers (§6).
- **Estimator:** weighting estimators \(\hat\tau_W=\bar Y_{T_1}-\sum_{i\in T_0}W_iY_i\), with "honest" weights that depend only on \((X_{1:n},T_{1:n})\) (eq. 1), plus an augmented version that uses an outcome regression fitted on separate folds (eq. 2).
- **Risk:** conditional MSE given the design.

**Key identity** (Proposition 1). Conditional bias is \(B(W;f_0)=\frac1{n_1}\sum_{T_1}f_0(X_i)-\sum_{T_0}W_if_0(X_i)\), and \(\mathrm{CMSE}=B^2+V^2\) with \(V^2=\sum W_i^2\sigma_i^2+\dots\). For the augmented estimator, the bias is \(B(W;f_0-\hat f_0)\) (Corollary 2).

**Matching as minimax** (Theorem 3).
- With \(\sigma=0\) and \(f_0\) \(\gamma\)-Lipschitz in a pseudo-metric \(\delta\):
  - nearest-neighbour matching \(=\arg\min_{W\in\text{simplex}}\sup_{\lVert f\rVert_{\rm Lip}\le\gamma}B^2\);
  - 1:1 matching \(=\) the same, with weights capped at \(1/n_1\).
- Kallus notes the link to Kantorovich–Rubinstein: **optimal matching minimises the Wasserstein distance** between treated and reweighted controls (p. 7).

**Generalized optimal matching (GOM)** (Definitions 4–5).
- For a seminorm \(\lVert\cdot\rVert\) on outcome functions, imbalance is \(\mathcal B(W;\lVert\cdot\rVert)=\sup_{\lVert f\rVert\le1}B(W;f)\). This is the dual norm of the bias operator, an integral probability metric (IPM).
- GOM chooses \(W\in\mathcal W\) to minimise \(\mathcal B^2+\lambda\lVert W\rVert_2^2\).
- Choices of weight set \(\mathcal W\):
  - general (signed);
  - simplex, which *prohibits extrapolation*;
  - capped simplex;
  - subsets and multisubsets.
- \(\lambda=\sigma^2/\gamma^2\) and varying \(\lambda\) traces the **balance–variance Pareto frontier**. Figure 1b shows it for nearest-neighbour matching ("BVE-NNM"), with 1:1M *off* the frontier.
- CMSE is bounded by \(\gamma^2(\mathcal E^2_{\min}+\lambda/n_1)\) (Corollary 6).

**Existing methods as GOM** (Table 1, §5).

| Method | Equivalent GOM |
|---|---|
| NNM, 1:1M | Lipschitz norm |
| Optimal-caliper matching | an averaged Lipschitz norm |
| CEM | \(L^\infty\) on piecewise-constant functions |
| Mean matching, near-fine balance | linear functions (\(\lVert\cdot\rVert_{2\text{-lin}}\) or \(\ell_p\)-lin), subset weights |
| Stable balancing weights (Zubizarreta 2015) | \(\ell_1\)-lin norm, simplex, \(\lambda>0\) |
| OLS regression adjustment with treated-centred interactions | linear norm with *general (signed)* weights (Proposition 26); ridge for \(\lambda>0\) (Proposition 27) |

Regression and mean-matching coincide only if exact balance is feasible *without extrapolation*. Simplex weights forbid extrapolation; general weights allow it.

**Kernel optimal matching (KOM)** (§4).
- GOM with an RKHS norm; the imbalance is the kernel MMD.
- It is a convex quadratic program (Proposition 15).
- **Bayesian reading** (Proposition 16): if \(f_0\sim GP(c,\gamma^2K)\) and noise has variance \(\sigma^2\), then the posterior CMSE equals \(\gamma^2(\mathcal E^2+\lambda/n_1)\) exactly. **KOM minimises expected (average-case) squared error**, not just a bound.
- **KOM++** picks \(\gamma^2,\sigma^2\), the kernel and per-covariate length-scales (ARD) by **marginal likelihood on control outcomes only** (§4.1, §4.5). Kallus notes this is not strictly "honest", but only a few hyperparameters are extracted.
- **Consistency** (Theorem 18): \(\sqrt n\) if \(f_0\) has finite RKHS norm (correct specification), and consistency regardless of specification with a \(C_0\)-universal kernel (Gaussian, Matérn).
- **Double robustness** (Theorem 19) for the augmented estimator. KOM followed by weighted least squares is consistent regardless of \(f_0\) and \(\sqrt n\) if \(f_0\) is linear (Theorem 20).
- **Semi-kernel optimal matching (SKOM)** (§4.4) leaves a parametric (e.g. linear) part *unpenalised*. With thin-plate or cubic spline kernels it balances all functions with square-integrable Hessian and matches means exactly. It is infeasible on the simplex if the treated mean lies outside the control convex hull; Kallus then advises plain KOM rather than allowing extrapolation.
- **Argument against Lipschitz:** Lipschitz balls are non-separable, which is "too little structure". A \(C_0\)-universal RKHS is separable yet still universal, which is why KOM beats NNM in his experiments (pp. 23–24).

**Partial identification under limited overlap** (Appendix B).
- Bias is bounded by \(\lVert[f_0]\rVert\cdot\mathcal B(W;K)\).
- With an a priori bound \(\hat\gamma\) on the RKHS norm (or roughness), *or* the KOM++ marginal-likelihood estimate \(\hat\gamma\), report the **interval estimate** \([\hat\tau_W\pm\hat\gamma\,\mathcal B(W;K)]\) (eq. 14), widened by standard errors.
- The stated justification: "we assume we can still judge the complexity of \(f_0\) on the treated population of X … by observing its values with noise on the control population".
- Simplex weights matter: signed weights can drive any imbalance to zero, but only by extrapolating, which makes such a "perfect match" deceptive (p. 39).
- Example 6 (no overlap) shows the bias-bound intervals restore coverage that standard intervals lose (Figure 7).

**Empirical example.** IHDP (Infant Health and Development Program) data with overlap deliberately degraded (§4.6). KOM++ with ARD has much lower MSE than IPW, 1:1M, CEM and others, and the KOM++ interval estimates cover.

**What it does not do.**
- It does not split by support; the bias bound is global.
- There is no sensitivity curve over \(\gamma\); it uses an a priori or estimated value.
- It gives no formal coverage theory for the data-estimated \(\hat\gamma\) (compare Armstrong–Kolesár's impossibility result).
- Treatment is binary in this paper.

*Plain language:* every matching or weighting method is secretly answering "which outcome functions am I protecting against?". Pick that class explicitly (Lipschitz, linear, smooth kernel). Then the method's imbalance is literally its worst-case bias, and with a Gaussian-process prior it is its expected squared bias. Keeping weights non-negative is what stops a method from extrapolating.

---

## 3. Ma, Huang & Namkoong (2026, arXiv v2), "A sensitivity approach to causal inference under limited overlap"

**Setting.**
- **Population:** finite population, fixed \(n\).
- **Estimand:** the sample ATE (CATE in Armstrong–Kolesár's sense), or a weighted version \(\tau_w\) (eq. 2.2).
- **Treatment:** binary.
- **Propensity score:** assumed *known* for most of the paper; the authors say the framework needs little of it.
- **Errors:** Gaussian with known variance for the finite-sample results. In practice \(\sigma\) is estimated by Abadie–Imbens nearest-neighbour residuals with \(J=2\) (eq. 4.2).

**Split.**
- Overlap is measured by \(q(x)=\min\{\pi(x),1-\pi(x)\}\) (eq. 2.13). Units with \(q<\epsilon\) form the non-overlap region.
- \(\tau=\tau_++\tau_-\), the overlap and non-overlap contributions on the empirical distribution (eq. 2.14).
- \(\epsilon\) is chosen by the analyst's usual trimming rule. In their experiments it is the value in a small grid that **minimises the length of the trimmed AIPW CI** (eq. 1.2).

**Method ("MP").**
- **Overlap region:** a standard asymptotic estimator, AIPW with a random-forest outcome model, gives inference for \(\tau_+\). Interpolation bias there is treated as negligible: they "assume that standard inferential tools are appropriate for estimating \(\tau_+\)" (p. 3).
- **Non-overlap region:** Armstrong–Kolesár's Donoho-style minimax affine estimator and fixed-length CI for \(\tau_-\) alone (eqs. 2.4–2.10, 2.15), over \(\mathcal F_L\), the Lipschitz class in \(\ell_2\) imposed on the observed points (eq. 2.12).
- **Combined CI** (eq. 3.3): a union bound, adding the endpoints of the \(\alpha/2\) intervals for \(\tau_+\) and \(\tau_-\).
- **Diagnostic** (eq. 2.16): \(T(\mathrm{MP}_\epsilon)=\max(|\mathrm{MP}^-|,|\mathrm{MP}^+|)\), the largest bias trimming could induce.

**Sensitivity and calibration of \(L\)** (§3).
- Vary \(L\) and watch the \(\tau_-\) interval.
- **Contextualise \(L\)** using the overlap region. For the fitted outcome regressor \(\hat\mu_Z\) on overlap units, compute pairwise slopes \(|\hat\mu_Z(x_i)-\hat\mu_Z(x_j)|/\lVert x_i-x_j\rVert\). Take for each \(i\) the \(p\)-th quantile over \(j\), then the max over \(i\) and over arms: \(\tilde L_p\) (eqs. 3.1–3.2).
- Results are reported against the **percentile \(p\)** (80–95%) rather than raw \(L\) (Figures 7, 8, 12).
- In their simulation, 80th–95th percentiles correspond to \(L\approx5\)–14. In the PennUI data they correspond to \(L\approx0.25\)–0.31.

**Analytic results.**
- Lemma 1: the minimax estimator is a matching estimator whose weights come from dual variables on binding Lipschitz constraints.
- Lemma 2 (1-D): extrapolation proceeds from the boundary of the overlap region inward.
- Lemma 5 gives a closed-form bias–variance trade-off in \(\delta\) for a stylised design.
- Lemma 6: a **confidence sequence** for sequential data collection.

**Empirical work.**
- *1-D simulation:* trimmed AIPW undercovers sharply as non-overlap grows. Full Armstrong–Kolesár minimax ("M") is valid but very wide. MP and the combined interval are valid and much shorter (Figures 8–10).
- *PennUI semi-synthetic study:* Pennsylvania Reemployment Bonus experiment, \(p=60\), \(n=539\), with observational data built by subsampling under a constructed propensity score (Lemma 7). Coverage of MP is 100% for \(\tau_-\) at \(L=0.32\). The combined interval covers \(\tau\) and is narrower than M.
- *Data collection* (§5): the expected MP width is computed *without outcomes* before sampling. It ranks two sampling plans correctly where propensity scores cannot tell them apart.

**What it does not do.**
- The cutoff is a single propensity threshold chosen to minimise AIPW CI length. There are **no curves over the cutoff**, and the split depends on the propensity model.
- Interpolation bias within the overlap region is not bounded.
- There is no formal result on what the data-calibrated \(L\) guarantees. Armstrong–Kolesár's adaptation bound says it cannot guarantee coverage, and the paper does not address this.
- The metric is plain \(\ell_2\) on raw covariates, even at \(p=60\). Outcome-informed metrics are not considered.
- Treatment is binary only; multi-armed treatments are named as future work. Continuous treatments and policies are not covered.
- No software is mentioned in the paper.

*Plain language:* trim as usual and analyse the well-overlapped units normally. For the trimmed units, report the worst-case range of their contribution under "the outcome can't change faster than \(L\)". Read \(L\) as "the \(p\)-th percentile of slopes we actually see where the data are good", and show how conclusions move as \(p\) rises.

---

## 4. Side-by-side comparison

| | Armstrong–Kolesár 2021 | Kallus 2020 | Ma–Huang–Namkoong 2026 |
|---|---|---|---|
| Estimand | CATE/CATT (any known unit weights); PATE by extension | SATT (extensions elsewhere) | Sample ATE; weighted |
| Treatment | Binary | Binary | Binary |
| Estimator class | Linear (affine) in outcomes, any design-dependent weights, signed allowed | Linear weighting in outcomes; weight set chosen (simplex forbids extrapolation); augmented version | AIPW in overlap region; minimax affine (A–K) in non-overlap region |
| Outcome assumption | Convex symmetric class; Lipschitz (optionally after an unrestricted linear part); monotone variants | Seminorm ball (Lipschitz, linear, \(L^\infty\), RKHS, semi-RKHS); GP-prior reading | Lipschitz (\(\ell_2\)) in the non-overlap region only; consistency of AIPW in the overlap region |
| Linear in observables? | No (nonparametric); linear model is the special case \(\mathcal F_{x,\rm Lip}(0)\) | No; linear only for linear-kernel or mean-matching norms; SKOM leaves the linear part free | No |
| Worst case or average case | Worst case (minimax) | Worst case bound; exact posterior risk under GP (KOM) | Worst case in the non-overlap region; asymptotic in the overlap region |
| Overlap handling | None needed; bias term absorbs it | Simplex weights avoid extrapolation; bias-bound interval | Explicit split by propensity threshold |
| Smoothness constant | Fixed a priori; range reported; data choice impossible | A priori or marginal-likelihood estimate (empirical Bayes) | Sensitivity curve; contextualised by percentiles of fitted slopes in the overlap region |
| Use of outcomes in design | None | Control outcomes for hyperparameters (KOM++) | Overlap-region outcomes for the AIPW fit and the \(L\) calibration |
| Interpolation bias | Bounded (dominates when \(p>2\)) | Bounded (inside the bias bound) | Not bounded in the overlap region |
| Software | R `ATEHonest` | Gurobi-based code (not packaged in the paper) | None stated |
