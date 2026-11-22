# What this project would add to the three closest papers

*Companion to [`investigation.md`](investigation.md) §2.6–2.7 and Rank 1, and to [`reading_notes_close_papers.md`](reading_notes_close_papers.md). Written 22 November 2026.*

## The starting point, in plain language

All three papers answer one question: when you estimate a treatment effect by comparing treated units with similar-looking comparison units, how wrong could you be because the comparison units are not *exactly* like the treated ones?

They share one move. They stop pretending the bias is zero. Instead they ask the researcher how fast the expected outcome can change as the covariates change. That is a smoothness or "roughness" assumption, the Lipschitz constant \(L\). Given that number, they compute the worst bias it allows and widen the confidence interval by it.

- **Armstrong & Kolesár (2021)** do this for the whole sample and choose the best weights given the assumption. They say the roughness must be fixed in advance, so results should be reported over a range of values.
- **Kallus (2020)** shows that every matching or weighting method is implicitly protecting against some class of outcome functions. Choosing that class explicitly lets you trade bias against variance optimally. He also gives an "average-case" version (a Gaussian-process prior) and an interval that uses a roughness estimated from the control group's outcomes.
- **Ma, Huang & Namkoong (2026)** split the sample at a propensity-score cutoff:
  - for well-overlapped units they use standard methods (AIPW);
  - only for poorly overlapped units do they use the worst-case bound;
  - they show how the bound grows as the assumed roughness grows, and translate roughness into "the p-th percentile of slopes we see where data are good".

## What additional work would add

Each item says what is missing, what we would do, and why it matters.

### 1. Curves over the support cutoff, not just the roughness

**Missing.** Ma–Huang–Namkoong pick one propensity cutoff, the one that makes the AIPW interval shortest, and treat it as fixed. Their only sensitivity curve is over roughness.

**What we would add.**
- Define "supported" without a propensity model: distance to the nearest comparison units, in multiples of their typical spacing, or a conformal test with false-discovery control.
- Show results across *many* cutoffs.
- In the "exclude" version, also show how the population being described changes, because dropping units changes the estimand.

**Plain language.** Where you draw the line between "comparable" and "not comparable" is a judgement call. Ma–Huang–Namkoong hide it, and their way of drawing it rewards short intervals. We would make it a visible dial. A propensity model can also miss a hole in the data: a region surrounded by comparison units but empty in the middle. Distance-based rules catch it (Illustration 4).

### 2. A "graded" hybrid that also bounds interpolation bias

**Missing.**
- Ma–Huang–Namkoong assume zero bias in the well-overlapped region.
- Armstrong & Kolesár apply one worst-case bound everywhere, which is valid but wide.
- Nobody does "average case where the data are rich, worst case where they are absent".

**What we would add.**
- Inside the support: an average-case allowance for interpolation bias, based on a Gaussian-process or kernel prior (Kallus's Bayesian reading).
- Outside the support: Armstrong–Kolesár worst-case bounds.
- A clear statement of what the combined interval guarantees.

**Plain language.** Even inside the comparable region, treated units never sit exactly on top of comparison units. The small gaps still cause bias ("interpolation"), and with several continuous covariates that bias does not disappear quickly as samples grow. Armstrong–Kolesár prove it dominates the noise whenever more than two covariates are continuous. Our Illustration 8 found that, when most units are supported, interpolation is 55–70% of the worst-case bias. Ignoring it, as the split method does, understates risk. Treating it with full worst-case pessimism, as Armstrong–Kolesár do, overstates it. Average-case inside and worst-case outside is the middle path.

**Risk.** Mixing a Bayesian (average-case) component with a worst-case component needs careful theory to say what "95%" means.

### 3. A theory of what "empirical roughness" can and cannot tell you

**Missing.**
- Armstrong & Kolesár prove you *cannot* pick the roughness from the data and still guarantee coverage.
- Kallus and Ma–Huang–Namkoong calibrate it from data anyway, as a practical guide, but neither says formally what that buys.

**What we would add.**
- (a) A one-sided **refutation test**: the data *can* rule out roughness values that are too small, because the fitted control outcomes already change faster than that.
- (b) A **breakdown value** reported as a multiple of the benchmark: "the conclusion survives unless the outcome changes 2.5× faster where we have no data than anywhere we do".
- (c) Conditions under which the benchmark carries over to the unsupported region, for example under a stationary Gaussian-process model.

**Plain language.** The data can tell you the outcome is *at least* this rough, never that it is *at most* this rough in places you have not observed. So the honest output is a curve with the benchmark marked on it, plus the multiple of the benchmark at which the conclusion flips. Item 3 makes that reading rigorous, much as the benchmarking literature did for omitted-variable sensitivity analysis.

### 4. Distance that works with many covariates

**Missing.** Ma–Huang–Namkoong use plain Euclidean distance on 60 raw covariates. Our Illustration 2 shows Euclidean "nearness" becomes meaningless in high dimension: everyone is far from everyone. Armstrong–Kolesár instead hand-specify a per-covariate bound on partial effects, which is excellent practice but manual. Kallus learns per-covariate relevance from control outcomes.

**What we would add.** A principled default that combines the two:
- interpretable per-covariate bounds that a researcher can defend ("the race gap is at most $2,500");
- or relevance weights learned from control outcomes with sample splitting;
- plus guidance on choosing and reporting them.

**Plain language.** "How rough is the outcome" only makes sense once you say rough *in which direction*. Political scientists can usually say which covariates matter and by how much. The method should let them state that directly.

### 5. Continuous treatments and policy questions

**Missing.** All three papers are binary-treatment only. Ma–Huang–Namkoong list multi-armed treatments as future work.

**What we would add.**
- The same machinery for dose-response questions ("what if everyone got dose \(t\)?") and shift policies ("what if everyone got 10% more?").
- Distance is measured in covariates *and* dose.
- Curves over dose and over shift size show which questions the data can actually answer (Illustration 3).

**Plain language.** With a continuous treatment, overlap usually fails in a hidden way. High doses are only given to certain kinds of people, so "everyone at a high dose" is pure extrapolation, while "everyone a little higher than now" may be well supported. This is the least-developed area and the clearest opening.

### 6. A reporting standard and software for applied researchers

**Missing.** No single tool produces the design-stage map, the support split, both sets of curves, the roughness benchmark and the breakdown value. Only Armstrong–Kolesár have released a package (see below).

**What we would add.** An R package, building on `ATEHonest` for the minimax part, plus a short reporting template for political science and economics papers.

### 7. The design-stage "overlap frontier" (secondary)

This is Kallus's balance–variance frontier, so it is not new as an object. What is new is using it *before* looking at outcomes:
- the "extrapolation floor";
- its comparison with a no-shift baseline;
- splitting the bias bound into interpolation, extrapolation and negative-weight "leverage" for each unit (Illustration 8).

**Plain language.** It is a map of where any estimator would have to stretch, drawn before you run one.

## Honest assessment

- Item 1 and item 6 are useful but incremental.
- **Items 2, 3 and 5 carry the methodological contribution.**
  - Item 2 needs new coverage theory.
  - Item 3 needs a formal account of benchmarking.
  - Item 5 is open ground.
- If those do not pan out, the project is still a strong applied-methods synthesis, but not a new method.

## Existing software

| Paper | Software | Status (as found 22 Nov 2026) |
|---|---|---|
| Armstrong & Kolesár (2021) | **R package `ATEHonest`**: matching and optimal linear estimators with honest (bias-aware) intervals under Lipschitz; includes the NSW example vignette | On GitHub (`remotes::install_github("kolesarm/ATEHonest")`), version 0.1.2 (2020), GPL-3; uses CVXR. Not found on CRAN. Named in the paper (footnote 10). |
| Kallus (2020) | None released with the paper | The paper solves kernel optimal matching as a quadratic program in Gurobi, a commercial solver. A follow-up applied paper reports R code with Gurobi. No official package or repository was found. |
| Ma, Huang & Namkoong (2026) | None stated | The paper mentions no code. Their max-flow step (Appendix B) modifies Armstrong–Kolesár's algorithm. No repository was found in a search. |
| Related: energy balancing (Huling & Mak 2024) | R package `WeightIt` (`method = "energy"`) is believed to implement it | **Not verified this session.** Check CRAN. |
| Related: continuous-treatment independence weights (Huling, Greifer & Chen 2024) | R package `independenceWeights` | On CRAN (verified earlier in this investigation). |
