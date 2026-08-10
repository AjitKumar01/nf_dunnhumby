# Audit: does the implementation match the theory?

Written while `run3` trains, against `paper/version_3.html`. Divergences only — the parts
that agree are in the verification tables of the paper and in `scripts/v3/validate*.py`.

Severity: **A** invalidates a reported number · **B** changes what the model is ·
**C** stated in the spec as future work and correctly absent.

---

## A1 — Seasonality is degenerate for 54.6% of trips

`fit.Batcher` builds the seasonal index as `week.clamp(0, 52)`. `WEEK_NO` runs 1–102, so
**every week from 52 onward maps to the single index 52** — 54.6% of all trips share one
seasonal parameter, and the second year of the panel has no seasonality at all.

The spec (§2, symbol table) defines `w = (WEEK_NO − 1) mod 52`. Version 2's pipeline does
this correctly; the v3 batcher does not.

Consequence: `μ_j'δ_w` is fitted on a corrupted covariate. Not fatal to the likelihood —
it is a mis-specified feature, not an inconsistency between energy and normaliser — but any
seasonal claim from this fit is void, and it wastes 52×K_t parameters.

**Fix:** `(week - 1) % 52`. One line.

## A2 — The likelihood omits the units factor entirely

The spec's likelihood (§16, Eq. 27) is

    log Σ_g π_g exp(E(S)) / (Z(g) − 1)   +   Σ_{j∈S} log P(q_j | j ∈ S)

`RaggedModel.loglik` implements only the first term. There is no units head anywhere in
`ragged.py` or `fit.py` — no NB parameters, no `q` in the data path beyond `line_units`
sitting unused in the index.

So the fitted object is **P(set)**, not **P(basket)**. That is a legitimate model, and
arguably the right first target, but it is *not* the likelihood the paper specifies, and a
held-out number from it is not comparable to one that includes units. §8 measures units as
worth a TVD improvement of 0.045 → 0.016, so the omission is not negligible.

**Fix:** either implement Eq. 19 or restate the paper's likelihood for this fit as the set
likelihood. The second is honest and cheaper; the first is what the spec claims.

## A3 — The trip-type mixture is absent, and §18's cost model assumed it

Eq. 7 carries `a_g` and `ϑ_g'α_j`; Eq. 27 is a mixture over `g = 1..G`. Neither exists in
`RaggedModel` — no `a_g`, no `ϑ_g`, no mixture weights, no sum over `g`. The fit is `G = 1`.

This matters twice over. It removes the device §14.3 examines for carrying basket-size
dispersion. And §18's cost table is built at `G = 3` — "the trip type multiplies almost
everything" — so the measured 3.4 s/iteration is for a model roughly 2.4× cheaper than the
one costed. The projection was not conservative; it was for a different model.

## B1 — `ρ_0` is applied to a clamped basket size

`energy()` uses `n = bincount(...).clamp(max=nmax)`, so a 117-line basket is charged
`ρ_0(60)`. Meanwhile `log_Z` sums only over subsets with `n ≤ 60`. The observed basket is
therefore scored while lying outside the support summed over — the 5.03% mismatch already
recorded (7,881 baskets from `R = 4`, 280 from `n_max`, max observed category count 23).

Clamping is the wrong repair either way: it silently relabels an out-of-support basket as
an in-support one. The two defensible options are to drop those trips, or to raise `R` and
`n_max` until the support covers the data — `R = 23`, `n_max = 117` covers all of it, at a
cost §18's formula prices as roughly `R` times the ESP term.

Measured earlier: this is *not* what made the likelihood positive (the violating baskets
were in-support), but it is still an inconsistency between energy and normaliser of exactly
the kind that produced the last bug.

## B2 — No penalty of any kind, on 449,257 parameters

`fit.py` has no weight decay, no L2, no prior. The spec's §18 lists penalty blocks with
coefficients. 449k parameters against 157k training trips, with 2,066 household vectors
each seen ~76 times, is the Neyman–Scott regime §19 flags — and nothing is shrinking them.

## B3 — λ_max(Λ) is never measured on real data

§14 makes `λ_max(Λ) < 1` the condition under which the mode is unique and the Laplace
proposal is sound, and §29 names "penalise λ_max" as the mitigation. The implementation
instead caps `‖φ_j‖`, which is a *proxy*: it bounds λ_max from above but the bound is loose
and depends on the purchase probabilities. λ_max is not computed anywhere in `scripts/v3/`.

So the central diagnostic of Part II is not being logged during the fit that Part II exists
to make possible. ESS is logged and is a downstream symptom; λ_max is the quantity.

## C — Correctly absent, and the spec says so

Recency, coupon eligibility, the days-of-supply proxy (Eq. 4), trip timing (§9), the state
recursions and Axiom O (§10), the causal tiers (§22), the environment (Part V).
`features.py` states the first three at the top of the file.

---

## What this means for `run3`

Even if it produces a negative held-out likelihood, it is the **set** likelihood of a
**G = 1** model with a **corrupted seasonal covariate**, **no regularisation**, and a
**5.03% support inconsistency**. That is worth having as the first valid fit, and it is not
the model the paper specifies.

Order to fix, cheapest first: A1 (one line) · B3 (log λ_max, ~10 lines) · B1 (raise R and
n_max, or drop out-of-support trips) · B2 (weight decay) · A2 and A3 (real work).

The audit itself is the lesson from the last bug: every defect above is a place where the
code and the paper disagree, and none of them would be caught by the existing tests, which
check the kernel against enumeration and never check the kernel against the specification.
