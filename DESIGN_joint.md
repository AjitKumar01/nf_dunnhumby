# A model that can score a whole basket

Branch `joint-likelihood`. Motivated by a fair criticism of the existing work: the
pseudo-likelihood, the biased generation start, and the split between `b` at category entry
and `u` at product choice all look like patches. They are — but they are patches around one
thing, and it is worth naming precisely.

## The single cause

`P(S | 𝒦) ∝ exp E(S)` has a normaliser summing over `𝒮(𝒦)`, which was measured at median
10^6.7 and maximum 10^124 baskets. Everything else follows from deciding that sum cannot be
computed:

| patch | exists because |
|---|---|
| Besag pseudo-likelihood instead of likelihood | can't normalise, so score conditionals instead |
| never scoring the joint | same |
| Gibbs from an interaction-free first draft | can't sample `P(S｜𝒦)` directly |
| 4 sweeps, no mixing diagnostic | same |
| entry decided on `b`, contents on `u` | can't use the basket before it exists |

So the question is not "are these patches good" but **"is the normaliser actually
intractable?"**

## It is not

`n = Σ k_c` is fixed *given* 𝒦, so `λ := 1/(n−1)` is a constant, not a function of S. Then

```
E(S) = Σⱼ bⱼ + (λ/2)( ‖Σⱼ φⱼ‖² − Σⱼ‖φⱼ‖² )
     = Σⱼ cⱼ + ½‖√λ Σⱼ φⱼ‖²,        cⱼ := bⱼ − (λ/2)‖φⱼ‖²
```

Apply the Gaussian identity `exp(½‖v‖²) = E_{z∼N(0,I)}[exp(zᵀv)]`:

```
exp E(S) = E_z [ ∏_{j∈S} exp( cⱼ + √λ zᵀφⱼ ) ]
```

Given `z` the product factorises over items, so summing over `𝒮(𝒦)` — "choose exactly
`k_c` from each category" — gives a product of **elementary symmetric polynomials**:

```
Z = Σ_{S∈𝒮(𝒦)} exp E(S) = E_{z∼N(0,I)} [ ∏_c e_{k_c}( { exp(cⱼ + √λ zᵀφⱼ) }_{j∈𝒞ₛ(c)} ) ]
```

`e_k` is computable exactly in `O(N_c · k_c)` by the standard recursion. **A sum over up to
10^124 baskets becomes a K-dimensional Gaussian expectation of cheap polynomials.**

Nothing about the model changes. This is the same energy, the same `P(S｜𝒦)`. Only the
inference changes.

## Verified, not asserted

**The identity.** Against brute-force enumeration on four small cases (K = 2, 3, 4, 8;
|𝒮(𝒦)| = 24 to 300), the Monte-Carlo estimate matches. A convergence study on one case
confirms it is *unbiased* rather than merely close: |error|/SE stays near 1 and
|error|·√n stays flat across n = 10³ to 10⁶.

**Naive Monte Carlo is nevertheless useless.** The estimator variance is governed by
`λ‖Σⱼ αⱼ‖²`, measured on the fitted model at median **8.0**, p90 **16.1**, p99 **31.0**.
Since the relative variance of `exp(zᵀv)` is `exp(‖v‖²) − 1`, that implies:

| | exponent | draws for 1% standard error |
|---|---|---|
| median basket | 8.0 | 31 million |
| p90 | 16.1 | 96 billion |
| p99 | 31.0 | 2.8 × 10¹⁷ |

This is the same pathology as the inclusive-value estimator, one level worse. Sampling from
`N(0, I)` is not viable and no amount of compute fixes it.

**Importance sampling from a Laplace proposal is viable.** Find the mode `ẑ` of
`log[N(z;0,I)·integrand(z)]`, build a Gaussian proposal there, and importance-sample.
Validated against exact enumeration at a realistic exponent (9.03, against a real median of
8.0):

| method | draws | error in log Z |
|---|---|---|
| naive MC | 20,000 | −0.104 |
| Laplace alone | 0 | +0.142 |
| importance sampling | 100 | **+0.004** |
| importance sampling | 10,000 | **−0.000** |

with effective sample size 0.95 of nominal.

**And it holds at the real dimension.** On real fitted α and real held-out basket
compositions at K = 64, using only a *diagonal* Laplace proposal: ESS/n **0.73–0.89**,
log Z standard error **0.023–0.038** from 256 draws, mode-finding 0.06–0.13 s per basket
with an off-the-shelf optimiser.

## What this buys

| | now | with a computable Z |
|---|---|---|
| training objective | pseudo-likelihood, no consistency guarantee | exact log-likelihood |
| "is this basket plausible?" | unanswerable | `log P(S｜𝒦)` directly |
| generation | Gibbs from a biased draft, T = 4, unvalidated | exact: draw `z`, then sample each category independently |
| entry vs contents | `b` for entry, `u` for contents | still `b`, and now provably so — see below |

The generation change is the sharpest. Given `z`, categories are **conditionally
independent** and within a category the law is "choose `k_c` items with weights
`exp(cⱼ + √λ zᵀφⱼ)`", which has an exact `O(N_c·k_c)` sampler by the same recursion that
computes `e_k`. So: sample `z` from its posterior, then sample each category exactly. **No
Gibbs, no burn-in, no mixing question, no interaction-free first draft.**

## What it does not fix

Honesty about scope, since the point of this exercise is to stop patching:

1. **The `b`-versus-`u` split at category entry survives**, and now for a stated reason
   rather than an evasion: entry is a decision about a category before any basket exists, so
   no basket-conditional quantity is available. What changes is that the *contents* factor
   is no longer approximated, so the inconsistency is isolated to one place instead of
   pervading the fit.
2. **The four other factors are untouched.** Incidence, breadth and units keep their
   assumptions.
3. **Cost.** At 0.1 s per basket for mode-finding, a naive implementation is roughly 32
   hours per fit. This needs batched Newton steps in torch with `ẑ` warm-started across
   iterations, which is the main engineering risk and is not yet demonstrated.

## Status

Feasibility established. Implementation not started.
