# Measured, not projected

The feasibility study projected ~1 hour per fit. That was wrong, and this file records what
the implementation actually costs so the estimate is not repeated.

## Correctness (all validated against brute-force enumeration)

| check | result |
|---|---|
| `log_esp` vs enumerating every k-subset, padding present | exact to 4e-16 |
| `inclusion_probs` vs brute force | exact to 6e-16; they sum to k |
| the gradient vs finite differences | 1e-10 |
| full `log Z` vs enumerating every basket in S(K), K = 4…32 | 0.001–0.026 nats |

## Speed, batch of 192 baskets, 6,000 iterations, single-threaded CPU

| category cap | max N | draws | mode (s) | log Z (s) | ESS | hours per fit |
|---|---|---|---|---|---|---|
| 40 (an approximation) | 40 | 64 | 0.42 | 2.59 | 0.657 | **5.0** |
| none (correct) | 177 | 64 | 1.49 | 9.21 | 0.689 | **17.8** |
| none, batch 96 | 177 | 32 | 0.78 | 4.78 | 0.752 | **9.3** |

## Why the projection was wrong

Three reasons, in order of size.

1. **It counted mode-finding only.** The 5.2 ms/basket measured in the feasibility study was
   the *mode*. Each of the importance draws then needs its own full pass over the same
   tensors, and there are 32–64 of them. That alone is most of the gap.
2. **Padding.** Stocked category sizes have median 30 but maximum 182, and a dense
   `[batch, draws, categories, items]` tensor pads every category to the batch maximum. That
   is roughly a 6x waste, and it is why capping at 40 looks 3.5x faster — the cap is not a
   speedup, it is a different and wrong calculation, since 38.6% of categories exceed it.
3. **The feasibility timing used one basket at a time in scipy**, which hid the memory cost
   of doing 192 at once.

## What would fix it

A ragged representation: concatenate all (basket, category) rows into one flat item list
with a row index, and use scatter-reduce instead of a padded dimension. Since 81.2% of rows
have k = 1 and reduce to a plain log-sum-exp, most of the work becomes a single
`scatter_logsumexp` over a flat array with no padding at all. Expected recovery is close to
the full 6x, taking a correct fit to roughly 3 hours.

That is the next thing to build, and it is engineering rather than research: the mathematics
above is verified and does not change.

## After the ragged rewrite (`ragged.py`)

Same mathematics, no padded dimension. Elementary symmetric polynomials come from power
sums via Newton's identities, each a `scatter_add` over a flat item array.

| batch | flat items | rows | draws | mode (s) | log Z (s) | ESS | hours per fit |
|---|---|---|---|---|---|---|---|
| 192 | 49,369 | 1,131 | 64 | 0.12 | 0.89 | 0.701 | **1.68** |
| 192 | 45,598 | 1,129 | 32 | 0.11 | 0.77 | 0.715 | **1.46** |
| 384 | 103,464 | 2,399 | 32 | 0.25 | 1.77 | 0.739 | 3.38 |

**17.8 h → 1.68 h, a factor of 10.6**, and no approximation: category sizes are exact.

Correctness is unchanged — the ragged and padded implementations agree to the last digit on
identical inputs (4.745300 both), and log e_k agrees to 0.

### Two things measured along the way

**Newton's identities are safe here but not unconditionally.** At a top-two log-weight gap
of 27 — a weight ratio of 1e12 — the k = 3 case carries 23% relative error from
cancellation. On the fitted model that gap has median 0.051 and maximum 1.081, so it never
fires. `cancellation_risk()` flags rows where it would.

**Over-dispersing the proposal makes things dramatically worse, which is the opposite of the
textbook advice.** Measured at K = 64: a factor of 1.4 drops ESS from 0.761 to 0.001, and
2.5 puts log Z 27 nats out. In high dimension an inflated Gaussian concentrates on a shell
away from the mode, where the integrand is negligible. Scale 1.0 is correct.

## The synthetic recovery test — the result

Data drawn exactly from P(S|K) by `sampler.py`, so the truth is known and there is no
misspecification. Only the Gram matrix alpha alpha' is identified, so recovery is scored on
that. 1,500 baskets, 96 items in 12 categories, K = 3, 60 households, 1,200 iterations,
three data seeds.

| seed | joint corr | pseudo corr | joint rel err | pseudo rel err |
|---|---|---|---|---|
| 0 | **0.8590** | 0.8551 | **0.5323** | 0.5541 |
| 1 | **0.8366** | 0.8357 | **0.5625** | 0.5872 |
| 2 | 0.8448 | **0.8609** | **0.6209** | 0.6273 |
| mean | 0.8468 | 0.8506 | **0.5719** | 0.5895 |

**Structure (correlation of the off-diagonal Gram entries): tied.** Joint wins two seeds of
three and is marginally behind on average.

**Scale (relative Frobenius error): joint wins all three**, by about 3%.

### What this says about the rebuild

The exact likelihood does *not* deliver the decisive advantage its theory implies. On data
the model itself generated, with no misspecification and a validated exact sampler, Besag's
pseudo-likelihood — the thing that looked like a hasty patch — recovers the structure
equally well and the scale only slightly worse.

**Caveat that cuts both ways.** Neither objective recovers well in absolute terms:
correlation 0.85 and relative error 0.57 against a perfect 1.0 and 0.0. With 564 free
parameters and roughly 6,750 item observations, most of the error is estimation noise
common to both objectives, so this design has limited power to separate them. A larger
synthetic sample would give a sharper comparison, and is the obvious next run.

**Verified along the way:** the exact sampler reproduces enumerated whole-basket
probabilities to within sampling noise (total variation 0.0073 against a noise floor of
0.0074, 18 baskets, 60,000 draws), and `sample_exactly_k` matches exact subset
probabilities on three configurations.
