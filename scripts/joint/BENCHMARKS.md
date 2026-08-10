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
