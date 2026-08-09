# Two investigations on branch `causal-mdp`

Both were opened by §13 of `paper/experiments.html`: promotional placement as the missing
identification signal, and whether the generator can drive a Markov decision process.
Neither result is what the handoff predicted.

---

## 1. Promotion was 32% of the price coefficient

### What was wired in

`pipeline/23_promo_data.py` turns dunnhumby's `causal_data.csv` (36.8M rows, 664 MB) into
a binary `(item, store, week)` panel of display and mailer placement, written to
`basket_input/promo.npz`. It does **not** rebuild `basket_input/` — re-running stage 22
would renumber items, stores and splits and invalidate every fitted model — so the store
map is recovered exactly by joining `BASKET_ID` against `data/tx.parquet`. All 115 stores
resolve.

One structural fact decides the encoding, and it is measured rather than assumed: **every
one of the 6,522,942 rows on our products has `display ≠ 0` or `mailer ≠ 0`**. The file
records only promoted cells, so absence is a genuine zero and the panel is dense binary.

`model/27` gained `--use-promo`, adding `w^promo_j · (display, mailer)` to `b_ijt` with
per-product loadings. Off by default; the flag is a strict no-op when unset, confirmed by
`33_verify_equations` still passing 30/30 on `ps_nested`.

### The first stage is strong, and it is the mailer

| | cells | mean Δlog p |
|---|---|---|
| no promotion | 305,340 | **+0.0217** |
| display only | 204,479 | +0.0015 |
| mailer only | 6,652 | **−0.0953** |
| display + mailer | 39,939 | **−0.1303** |

Promoted minus unpromoted is **−0.0437 log points, t = −98**. In-store display barely moves
price; the weekly circular moves it a lot. That is the right split — the circular is where
price promotions are announced, while display placement is merchandising.

### Controlling for it moves the estimand by a quarter

`pr_on` is `ps_nested`'s recipe and seed with `--use-promo`.

| | no control | + promotion |
|---|---|---|
| item log-likelihood | −2.5118 | **−2.4841** |
| top-1 | 0.3396 | 0.3466 |
| incidence NLL | 0.1206 | 0.1204 |
| **price coefficient** | **+0.6951** | **+0.4753** |
| κ | 0.8723 | 0.8725 |
| wrong-sign share | 24.02% | **30.03%** |

The price coefficient falls **31.6%** — 21 refit standard deviations — while the fit
*improves* by +0.0278 nats, 2.5× refit noise. Fitted loadings are positive on both
channels (display +0.134, mailer +0.156): placement raises purchase probability on its own.

Carried through to the estimand:

| margin | no control | + promotion |
|---|---|---|
| allocation | −0.9562 | −0.6956 |
| incidence | −0.1393 | −0.0925 |
| quantity | −0.1115 | −0.1115 |
| **total own-price elasticity** | **−1.2070** | **−0.8996** |

**−25.5%.** The quantity margin is untouched, correctly — `w^promo` enters the choice
margin only, and the quantity head carries its own price coefficient.

### What this does and does not establish

**Does.** A quarter of the headline elasticity was promotion, not price. The placebo could
not have found this: permuting the price panel destroys the promotion association too, so
it collapses either way. This is the confound §6 of the experiments page names as the open
one, and it is now measured rather than speculated about.

**Does not.** This is a **control, not an instrument**. Being in the weekly circular is
advertising, which moves demand on its own, so mailer fails the exclusion restriction and
cannot instrument for price. Absorbing the promotion channel leaves the residual closer to
a pure price response, but it is still not identified against unobserved demand shocks the
retailer responds to — a category the buyer expects to sell well gets both a promotion and
a price cut, and neither variable separates those.

**One result points the wrong way and should not be buried:** the share of
(household, product) pairs with an economically wrong sign rises from 24.0% to 30.0%. With
the promotion channel absorbed, more of the residual price variation is being fitted with
upward-sloping demand. That is what you would expect if the remaining variation is
substantially endogenous, and it argues the corrected elasticity is still not clean.

---

## 2. The rollout does not compound, and the stated lower bound is not one

`eval/48_mdp_rollout.py` implements closed-loop trajectory simulation at three feedback
levels. Every generation result in the repository before this is **open loop**: recency is
read from the household's real history, so a generated novel product never makes the next
generated basket more novel.

| mode | what feeds back |
|---|---|
| `open` | nothing — recency from the real history (what the repo does today) |
| `recency` | generated purchases update `d.state` / `d.cat_state` (the true MDP transition) |
| `full` | recency plus the frozen habit counts `loyal`, `freq`, `hh_cat` |

Open mode reproduces `46_horizon` exactly — real 3.12 → 35.19, drift 1.165 → 1.262 against
its 1.156 → 1.261 — which validates the implementation before anything is concluded from it.

### The result

| mode | drift, step 1 → step 12 |
|---|---|
| open | 1.165 → **1.262** |
| recency | 1.165 → **1.261** |
| full | 1.165 → **1.270** |

§11 of the experiments page calls its drift measurement *"a lower bound, since a true
rollout would compound further."* **It is not a lower bound.** Closing the loop on the
recency state moves the twelve-step ratio by **0.001**. Updating the frozen habit counts as
well moves it to 1.270 — 0.008, and in the direction of slightly *more* drift, not less.

The novelty excess is therefore a **per-trip bias that accumulates linearly**, not a
feedback loop that amplifies. That is a materially better failure mode: a constant bias can
be calibrated out, a feedback loop cannot. It also means the frozen habit terms — which
looked like the obvious structural ceiling, since a simulated household can never become
loyal to something new — are **not** the binding constraint.

### Efficiency

**10.2 ms per basket**, single-threaded, 105 baskets/second. Profile:

| | share | what it is |
|---|---|---|
| `cat_utilities` | **70%** | all 188 categories × up to 225 items per trip — ~42,000 item utilities to place ~8 items |
| `state` → `searchsorted` | **25%** | recency by binary search on a 1.2M-element key array |

A policy-learning run needing 10⁶ trajectory steps costs ~2.6 hours single-threaded, or
about 20 minutes across 8 cores. That is not the bottleneck for this project.

Two optimisations follow directly from the profile, and one is already demonstrated:

1. **Dense recency instead of `searchsorted`.** `SimHistory` stores last-purchase day as an
   `(H, S)` array and indexes it in O(1). The closed-loop modes ran *faster* than open —
   109 against 103 baskets/s — despite doing strictly more work. Worth ~25%.
2. **Cache or subsample `cat_utilities`.** `b` does not vary within a trip, and incidence
   needs only the inclusive value per category. Caching per `(household, store, week)`, or
   sampling categories with importance weights, targets the 70%.

### What is actually missing for an MDP

In order of how much they matter, with evidence rather than assertion:

1. **Trip timing is not modelled at all.** The model conditions on the day and never
   generates it, so inter-trip gaps are exogenous and this rollout holds them at the
   household's real trip days. A retailer's central question — *if I promote this week,
   does the household come in sooner?* — is not answerable, and no amount of basket
   accuracy fixes that. This needs a trip-incidence hazard and is the largest gap.
2. **Prices are exogenous.** The model conditions on Δlog p and never generates it, so a
   policy's action has no effect on future prices. Now that promotion is in the data, a
   placement policy is at least representable, which was not true before §1.
3. **Basket size is biased 9% low** (7.47 items against 8.18) and categories 12% low. A
   policy trained on this systematically under-estimates volume, and the cause is not
   established.
4. **Reward is revenue only.** Absolute prices exist — `log_price.npy` is finite over the
   whole grid, median $1.89 — so `Σ price × units` is computable per generated basket. What
   is missing is cost, so margin, which is what a pricing policy actually optimises.
5. **No budget constraint**, so a policy cutting prices everywhere produces unbounded
   purchasing rather than substitution.
6. **The frozen habit terms are not worth fixing yet** — measured at 0.008 on the twelve-step
   ratio. This is the one item on the list that the evidence *removes*.

---

## 3. Two corrections to the generator, and what they buy

Both follow from §2's finding that the drift is a linear per-trip bias rather than a
feedback loop: a per-trip bias is exactly the thing a per-trip correction can remove.
Both are off by default, so every existing artefact reproduces unchanged.

### `require_nonempty` — implement the n ≥ 1 conditioning at generation

Spec Eq. 8 divides by `1 − ∏_c P(y_c = 0)`; the training loss implements it, the generator
never did and emitted an empty basket **4.42%** of the time against a real 0%. Redrawing
the composition until something is entered is exact rejection sampling from the
conditioned law.

Measured: empty rate **4.42% → 1.80%**, mean items **7.479 → 7.572**. That closes about
**10%** of the basket-size shortfall, not the 39% predicted beforehand. The prediction
assumed conditioning on the finished basket being non-empty; what Eq. 8 actually says, and
what is implemented, is that the *composition* is non-empty. The residual 1.80% are trips
that enter a category and then place no item in it — a separate defect, and now a
localised one.

### `item_temp` — sharpen the item draw without touching the inclusive value

The temperature sweep in `42_limitations` scaled `m.item_utility` globally, which also
scales IV, which feeds incidence — so sharpening inflated basket size 7.55 → 12.39 and the
result was confounded. That section said a clean test "would scale only inside the item
softmax and leave IV alone" and had not been run. It is now: IV is taken from the
untempered utilities, and the temperature applies only to the pass-1 draw and the Gibbs
resample.

| item_temp | items | categories | novel item % | never-bought sub % |
|---|---|---|---|---|
| 1.00 | 7.572 | 5.667 | 49.17 | 18.96 |
| 0.90 | 7.572 | 5.667 | 45.69 | 17.66 |
| 0.85 | 7.572 | 5.667 | 43.75 | 16.98 |
| **0.80** | **7.572** | **5.667** | **41.95** | 16.28 |
| *real* | *8.360* | *6.494* | *41.87* | *12.01* |

**The decoupling works exactly.** Basket size and category count are identical to three
decimals across the whole range, where the confounded version moved them by 64%. And
**T = 0.80 calibrates the novel-item rate to within 0.1 point** of real.

The sub-commodity never-bought rate improves only 18.96% → 16.28% against a real 12.01%,
and that is informative rather than disappointing: sharpening fixes *which item within a
sub-commodity* gets picked, but which sub-commodity is reachable at all is set by incidence
and breadth, which the temperature now deliberately does not touch. The residual novelty
is a composition-stage problem, and this localises it.

### The payoff on rollout drift

Closed-loop, 600 households, twelve trips, `recency` mode:

| | step 1 | step 4 | step 8 | step 12 |
|---|---|---|---|---|
| baseline | 1.165 | 1.217 | 1.253 | **1.261** |
| + both corrections | **1.016** | 1.040 | 1.097 | **1.106** |

Twelve-step drift falls from a **26% excess to 11%**, and step-one novelty is within 1.6%
of real. Throughput is unchanged at 112 baskets/s — the rejection loop costs nothing
measurable because it fires on 4% of trips.

This does not make multi-step rollouts sound. Basket size is still 9% low, the composition
stage still over-produces unfamiliar sub-commodities, and trip timing is still not modelled
at all. But the horizon over which a trajectory stays usable is materially longer than it
was, and it was bought with two localised changes and no refit.

---

## 4. A quarter of the price association sits in imputed prices

Appendix B.1 records that only ~25% of the item × day price grid is observed and the rest
carries the last price forward, then moves on. Nobody had asked whether the fitted response
comes from the observed minority or the imputed majority. `eval/50_price_observability.py`
asks it, model-free and with no refit. An item-day is observed exactly when at least one
transaction line exists for it that day — the same condition under which stage 22 writes a
real median price instead of carrying one forward.

### The descriptive table is confounded and gives the wrong sign

| price support on the week's representative day | cells | share | elasticity |
|---|---|---|---|
| carried forward (0 lines) | 32,045 | 64.2% | −0.4102 |
| 1 line | 12,639 | 25.3% | −0.5056 |
| 2–4 lines | 4,533 | 9.1% | −0.6049 |
| 5–19 lines | 633 | 1.3% | −0.9505 |

Read directly this says the response is *stronger* where prices are observed, so imputation
merely dilutes it. That reading is wrong. Median training lines per item across the four
strata run **125 / 152 / 293 / 1,929**, over **5,158 / 4,628 / 2,257 / 212** distinct items:
the strata are different item populations, and the gradient is popularity, not
observability. Popular items are the promoted, brand-switchable ones that §9 of the
experiments page already shows are the elastic ones.

### Controlled, it reverses

Interacting the elasticity with "the price was observed", within item, restricted to the
3,869 items that supply both kinds of week (39,244 cells, 39.4% observed):

| | elasticity |
|---|---|
| carried-forward weeks | **−0.7383** (se 0.0144) |
| differential when observed | **+0.1959** (se 0.0056), t = **+35.3** |
| implied on observed weeks | **−0.5424** |

The association is **weaker where prices were genuinely observed**, and about **27%** of it
sits in cells whose price was imputed rather than measured. The headline elasticity is
partly a property of the carry-forward rule.

This compounds with §1 rather than duplicating it. Promotion accounted for 32% of the
*coefficient*; imputation accounts for ~27% of the *association*, and the two are different
mechanisms — one is an omitted cause of price, the other is measurement of price. Neither is
addressed by the placebo.

### Three readings of one test, and why the last is right

Worth recording, because the first two were mine. The stratified table said "stronger where
observed". A sign bug in the scripted conclusion — comparing negative elasticities with `<`
instead of by magnitude — reported "weaker" for the wrong reason and by accident got the
direction that later proved right. Only the within-item interaction, which holds the item
population fixed, gives an answer that means anything. The lesson is the design, not the
arithmetic: a stratified comparison across selected populations cannot answer a question
about measurement.

---

## Reproducing

```bash
export OMP_NUM_THREADS=1
cd scripts/pipeline && python3 23_promo_data.py             # ~4 min, writes promo.npz
cd ../model && python3 27_nested_basket.py --label pr_on \
    --iters 6000 --l2-incidence 1e-4 --seed 0 --use-promo   # ~25 min
cd ../eval
python3 49_promo_identification.py                          # the table in §1
python3 48_mdp_rollout.py --label ps_nested --n-households 600 --horizon 12
```
