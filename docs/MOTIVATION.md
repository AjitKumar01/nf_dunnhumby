# Why this exists, and what a retailer can do with it

## The question a retailer actually has

Not *"how many units of product X will sell next week?"* — that question has been answered
adequately for decades by per-SKU forecasting. The questions that are hard, and that per-SKU
models answer badly or not at all, are the ones where **the basket is the unit**:

> If I cut the price of a store brand by 15%, how much of the lift is **new demand**, how
> much is **stolen from the brand next to it**, and does the customer's **total basket** get
> bigger or just rearranged?

> If I send this household a coupon, will they buy something they otherwise would not — or
> would they have bought it anyway at full price?

> If I delist this product, how much of its revenue **walks out with it** because it was
> bought alongside things I still stock?

Each of these is a question about *a set of products bought together*, under a price change
that did not happen. That is what this model is for.

---

## Why a basket model, and not a shelf of per-SKU regressions

A per-SKU demand model treats each product's sales as its own time series. It works, and it
fails in three specific ways that matter commercially.

**It double-counts.** Fit a promotion lift for product A and separately for product B, and
if A and B are substitutes you have counted the same customer twice. Category managers
handle this with judgemental "cannibalisation factors". A basket model does not need them:
substitution falls out of the joint distribution.

**It cannot answer basket questions.** "Does the basket get bigger?" is not expressible in a
per-SKU model — there is no basket in it. Yet that is the question behind every markdown
decision, because a promotion that grows units but shrinks margin per trip may be a loss.

**It cannot be simulated.** To evaluate a *pricing policy* — a rule that sets prices week
after week, reacting to what happened — you need to generate plausible baskets under
prices that never occurred. A regression predicts a conditional mean; it does not produce
baskets you can push through a P&L.

This repository models the **joint distribution over the whole basket**, so all three go
away at once. The price is that the normalising constant sums over $2^{5312}$ possible
baskets — which is the technical problem [`THEORY.md`](THEORY.md) spends its length solving.

---

## What it can do today

Three capabilities, in order of how well established they are.

### 1. Score any basket, and rank what comes next

Given a household, a store, a day and prices, the model assigns a probability to any basket.
Two immediate uses:

- **Complete-the-basket recommendation.** Rank every product in the store's assortment by
  $\pi_j = P(j \in S)$. Measured: MRR 0.076, Recall@5 12.3%, median rank 380 out of ~5,300.
- **Anomaly and audit.** A basket the model finds very unlikely is worth a look — mis-scans,
  fraud, or a household whose behaviour has shifted.

**How good is that, honestly?** Recall@5 of 12% means that when you offer five suggestions,
roughly one in eight times one of them is a product the customer actually bought on that
trip. Useful for a slot on a receipt or an app; not a solved problem.

### 2. Answer price counterfactuals

This is what the model is built for, and where most of the engineering went. It answers
three *different* price questions with one mechanism:

| question | model quantity | fitted | data says |
|---|---|---|---|
| I raise **one** product's price 10%. What happens to it? | own-price elasticity | $-0.68$ | $-0.79$ |
| ...and what happens to its **rivals**? | cross-price elasticity | $+0.044$ | $+0.099$ |
| I raise **all** prices. Does the basket shrink? | aggregate elasticity | $-0.13$ | $-0.12$ |

Own-price and aggregate are close to what the data shows. **Cross-price is a third of where
it should be** — the model substitutes in the right direction but too weakly, and
[`THEORY.md` §12](THEORY.md) explains exactly why and what would fix it.

**What that supports today.** Directionally-correct markdown analysis for a single product,
including the effect on the basket. **What it does not yet support.** Precise cannibalisation
accounting between close rivals, because that is the number that is a third too small.

### 3. Serve as a simulator

The model can **generate baskets**, not just score them — exactly, with no Markov chain
([`THEORY.md` §9](THEORY.md)). Set any price vector you like, sample a week of trips, and
push them through a margin calculation. That makes it a *retail environment* in the
reinforcement-learning sense: a pricing or coupon policy can be evaluated over thousands of
simulated weeks before it touches a real store.

Measured: generated basket sizes reproduce the population (model $\mathbb{E}[n]=8.0$ against
7.82 observed) but **compress between customer segments**, and only 1–4 of the top 10
products per segment match. So aggregate simulation is trustworthy; segment-level simulation
is not yet.

---

## A worked commercial example

Take a household that buys a mid-price coffee. The retailer is considering a 15% price cut
on the store brand next to it.

A per-SKU model says: store-brand units rise by its own elasticity, and someone applies a
cannibalisation factor from a spreadsheet to guess the effect on the branded coffee.

This model answers in one pass, because all three come from the same joint distribution:

1. **Own effect** — store-brand $\pi_j$ rises. At the fitted own-price elasticity of $-0.68$,
   a 15% cut lifts its purchase probability by roughly 10%.
2. **Cross effect** — the branded coffee's $\pi_k$ *falls*, because the store brand became
   relatively cheaper within its reference group ([`THEORY.md` §10.4](THEORY.md)). The model
   gets the sign and mechanism right; the magnitude is currently understated.
3. **Basket effect** — total basket size barely moves. The aggregate elasticity is $-0.12$:
   groceries are necessities, and a cheaper coffee does not make people buy more of
   everything else. **This is the number a per-SKU model cannot produce at all**, and it is
   often the one that decides whether the promotion pays.

---

## Where it is not ready

Stated plainly, because a model used past its evidence is worse than no model.

| gap | consequence | status |
|---|---|---|
| Cross-price elasticity is a third of target ($+0.044$ vs $+0.099$) | cannibalisation between close rivals is understated | cause identified; fix is a blended price reference |
| **Household price sensitivity is not identified** — deleting it changes ranking by 0.1% | the model personalises *what* you buy, not *how you respond to price*. **Coupon targeting needs the second** | open |
| Demand is inelastic ($-0.68 > -1$) | revenue rises monotonically with price, so the model alone gives **no interior optimum** for a pricing policy — competition and the outside option, which are not in the data, must supply it | inherent to this dataset |
| Segment-level generation compresses | simulate at population level, not per segment | open |
| Elasticity targets rest on **one** estimator | a second, independent estimator disagrees at 22.9σ; if the target is wrong, the whole price calibration moves with it | open, and the most important one |

That last row is the one to watch. Everything in the price block is calibrated to an
own-price elasticity of $-0.79$ estimated from an item-week panel. A cross-store estimator
gives $-0.13$. The second is built on thin data and is probably attenuated — but *probably*
is doing real work in that sentence.

---

## What this is not

- **Not a causal inference framework.** Prices here were set by a retailer, not randomised.
  The elasticities are conditional associations with fixed effects and promotion controls,
  not experimental effects. Used for ranking decisions they are informative; used as
  guaranteed treatment effects they are overclaimed.
- **Not a production system.** It is a research model with a reproducible pipeline. There is
  no serving layer, no monitoring, no retraining schedule.
- **Not validated on coupon campaigns.** The dunnhumby coupon data uses a different product
  and household ID space (560 products, 2,084 users against this model's 5,455 and 2,066),
  so the coupon-targeting claim above is a *capability*, not a measured result.

---

## Where to go next

| you want | read |
|---|---|
| the mathematics, built from scratch with a worked example | [`THEORY.md`](THEORY.md) |
| how the code is laid out and what runs where | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| how raw dunnhumby CSVs become tensors | [`DATA_TO_MODEL_INPUT.md`](DATA_TO_MODEL_INPUT.md) |
| to just train it | [`../README.md`](../README.md) |
