# TODO

Ordered by value. Each item states what is known, what is assumed, and what would settle it.

## 1. rho_c at sub-commodity granularity

`rho_c` is the model's only broad basket-conditioning channel: it fires for 47.2% of
held-out items against phi's 4.5%. But it is fitted at **commodity** granularity (280
groups), and when it fires it boosts **518 products on average** -- the largest commodities
hold up to 1,774 -- so it promotes ~99 wrong candidates per right one. Measured cost:
**-22% MRR on validation, -35% on test** (0.0504 -> 0.0395, 0.0459 -> 0.0297).

Sub-commodity has 758 groups, median 3 products, and boosts **43** when it fires -- 12x more
specific for 2.7x less coverage (fires 17.8%).

Known: this will NOT flip rho_c's sign. Within-group clustering strengthens at finer
granularity (obs/null 1.47 at commodity, 4.15 at sub-commodity, 5.35 at brand x
sub-commodity), so rho_c stays negative. The gain is precision, not a new mechanism.

Cost: the partition defines the ragged row structure, so C goes 280 -> 758 and rows get
smaller, which should let R drop below 23 and may make training *faster*. Every existing
checkpoint becomes incomparable; needs a fresh run.

## 2. Build the phi mask by LIFT, not by co-purchase MASS

`pairmask.py` ranks by total co-purchase count, which favours frequent staples: the 20
selected are buns, milk and salad vegetables. Frequent items co-occur often but not
necessarily with the highest lift. Selecting by lift should spend the scarce `c` budget on
structure that popularity does not already capture.

Current mask does work: correlation with empirical log-lift is +0.486 over its 190 pairs,
and its strongest learned pair (HAMBURGER BUNS + HOT DOG BUNS, phi'phi 0.517) is the
strongest real pair in the data (lift 9.65). It is capped at roughly a third of true
strength by the `c` budget.

## 3. Per-product price response, and its coupling to Var(n)

`beta` has **no relationship** to empirical per-product price response (corr +0.016), and a
20% coupon on one product moves its purchase probability by 0.8% (own elasticity -0.008).
So coupon targeting is unsupported: the action has no effect.

The cause is structural, not a missing penalty. Proposition 1 ties the two calibrations:

    aggregate elasticity = -(gamma.beta) * Var(n) / E[n]

The fitted model has gamma.beta = 0.0154 and Var/E = 11.25, giving -0.174 (measured -0.117,
data -0.121). A realistic per-product elasticity of -1 would need gamma.beta ~ 1, which at
Var/E = 11.25 implies an aggregate of -11. **The two targets cannot both be met** while the
price channel acts only through basket size.

What this means: the model has no product-level substitution. A price cut on one product
can only make baskets *bigger*, never shift share from a competitor. phi phi' is PSD so it
cannot express substitution, and rho_c is fitted negative at every granularity.

Open, and needed before any coupon MDP:
  - a price-conditional test of whether share genuinely shifts between close substitutes
    (the co-occurrence test above does NOT answer this)
  - if it does, an indefinite interaction form, since a Gram matrix cannot represent it

## 4. The pricing MDP has no interior optimum

Margin rises monotonically with price; the optimum is at the boundary. Units elasticity is
-0.99 to -0.64 over the tested range, so |e| < 1 everywhere, revenue rises with price and
cost falls with it.

Root cause: trip occasions are **replayed from data**, so a price rise costs zero visits.
The force that creates an interior optimum -- customers leaving -- cannot be represented,
because the model has no theory of arrivals.

Options, in increasing order of work and honesty:
  - a price band (how markdown is actually posed; encodes that the action space is bounded)
  - a volume or share floor
  - a participation model: visit frequency as a function of prevailing prices. Trip
    timestamps per household make this estimable, but it is a new component.

## 5. Unaudited: the units model

Most of the MDP's price response comes from the shifted-negative-binomial units model
(-0.71 of the -0.83 total), not from the basket model. Nothing in this project's auditing
has touched `a_q`, `gamma_q`, `beta_q`, `log_r`.

## 6. E[n] and Var(n) remain uncalibrated

Training to 2.0 epochs improved likelihood by 2.9 nats and did not move either. E[n] 8.95
against 7.87 observed on validation, 9.88 against 8.04 on test; varpop 150 against 85. The
failure is concentrated: median E[n] tracks well (6.6-7.1) while the mean is dragged by a
tail of trips predicting very large baskets.
