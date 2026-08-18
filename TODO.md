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

## 3. The per-product own-price effect is ~80x too weak

Measured, on run90_best:

    own-price   d log(purchases) / d log(price)    data -0.659    model -0.008
    cross-price d log(B)         / d log(p_A)      data -0.057    model -0.0002
                                                   (t = -1.4, i.e. indistinguishable from 0)

So a 20% coupon on one product moves its purchase probability by 0.8%, and coupon
targeting is unsupported: every allocation rule scores the same because the action does
nothing.

**Correction to an earlier reading.** This was recorded as "the model has no product-level
substitution, and phi phi' is PSD so it cannot represent it, therefore an indefinite
interaction form is needed."  That overstated the problem.  A cross-price regression over
3,956 sibling pairs in the same sub-commodity finds **no detectable substitution in the
data either** (cross coefficient -0.057, t = -1.4, positive in 48.6% of pairs -- a coin
flip).  The model reproducing ~zero cross-price response is therefore FAITHFUL, not a
defect, and no new interaction form is required for it.

The real gap is single-dimensional: the own-price coefficient.  gamma.beta = 0.0154 where
the data implies something near 1.  Proposition 1 explains why the fit chose that:

    aggregate elasticity = -(gamma.beta) * Var(n) / E[n]

with Var/E = 11.25 the fitted 0.0154 already produces -0.174 against a -0.121 target.  A
realistic gamma.beta ~ 1 would give an aggregate of -11.  The two calibrations are in
direct conflict *while price acts only through basket size*.

What would settle it: give price a share-shifting route, so a cut can move WHICH product is
bought rather than only how many.  Then gamma.beta can be large without the aggregate
exploding.  Until then the model supports chain-wide markdown and not targeted coupons.

Caveat on the data figures: the regression is confounded by promotions, seasonality and
endogenous pricing, so -0.659 is indicative rather than a clean elasticity.  The comparison
that matters is the order of magnitude, which is two.

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
