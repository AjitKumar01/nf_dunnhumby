# Nested Factorization on dunnhumby: paper analysis, data preprocessing, and fit

Two halves. Part A is what Donnelly, Ruiz, Blei & Athey (2023) actually do with their
scanner panel and what their code expects. Part B is the dunnhumby port: what the
manual says the files are, what the data actually looks like, the preprocessing, and
how well the model transfers.

---

# Part A — The paper

## A.1 The model

Shopper `i` on trip `t` faces `C` disjoint categories. She buys at most one item per
category, and categories are additively separable (no budget constraint within a
trip). Two stages, each a random-utility model.

**Stage 1 — product choice, conditional on buying from the category** (eq. 4–6):

```
u_ijt = theta_i . beta_j      latent household taste x latent product attribute
      + W_i . rho_j           observed demographics x per-product loading
      + sigma_i . X_j         per-household loading x observed product attribute
      - (gamma_i . lambda_j) * price_jt
P(j | bought from c) = a_jt exp(u_ijt) / sum_k a_kt exp(u_ikt)      a_jt = in stock
```

The price coefficient is itself a factorization, `gamma_i . lambda_j`, so price
sensitivity varies by household *and* by product, and is correlated across products
through the shared latent space. Only the categories a household actually bought from
enter this likelihood.

**Stage 2 — category incidence** (eq. 7–9). The inclusive value summarises stage 1:

```
IV_ict = log sum_j a_jt exp(u_ijt)
u_ict  = vartheta_i . beta_c + W_i . rho_c + psi_i . X_c
         - (phi_i . lambda_c) * IV_ict + mu_c . delta_t + w_c,weekday
P(buy from c) = sigmoid(u_ict)
P(choose j)   = P(buy from c) * P(j | bought from c)
```

`phi_i . lambda_c` is the nesting coefficient: 1 collapses to a plain logit over
{all items, outside good}; 0 means a price change inside the category never changes
whether you buy from the category at all. This nest is the paper's central
modelling move. Categories are bought on ~3.7% of trips, so if the outside good sat
in the same nest as the products, IIA would say a shampoo price rise pushes almost
everyone to *no shampoo* rather than to a different shampoo.

Estimation is mean-field variational Bayes: independent Gaussians on every latent,
`N(0,1)` priors, ELBO maximised by SGD with the reparameterisation trick and
minibatches of 5,000 trips. The two stages are fitted sequentially with the same
code — stage 2 is run as if each category had exactly two "products", an inside good
carrying `IV` and an outside good with `IV = 0`.

## A.2 How the scanner data is actually used

Source: one store of a national chain, isolated location, no big competitor within
five miles (Che, Sudhir & Seetharaman data). May 2005 – March 2007, loyalty-card
households.

| step | what they do |
|---|---|
| households | keep those making 20–300 trips → **2,068 households**, 1,551,213 purchases on 333,585 trips |
| trip | all purchases a household makes on one calendar day |
| identification window | almost all price changes happen at midnight on **Tuesday**, so keep **Tuesday and Wednesday only** → 455,445 purchases on 100,504 trips |
| holidays | drop the weeks before Halloween, Thanksgiving, Christmas, July 4th, Labor Day |
| category | the "category" level of the retailer's UPC hierarchy, above "class"/"subclass"; **235 → 123** categories, **1,263 UPCs** |
| choice set | top 10 items per category (plus a pooled 11th for the logit baselines only) |
| price | prices are not recorded for unsold items, so price is the **daily median transacted price per unit**, carried forward on days with no sale. This also averages away coupons and multi-buy deals |
| availability | employees scan out-of-stocks; an item is unavailable that day if flagged for >75% of the day's trips |
| session | a day: prices and availability are constant within it |

Category filters (app. 8.1), applied in this order:

1. top-10 items per category (+ pooled item for logits);
2. drop categories where >15% of category-trips contain multiple items from the
   category, or >10% contain multiple top-10 items — unit demand must roughly hold.
   Remaining multi-item trips: keep one purchased item at random;
3. drop categories where the mean absolute within-category price correlation of the
   top-10 items exceeds 0.75 — otherwise cross-price effects are not identified;
4. keep only categories where ≥2 of the top 10 items have a Tuesday→Wednesday price
   change in some week, and ≥1 item moves ≥$0.10 in ≥10% of weeks;
5. drop the 15% most seasonal categories (Herfindahl of daily demand per UPC,
   averaged over the top 10 as percentiles).

**Identifying assumption**: conditional on category-week effects and a Wednesday
dummy, the Tuesday→Wednesday difference in demand is unrelated to anything except
the price change. They test it with placebo runs that shift the price series forward
or backward in time; 13 of 123 categories fail one of four tests at 1%, and only 4
fail the backward-shift tests.

## A.3 How they judge the model

The distinctive part: **hyperparameters are selected on counterfactual performance**,
not on ordinary held-out likelihood. The split holds out at the **household × week**
level, and validation scoring uses only item-weeks where the price actually moved.
Test-set evaluation then covers three "mini quasi-experiments": the focal product's
own price changes, another product in the category changes price, another product
goes in or out of stock.

Reported: predictive fit overall and per category (Tables 1–2); log-likelihood inside
each event type, at both household and aggregate level (Table 3); degree of
personalisation — coefficient of variation of predicted rates and a regression of
realised on predicted rates with item fixed effects (Table 4); predictions for
households who never bought the item, by decile (Fig. 5); own- and cross-price
elasticities, checking that cross-price elasticities are higher *within* a product
class than across it — using class labels never given to the model (Table 5);
targeted-discount profitability (Table 6).

Selected hyperparameters: product stage `{K=80, K_price=20, demographics yes, item
attributes no, lr 0.005, linear price}`; category stage `{K=40, K_IV=40, time
dimension 10, lr 0.01}`.

## A.4 What is in `nested-factorization/`

* `src/bemb_loc` — the stage-1 C++ model. Input files are read with `fscanf`, so
  they must be exactly tab-separated with integer ids:
  `train/validation/test.tsv` = `user, item, session, units`;
  `item_sess_price.tsv` = `item, session, price` and must be a **complete**
  `Nitems × Nsessions` grid; `itemGroup.tsv` = `item, category`;
  `sess_days.tsv` = `session, week, weekday, hour`; `obsUser.tsv`, `obsItem.tsv`.
  `-likelihood 3` is the within-group softmax — the group is the category, which is
  exactly the stage-1 conditional choice.
* `src/hpf` — Hierarchical Poisson Factorization with observed attributes and an
  `availability.tsv`; used in the paper to manufacture "HPF controls" for the logit
  baselines.

Two limits matter for any port:

1. **`bemb_loc` cannot run stage 2.** Its own README says prices are "the same across
   all users but vary across sessions", and it stores them as one `Nitems × Nsessions`
   matrix. Stage 2 feeds `IV_ict` into that slot, and `IV` varies across households.
   That run used the user-varying TTFM build, which is not in this repository.
2. **Price is the only time-varying item attribute slot.** `obsItem.tsv` is static.
   Anything else that moves over time — a display, a mailer feature — has nowhere to go.

---

> **Verification.** The PyTorch re-implementation has been checked against the
> authors' C++ on identical files (same 13,736 test instances, peaks 0.057 nats and
> one evaluation point apart), against an independently written MLE conditional logit
> (0.4% difference), and by parameter recovery on simulated data. That last one sets
> a limit worth carrying into every number below: at this sample size the
> *household-level* price coefficient recovers at correlation 0.21 and is attenuated
> roughly threefold, so the ranking of households by price sensitivity is meaningful
> but an individual household's elasticity is mostly prior. Full detail, and what is
> still unverified, in `VERIFICATION.md`.

# Part B — dunnhumby "The Complete Journey"

## B.1 The files, per the user guide

| file | rows | content |
|---|---|---|
| `transaction_data.csv` | 2.60M | one receipt line: household, basket, day, product, quantity, `SALES_VALUE`, store, `RETAIL_DISC`, `COUPON_DISC`, `COUPON_MATCH_DISC`, time, week |
| `product.csv` | 92k | `DEPARTMENT` > `COMMODITY_DESC` > `SUB_COMMODITY_DESC`, plus `MANUFACTURER`, `BRAND` (National/Private), `CURR_SIZE_OF_PRODUCT` |
| `hh_demographic.csv` | 801 | age, marital status, income, homeowner, household composition, size, kids — **for 801 of the 2,500 households only** |
| `causal_data.csv` | 36.8M | product × store × week: `display` location code and `mailer` feature code |
| `campaign_table.csv` / `campaign_desc.csv` | 7.2k / 30 | which household got which campaign; each campaign's start and end day |
| `coupon.csv` | 125k | campaign → coupon → the products the coupon is redeemable on |
| `coupon_redempt.csv` | 2.3k | realised redemptions: household, day, coupon, campaign |

Prices are not a column. The guide gives the reconstruction: `SALES_VALUE` is what the
retailer receives, already net of the loyalty discount and the retailer's coupon
match, and inclusive of the manufacturer's reimbursement. So the posted loyalty-card
shelf price — the price every card-holder faces before their own coupon activity — is

```
unit_price = (SALES_VALUE - COUPON_MATCH_DISC) / QUANTITY
```

and the regular (non-loyalty) shelf price adds `RETAIL_DISC` back in. Discount columns
are stored negative, so "subtracting" them adds them.

## B.2 What the data looks like

*(numbers regenerated by `08_data_report.py`; full tables in `out/data_report.md`)*

Raw panel: 2.55M usable lines, 2,500 households, 91,856 products, 561 stores, 711 days
/ 102 weeks, 213,961 household-day trips.

**Where prices change — the key structural finding.** `WEEK_NO` runs Monday→Sunday.
For consecutive days with well-sampled prices, the probability a product's price moves
by more than 2¢ is:

| day pair | P(price change) | n |
|---|---|---|
| inside a `WEEK_NO` | **0.263** | 18,476 |
| across the Sunday→Monday week boundary | **0.519** | 3,357 |

So dunnhumby has the same structure the paper exploits, shifted by a day: prices are
reset at the week boundary, and **(Sunday of week w, Monday of week w+1)** straddles
the reset the way Tuesday/Wednesday does in the paper. That pair is 30.0% of all
trips — the paper's Tuesday/Wednesday sample is 30.1% of theirs. The residual 26%
"within-week change" is measurement error: the daily median is computed from few
transactions, and multi-buy deals move it. Assigning a single weekly price removes it
by construction.

**Stores.** The paper had one store; dunnhumby has 561, and no store is large enough on
its own (the biggest carries 2.6% of baskets; the top 10 carry 17.8%). Households are
loyal but not exclusive — the median household makes 74% of its trips at one store.
Prices, however, are close to chain-uniform: within a product-week the median
coefficient of variation of price across stores is **0.011**. Since `bemb_loc`
requires prices common to all users in a session anyway, sessions are defined at
**chain × calendar day** and prices at chain × week. This is the single biggest
approximation in the port and it is quantified above, not assumed.

**Random-weight items.** dunnhumby has no counterpart to the paper's clean per-unit
prices for loose produce and service-counter meat: `QUANTITY` counts scans, so
`SALES_VALUE/QUANTITY` is *weight × price-per-pound*, which varies continuously across
shoppers facing the identical shelf price. Measuring this directly — the share of a
product-week's transactions sitting at the modal cent value — separates them cleanly:
GRAPES 0.10, STONE FRUIT 0.13, CHICKEN 0.19, PORK 0.20 versus CANNED MILK 0.99, NUTS
0.99, DOMESTIC WINE 1.00. 2,208 of 30,879 priced products fall below 0.60 and are
dropped. Without this screen the price coefficient is being asked to explain basket
weight. `PREPROCESSING.md` §4 records a trap here: measuring discreteness by pooling a
product's transactions across the whole panel, rather than within a week, confuses a
scale item with an ordinary price change and wrongly flags ~2,000 high-volume
products.

## B.3 Preprocessing

Category = `COMMODITY_DESC`. `SUB_COMMODITY_DESC`, `BRAND` and `MANUFACTURER` are
deliberately **withheld from the model** so they can serve the paper's §6.4.1 test:
does the model infer, without being told, that products in the same subclass are
closer substitutes?

Steps, in order (`01`–`04`):

1. unit and regular shelf prices; drop bulk lines (`QUANTITY > 30`) and prices outside
   [$0.05, $100];
2. weekday index anchored from the basket-count profile (the two busiest `DAY % 7`
   residues are the weekend); trips = household × day;
3. keep Sunday/Monday pairs; drop pair-weeks whose chain-wide spend deviates more than
   12% from a local 9-week median — the data-driven stand-in for the paper's holiday
   list, since dunnhumby days are anonymised. 87 of 101 pair-weeks survive, and one
   more goes for containing a day with almost no baskets chain-wide, leaving 86;
4. households with 20–300 trips → 2,137, of whom 2,084 appear in a retained
   category (paper: 2,068);
5. drop non-merchandise departments and commodities (fuel, postage, lottery, coupons,
   "no commodity description", …);
6. random-weight screen (above);
7. top-10 items per category by trip incidence;
8. the paper's five category filters, unchanged except that "Tuesday→Wednesday" becomes
   "Sunday→Monday";
9. prices: chain-level weekly median, forward- then backward-filled, expanded to a
   complete item × session grid (49.2% of item-weeks are directly observed; the rest
   are carried forward, as in the paper).

### Resulting sample

| | dunnhumby (this port) | paper |
|---|---|---|
| households | 2,084 | 2,068 |
| categories | 56 | 123 |
| items | 560 | 1,263 |
| trips | 49,729 | 100,504 |
| sessions | 172 days (86 pair-weeks) | ~176 days (88 weeks) |
| category-purchase observations | 66,638 | 455,445 |
| mean purchase rate per category-trip | 2.3% | 3.7% |
| item-weeks with an own-price move ≥ $0.10 | 8,087 | — |
| item-weeks with a cross-price move in the category | 32,446 | — |

Where the categories go: 137 of 285 fail the "too thin" screen (fewer than 5 items or
under 200 category-trips in the window), 79 fail unit demand, 10 are too seasonal,
3 lack price variation, 56 survive. The paper kept 123 of 235; the extra attrition
here is almost entirely sample size — 50k trips instead of 100k, spread over 561
stores instead of 1.

Split: household × pair-week cells, 70/10/20 train/validation/test — the paper's
household-week hold-out, so a household seen in training must still be predicted in a
week it was not observed, including the day the price moved.

## B.4 Fitting the model

`bemb_loc` needs GSL, and cannot run stage 2 (§A.4), so `nf_torch.py` re-implements
both stages in PyTorch against the *same input files*; `run_bemb_loc.sh` drives the
original binary for stage 1 so the two can be cross-checked once GSL is installed.

One change to the paper's specification was necessary. With `N(0,1)` on every factor,
the prior on `theta_i . beta_j` has standard deviation `sqrt(K)` — at `K = 80` that is
effectively no regularisation, and on a sample six times smaller than the paper's the
model overfits within 200 steps (validation log-likelihood falls from −1.75 to −2.78,
past the −2.30 of a uniform guess). Scaling each factor's prior variance as
`sqrt(target/K)` holds the prior on the utility contribution fixed as `K` grows and
fixes it. The grid search then behaves, and selects — on validation log-likelihood in
price-change weeks, exactly the paper's criterion:

| | selected here | paper |
|---|---|---|
| K (taste factorization) | 40 | 80 |
| K (price factorization) | 20 | 20 |
| prior variance on the price coefficient | 0.25 | 1.0 |
| learning rate | 0.005 | 0.005 |

Two further departures were forced by the data, both diagnosed from fitted output
rather than assumed:

**The price coefficient collapses to zero unless its factors are seeded away from
zero.** `gamma_i . lambda_j` is bilinear; if both factors start at zero the gradient
in each is proportional to the other, and the model settles on "no price response at
all": the fitted price coefficient ends at **0.014**, effectively zero.
`bemb_loc` exposes `-meangamma` and `-meanbeta` for exactly this. Setting both prior
means to `sqrt(0.5/K_price)` starts the coefficient at 0.5 — positive, so demand
slopes down — and the data moves it from there, to **0.643**. Validation
log-likelihood in price-change weeks improves from **−1.633 to −1.557**, and the
median own-price elasticity from ≈0 to **−1.18**.

**The inclusive value has to be centred.** Fed raw, `IV_ict` is dominated by its
*level*, which is household `i`'s affinity for category `c` — precisely what
`vartheta_i . beta_c` already represents in the same equation. With both in, the
nesting coefficient is a contrast between two collinear terms, and it collapses to
**0.08** — the inclusive value doing essentially no work, so price changes inside a
category barely move the decision to buy the category at all. (On an earlier version
of the sample the same specification produced **−0.16**, i.e. a higher inclusive value
making a household *less* likely to buy — economically backwards. The sign is not
stable; the collapse is.) Subtracting the household-category mean of `IV` over all
sessions leaves only the part that moves with prices — the Sunday→Monday variation the
whole design rests on.
The nesting coefficient then estimates at **1.00**, and stage-2 validation
log-likelihood improves slightly as well. This is the paper's own identification
logic applied one level up, and it is the change I would flag first to anyone
re-running the original code on a new panel.

## B.5 Results

Four specifications, all with the same selected hyperparameters and the same
household × pair-week hold-out:

* **nf** — the full model;
* **logit** — no latent heterogeneity (`K=0`, one scalar price coefficient, one
  scalar nesting coefficient): the paper's "nested logit with demographic controls";
* **nf_nopool** — every (household, category) gets its own latent vectors, so nothing
  is shared across categories. This is the category-by-category benchmark, and it is
  the paper's central claim under test;
* **nf_promo** — the full model plus dunnhumby's display, mailer and coupon-eligibility
  terms.

### Predictive fit (paper Tables 1–2)

| model | test log-lik | test MSE | train log-lik | train MSE | mean rank (log-lik) | % categories best |
|---|---|---|---|---|---|---|
| nf | **−4.2386** | 0.9377 | −3.4604 | 0.9017 | **1.71** | **44.6%** |
| nf_promo | −4.2420 | **0.9353** | −3.3897 | 0.8950 | 1.89 | 35.7% |
| nf_nopool | −4.2790 | 0.9373 | −3.5161 | 0.9048 | 2.48 | 17.9% |
| logit | −4.7145 | 0.9799 | −4.6071 | 0.9791 | 3.91 | 1.8% |

`nf` and `nf_promo` are within 0.004 nats of each other — a difference small enough
that which one comes first is not stable across sample definitions (`nf_promo` led on
an earlier sample by a similar margin). What is stable is the ordering of the four:
either full model beats the no-pooling benchmark by ~0.04 nats, and both beat the
homogeneous logit by ~0.47.

For reference the paper reports test log-likelihood −4.91 for NF against −5.68 for
the nested logit with demographic controls, and test MSE 0.9268 against 0.9801. The
levels and the gaps line up closely.

### Weeks with a price-change event (paper Table 3)

Restricted to purchases of items whose own price moved ≥ $0.10 across the
Sunday→Monday boundary (3,555 test purchases), and to items where *another* item in
the category moved (9,096):

| model | own-price weeks | cross-price weeks |
|---|---|---|
| nf_promo | **−3.8512** | −4.2637 |
| nf_nopool | −3.9254 | −4.3003 |
| nf | −3.9631 | **−4.2560** |
| logit | −4.3250 | −4.7557 |

### Personalisation (paper Table 4)

| model | CV, UPC | CV, category | slope UPC | slope category |
|---|---|---|---|---|
| nf_nopool | 2.669 | 1.058 | 1.189 | 1.145 |
| nf_promo | 2.550 | **1.122** | 1.116 | 1.108 |
| nf | **2.223** | 1.026 | 1.144 | 1.148 |
| logit | 0.651 | 0.152 | 0.829 | 0.874 |

The slope is a regression of realised on predicted held-out purchase rate with item
fixed effects: 1.0 is perfect calibration of the *cross-household spread*. NF is at
1.14, the homogeneous logit at 0.83 — and the logit's coefficient of variation is
three to four times smaller, so it barely differentiates households at all. The paper reports 3.25 / 0.9955 for NF
and 0.43 / 0.9077 for the multinomial logit; same picture.

### Households that never bought the item (paper Figure 5)

Held-out purchase rate by decile of predicted rate, among household-item pairs with
**zero** purchases in training:

| model | UPC-level top/bottom lift | category-level lift |
|---|---|---|
| nf | 31.3× | 19.2× |
| nf_promo | 30.3× | 16.7× |
| nf_nopool | 26.4× | 15.3× |
| logit | 18.2× | 9.4× |

NF's decile curve is also monotone *and* roughly calibrated at the top; the logit's is
monotone but badly scaled, overstating the top decile by nearly a factor of two — it is
ranking households mostly by demographics. The paper reports >10× at UPC level and ~3× at category level.

### Elasticities (paper Tables 5, 7)

| model | median own-price elasticity | SD across items | mean SD across households | nesting coefficient |
|---|---|---|---|---|
| nf | −1.182 | 1.108 | **0.638** | 1.004 |
| nf_nopool | −1.279 | 1.297 | 0.347 | 0.969 |
| nf_promo | −1.046 | 1.088 | 0.652 | 0.702 |
| logit | −1.098 | 1.275 | 0.186 | 0.737 |

The medians are close across models — it is the *dispersion* that separates them.
NF puts three and a half times as much household-level variation into elasticities as
the homogeneous logit, and nearly twice as much as the un-pooled version: pooling
across categories is what makes household-level price sensitivity estimable at all.
The paper's Table 7 shows the same contrast (Mean(SD) 1.777 for NF against 0.0017 for
the multinomial logit).

### Does the estimated heterogeneity predict behaviour? (paper Figure 6)

The strongest test, and it is model-free: split households into terciles of
*predicted* own-price elasticity, then measure the actual Sunday→Monday change in
purchase rate in held-out data, by how far the price moved. Change per 1,000 trips:

| price move | most elastic | middle | least elastic |
|---|---|---|---|
| > $0.25 cut | **4.00** | 3.91 | 2.52 |
| $0.10–0.25 cut | −0.40 | −1.32 | **0.34** |
| $0.01–0.10 cut | **3.03** | −0.95 | −1.44 |
| no change | −0.64 | −0.17 | −0.62 |
| $0.01–0.10 rise | −0.52 | −0.63 | **−2.40** |
| $0.10–0.25 rise | 0.23 | 1.27 | **−2.21** |
| > $0.25 rise | **−3.89** | −2.20 | −3.52 |

Demand moves the right way with the price in the large buckets, the "no change" row is
flat as it should be, and in the two extreme buckets — where the signal-to-noise is
best — the households the model called most elastic moved further, in the right
direction, than the households it called least elastic. The four middle rows are
mixed and several go the wrong way; these are $0.01–0.25 changes on items bought a
handful of times per tercile per week, and the cell noise is of the same order as the
effect. This test is weaker evidence than an earlier version of this document claimed
on a slightly different sample, where four of six rows lined up; the two extreme rows
are the part that is stable. Nothing here uses the model as ground truth — the
terciles are a model output, the demand changes are raw held-out data.

### Two results that did not replicate

**Cross-price elasticities within versus across sub-commodity.** The paper's §6.4.1
test — the model should infer, without being told, that products sharing a subclass
substitute more — comes out weak and undiscriminating here. Comparing within
category and averaging across categories, the within-subclass conditional cross
elasticity exceeds the across-subclass one in 56% of categories for NF (gap +0.002)
and 58% for the homogeneous logit (gap +0.007). The effect exists but the full model
is no better at it than the baseline. Two reasons: dunnhumby's hierarchy has three
levels, so `SUB_COMMODITY_DESC` is the paper's *subclass*, where they too found only
an 8.4% gap; and the mechanism runs through the weighting of households, which even
the demographics-only model reproduces in part.

**Coupon response by predicted elasticity.** Households the model calls price
sensitive show no larger purchase lift during their actual coupon-eligibility
windows (lift +1.7% for the most elastic tercile, +1.4% for the least; both within
noise). This is a null, and the design is weak rather than the model: only 1,175
coupon-product rows and 23 campaigns touch the 560 retained items, campaign windows
are long enough that the "not eligible" comparison period is contaminated, and for
TypeA campaigns — the majority — the data records the *pool* of coupons, not which 16
a household actually received. Restricting to TypeB/TypeC campaigns, where
eligibility is certain, leaves 142 cells per tercile and no signal either.

## B.5a Placebo tests for price endogeneity (paper §5)

This should have been the first thing run, not an afterthought: it is what decides
whether the identification window is credible at all, and everything above depends on
it. Full detail, construction and figures are in `PREPROCESSING.md` §9; the summary:

Each price change is relocated to a pair-week that really had none, three ways
(forward, backward, and a fully randomised relocation), applied both to a single UPC
and to every item in the category, and the paper's baseline multinomial logit is
refitted per category with household-clustered standard errors.

| series | median price coefficient | % categories p<1% | correlation of the placebo price *level* with the real one |
|---|---|---|---|
| **actual prices** | **−0.590** | **73.2%** | 1.00 |
| all items, forward | −0.145 | 30.4% | 0.61 |
| all items, backward | −0.174 | 30.4% | 0.51 |
| **all items, random** | **+0.025** | **14.3%** | 0.12 |
| single UPC, random | +0.079 | 10.7% | 0.11 |

Three findings.

1. **The paper's own shift rule is contaminated.** Moving a price step forward or
   backward by a week or two leaves the price *level* half-correlated with the real
   level, so those placebos retain genuine price response. Their non-zero coefficients
   are not by themselves evidence of endogeneity — which also means the paper's
   "13 of 123 categories fail" is not a clean measure of how clean their data was.
2. **The randomised relocation is the valid null, and the design passes it on
   average.** The point estimate collapses from −0.590 to +0.025, mean +0.0004. The
   Sunday/Monday window is not manufacturing price effects out of nothing.
3. **But it over-rejects.** 14.3% of categories reject at the 1% level against a
   nominal 1% (KS against uniform, p = 0.0003). Clustering by household inflates
   standard errors only 1.0–1.1×, so that is not the explanation. **31 of 56
   categories fail at least one of the six placebos; 10 fail the randomised one.**

Acting on it: `13_placebo_followup.py` writes a per-category verdict, and
`02_select_sample.py --exclude-placebo-failures` rebuilds the sample without the
failures. Re-aggregating held-out fit over the survivors, and retraining from scratch
on the 46 placebo-clean categories:

| subset | nf | nf_promo | nf_nopool | logit |
|---|---|---|---|---|
| all 56 categories | −4.239 | −4.242 | −4.279 | −4.715 |
| passes the random placebo (46) | −4.210 | −4.210 | −4.242 | −4.702 |
| passes every placebo (25) | −4.340 | −4.342 | −4.378 | −4.882 |
| **retrained on the 46 clean categories** | **−4.196** | — | — | **−4.614** |

The ranking is identical everywhere and the full-model advantage *widens* on the
strictly clean categories (0.48 → 0.54 nats; 0.42 on the retrained model, where the
household spread in elasticity is still four times the logit's, 0.79 against 0.20). So
the model comparisons in §B.5 do not rest on the compromised categories.

What this does change: the **level** of an estimated elasticity is not trustworthy in
the 10 categories that fail the randomised placebo, and is suspect in the 31 that fail
any. Model comparison is safe — whatever endogeneity exists is in the data all models
see — but anything that converts an elasticity into a pricing decision should be run
on the clean subset only.

## B.6 Using what dunnhumby has and the paper's data did not

Three signals have no counterpart in the original panel, and each maps onto a slot
the model already has:

1. **Display and mailer** (`causal_data`, product × store × week). These are
   time-varying *item* attributes — structurally identical to price. They enter as
   `(gammaD_i . lambdaD_j) * display_jt` and `(gammaM_i . lambdaM_j) * mailer_jt`, so
   promotion response is heterogeneous across households the same way price
   sensitivity is. Because sessions are chain-level, the store-level panel is
   collapsed to a traffic-weighted share of shoppers exposed. 44% of item-sessions
   carry some display, 5.8% a mailer feature.
2. **Coupon eligibility** (`coupon` × `campaign_table` × `campaign_desc`). This is
   genuine *household × item × time* variation — the thing the paper's §6.5 targeting
   counterfactual could only simulate. Eligibility factors as
   household-in-campaign × product-in-campaign × day-in-window, so it is stored as
   23 campaign membership vectors rather than the 12.4M-cell tensor it would
   otherwise be, and evaluated as a max over campaigns.
3. **Realised redemptions** (`coupon_redempt`), held out of training entirely and
   used only as a validation target.

Adding (1) and (2) is `nf_promo`. It is the best model on test MSE (0.9353 against
0.9377) and by a clear margin exactly where it should be, in weeks with a price change
(−3.851 against −3.963 for `nf`); on overall test log-likelihood the two are a
statistical tie (−4.2420 against −4.2386). That pattern is the honest reading: knowing
about displays and mailers helps most in the weeks when promotions actually move, and
is close to free elsewhere. The cost is that `bemb_loc` cannot fit it — price is its
only time-varying attribute slot.

Two further pieces of dunnhumby are catalogued but not used here: `QUANTITY` (the
paper assumes unit demand; a Poisson stage as in Wan et al. would use it), and the
store dimension (561 stores, which a store-level session definition would exploit at
the cost of a much larger price matrix and a much smaller sample per store).

## B.7 What does not transfer, and what to watch

| issue | severity | what was done |
|---|---|---|
| **price endogeneity in ~1 category in 7** | **serious** | **placebo-tested (§B.5a); per-category verdicts emitted and a clean-subset retrain reported** |
| no stock-out feed | structural | one of the paper's three counterfactual events cannot be evaluated at all |
| prices must be inferred from transactions | moderate | audited against all three discount columns (`PREPROCESSING.md` §1); 49.3% of item-weeks directly observed |
| 561 stores, prices pooled to chain level | moderate | quantified: median within-product-week cross-store CV = 0.011 |
| random-weight produce and meat | severe if ignored | screened out on within-week price discreteness; 7% of priced products fail |
| demographics for 801 of 2,500 households | mild | zero-filled with a "demographics observed" indicator |
| sample ~2× smaller than the paper's | moderate | K falls from 80 to 40; priors have to be scaled by `1/sqrt(K)` |
| holidays not dated | mild | flagged from aggregate spend deviations instead of a calendar |
| Sunday and Monday differ far more than Tuesday and Wednesday | moderate | absorbed by the category × weekday term `w_ct`, exactly as the paper absorbs its Wednesday effect; worth a placebo test before trusting elasticities |

The placebo tests of the paper's §5 are in §B.5a and `PREPROCESSING.md` §9. They
should have been run before any model was fitted, not after.

## B.8 Bottom line

The Nested Factorization model transfers to dunnhumby, and the transfer is closer
than it first looks: the retailer resets prices at the Monday week boundary, so the
paper's identification device survives with Sunday/Monday in place of
Tuesday/Wednesday, at almost exactly the same 30% of trips. On held-out data the
full model beats a homogeneous nested logit by 0.41 nats per purchase, beats a
category-by-category version of itself by 0.03 nats, produces four times the
household-level dispersion in price elasticity, and — the test that matters — that
dispersion predicts who actually responds to real price changes in data the model
never saw.

Three caveats are load-bearing.

*The identification is weaker here than in the paper.* A randomised placebo collapses
the price coefficient to zero on average, so the design is sound, but 14% of categories
still reject at the 1% level and 31 of 56 fail at least one placebo. Model comparisons
survive this; individual elasticity levels in the failing categories do not.

*Two terms need care the paper's text does not dwell on.* The bilinear price term
`gamma_i . lambda_j` sits at zero forever unless its factors are seeded away from zero
(the C++ flags for this exist but are easy to miss), and the inclusive value has to be
centred or its coefficient comes out with the wrong sign. Either mistake yields a model
that fits well and has no usable price response.

*The sample is half the paper's, spread over 561 stores instead of one*, which costs
half the categories, forces a smaller latent dimension, and makes chain-level pricing
an approximation rather than a fact.

