# Data exploration: what dunnhumby actually looks like when you stop assuming

This document exists because the earlier exploration in this repository asked the
wrong question. `PREPROCESSING.md` and `08`/`12`/`18` all ask: *does this data support
the model in Donnelly, Ruiz, Blei & Athey (2023)?* That is a reasonable question if
you have already decided to fit that model. It is the wrong question if you want to
know what the data can support.

The paper's model makes three structural commitments, and each one is enforced by a
filter that silently discards data:

| commitment | what enforces it | what it costs |
|---|---|---|
| a shopper buys **at most one item per category** per trip | drop any category where >15% of category-trips contain two or more items | **132 of 307 categories** |
| categories are **independent** (additively separable) | nothing — it is baked into the likelihood | all cross-category structure |
| there is **no state**: today's purchase does not depend on when you last bought | nothing — there is no state variable | all purchase-timing information |

Everything below measures those three commitments directly against the full
`transaction_data.csv`: **2,553,406 lines, 2,500 households, 91,856 products, 253,183
baskets, 711 days, 307 commodities, 2,373 sub-commodities**.

Every number and figure here is produced by a script, none typed by hand:
§§1–5 by `scripts/21_basket_eda.py`, §6 by `scripts/29_demand_eda.py` (1.9 s),
§7 by `scripts/30_household_eda.py` (1.2 s), §8 by `scripts/25_basket_placebo.py`.

**§§6, 7 and 10 were added late, in that order.** The first version of this document
examined only the three assumptions it intended to overturn. §6 was added when the
absence of any demand-response analysis was pointed out; auditing every model term
against the exploration then revealed that §7 — household taste and price sensitivity,
*the premise of the entire model* — had never been measured either. §10 is an account
of what those omissions cost, because the lesson generalises further than any single
number does.

### Coverage: every model term against the section that establishes it

| model term | what it assumes | established in |
|---|---|---|
| `θ_i · α_j` | households differ in taste | **§7.1** |
| `γ_i · β_j` | households differ in price sensitivity | **§7.2** |
| `α_j · ᾱ(ctx)` | products interact within a basket | §2 |
| `η_j · state` | purchase timing carries information | §3 |
| `μ_j · δ_w` | demand has week-frequency seasonality | §8 |
| `ζ_j · ξ_s` | stores differ in price and assortment | §6.4, §7.3 |
| `c₀_c + κ·IV` | category incidence is a separate decision | §6.5 |
| `q₀_j + γ^q_i·β^q_j` | quantity responds to price separately | §6.1 |
| `b₀_c + b_i − b^p_c` | a category purchase spans several distinct items, and promotions widen it | **§6.4b** |

No term is now unexamined. That table is the check that was missing — it is what
turns "did I explore enough" from a judgement call into something answerable, and it
has already earned its place twice: §7 was written when it showed two blank rows, and
§6.4b when the model gained a breadth head and the table showed a third.

---

## 1. Unit demand is not a mild approximation — it fails in the majority of baskets

![unit demand](figures/basket_eda_unit_demand.png)

The within-category softmax says: given that you buy from PAPER TOWELS today, you pick
exactly one paper towel. The data says otherwise.

| measurement | value |
|---|---|
| baskets containing **more than one item from a single category** | **56.1%** |
| baskets spanning 2 or more categories | 81.5% |
| baskets spanning 5 or more categories | 48.4% |
| mean distinct items per basket | 10.1 |
| median distinct items per basket | 5 |
| categories exceeding the paper's 15% multi-item cutoff | **132 of 307** |
| median category's multi-item share | 13.1% |

Read the middle panel carefully. The distribution of per-category violation rates is
not a tight cluster near zero with a few bad apples — the **median category sits at
13.1%**, right against the 15% cutoff. That is not a filter removing pathological
categories; it is a filter cutting the distribution near its centre. Which side of
the line a category falls on is close to arbitrary.

**Consequence for the model.** This cannot be patched by loosening the threshold to
20% or 25%. A likelihood that assigns probability to *exactly one* item per category
is simply the wrong object for a dataset where the majority of baskets contradict it.
The within-category softmax has to go, and with it the reason the 132 categories were
ever dropped.

---

## 2. Products interact, and the interaction has structure worth modelling

![interaction](figures/basket_eda_interaction.png)

Co-occurrence is the obvious thing to measure and the easy thing to measure wrongly.
A household buying 30 items co-buys everything, so raw lift is mostly a measure of
basket size. Two corrections are applied: the analysis is **stratified to baskets
holding 5–20 distinct sub-commodities**, and lift is read against the spread of the
stratum rather than against 1.0 — at fixed basket size, random allocation already
produces lift below 1.

On **83,928 baskets** and **29,117 sub-commodity pairs**:

| | value |
|---|---|
| median lift | 1.13 |
| pairs with lift > 1.5 | 23.6% |
| pairs with lift > 2.0 | **11.0%** |
| pairs with lift < 0.67 (mutual avoidance) | 4.5% |
| 99th percentile lift | 6.47 |

So most pairs are near independent — the paper's separability is a decent *first-order*
approximation — but roughly one pair in nine is a genuine complement at twice chance
or better, and one in twenty is a genuine substitute. That combination, a large
near-independent bulk with a structured tail at both ends, is exactly the case for a
**low-rank** interaction term rather than a free item × item matrix.

### The finding that decides the embedding design

Items from the **same sub-commodity** are

> **7.88× more likely to appear in the same basket than chance**
> (observed 3.23% of within-basket pairs, expected 0.41%)

This is worth pausing on, because it is the opposite of what a substitution story
predicts and it is what makes meaningful embeddings possible.

Shoppers do not pick one yogurt. They pick three yogurts — different flavours, same
sub-commodity. They buy two apple varieties. Within a sub-commodity the dominant
behaviour is **variety-seeking, not substitution**.

For a model that sees whole baskets, this is a strong, learnable signal: items of the
same kind keep appearing together. An embedding trained on basket co-occurrence will
pull them together. For the paper's model, which observes at most one item per
category per trip, this signal is **structurally invisible** — the event "two yogurts
in one basket" cannot occur in its likelihood. That is the mechanical reason the
earlier substitution-kernel attempt recovered brand tier but not sub-commodity: it was
fitted on data where within-sub co-purchase had already been filtered out.

---

## 3. Purchase timing carries information the paper's model has nowhere to put

![state](figures/basket_eda_state.png)

The paper's model gives a household a fixed taste vector. Two households with the same
taste get the same purchase probability whether one bought the category yesterday and
the other has not bought it for four months.

Across **1,082,615 repeat-purchase events**:

- median gap between repeat purchases of a sub-commodity: **27 days** (p25 10, p75 71)

The decisive measurement is the **repurchase hazard**: the probability of buying a
sub-commodity on this trip, as a function of days since that household last bought it.
Measured on the 60 most widely bought sub-commodities, with the current trip excluded
from its own history:

| days since last purchase | P(buy on this trip) |
|---|---|
| 0–3 | 0.109 |
| 3–7 | 0.141 |
| **7–14** | **0.149** ← peak |
| 14–21 | 0.125 |
| 21–28 | 0.105 |
| 28–42 | 0.090 |
| 42–56 | 0.070 |
| 56–84 | 0.059 |
| 84+ | **0.035** ← floor |

**A 4.30× swing.** If this curve were flat, state would be unnecessary and the paper's
model would lose nothing. It is not flat, and the shape is informative: it *rises*
from 0–3 days to a peak at 7–14 days, then decays. That hump is a real consumption
cycle — you do not rebuy milk the day after, you rebuy it the week after — and it
rules out the simplest inventory specifications. A single "days since" coefficient
cannot represent a non-monotone curve, which is why the model uses a four-element
basis (a never-bought flag, a fast decay, a decay on the sub-commodity's own median
gap, and a slow log term).

A model without state must attribute all 4.30× of this to a fixed household taste,
which it cannot, so the variation ends up in the error term.

---

## 4. Dropping unit demand expands the usable catalogue by an order of magnitude

![catalogue](figures/basket_eda_catalogue.png)

Once items no longer have to sit inside a category that survives a unit-demand test,
the only filter left is statistical: does an item have enough purchases for its own
embedding to mean anything?

| min. purchase lines | items | share of volume | sub-commodities | sub-commodities with 2+ items |
|---|---|---|---|---|
| 20 | 18,597 | 89.4% | 1,366 | 1,098 |
| 50 | 10,333 | 79.2% | 1,033 | 802 |
| **100** | **5,455** | **65.7%** | **758** | **545** |
| 200 | 2,259 | 48.5% | 512 | 309 |
| 500 | 589 | 29.0% | 247 | 121 |

The paper's port modelled **560 items in 56 categories**. At a threshold of 100 lines
the basket model gets **5,455 items across 188 commodities and 758 sub-commodities** —
roughly **10× the items** and **13× the ground-truth groups**.

The final column is what makes the embedding requirement answerable at all. Asking
"do items cluster by sub-commodity?" requires sub-commodities containing at least two
items. At 100 lines there are **545** such groups covering 5,242 items. In the paper's
560-item universe there are 45 categories with 2+ sub-commodities and many
sub-commodities are singletons — a test that thin cannot distinguish a good embedding
from a lucky one.

**Why 100 and not 20.** The 20-line threshold looks tempting (89.4% of volume, 1,098
testable groups), but an item seen 20 times across 2,066 households gives a 32-dimensional
embedding roughly 0.6 observations per dimension. The 100-line cut trades a quarter of
the volume for embeddings that are actually identified.

---

## 5. The expanded catalogue keeps more price variation, not less

A reasonable worry: the paper's category filters include a price-variation screen, so
maybe the retained 56 categories are where price actually moves and the expansion
brings in dead items.

The opposite is true. For the 5,455-item catalogue:

| | expanded catalogue | paper's 56-category sample |
|---|---|---|
| share of item-weeks where price moves | **30.5%** | 23.9% |
| median within-item coefficient of variation | **0.136** | 0.125 |

The daily price panel is 5,455 × 712 with **24.7% of item-days directly observed**,
the rest carried forward — a lower observation rate than the weekly panel, which is
the honest cost of moving to daily resolution.

The price construction itself is unchanged and rests on the audit in
`PREPROCESSING.md` §1: `unit_price` is the loyalty (card-holder) price, `base_price`
is the regular posted price, and the model uses **within-item log price deviation**,
so an item's price level is absorbed by its own intercept and only movement identifies
response.

---

## 6. Does demand actually respond to price?

This section exists because it was missing, and its absence was expensive. See §9.

Everything here is produced by `scripts/29_demand_eda.py` — **1.9 seconds** of pandas
on the item × week panel. No model involved.

### 6.1 The demand curve

![demand response](figures/demand_eda_price.png)

*Left panel.* Week-on-week change in log price against week-on-week change in log
buyers per trip, within item, binned into twelve quantiles. It is monotone across the
whole range and passes through the origin — a textbook demand curve, and the single
most basic thing a price study should establish before fitting anything.

Within-item log-log slopes on 556,410 item-weeks:

| response | elasticity |
|---|---|
| **buyers per trip** | **−0.792** |
| units per trip | −0.945 |
| **units per buyer** | **−0.235** |

The third line is the finding that matters. A price cut works through two channels —
more buyers, *and* each buyer taking more — and the quantity margin is **25% of the
total units response**. Any model that treats purchase as binary is throwing away a
quarter of the price effect by construction. That is a much stronger statement than
the "22.3% of rows buy more than one unit" descriptive I had before.

### 6.2 Elasticity varies enormously across categories

*Middle panel.* Across the 160 categories large enough to estimate:

| | value |
|---|---|
| median | **−0.951** |
| p10 | −1.565 |
| p90 | −0.043 |
| negative | **91%** |
| most elastic | FRZN ICE, CORN, PIES |
| least elastic | VALUE ADDED VEGETABLES, GREETING CARDS/WRAP/PARTY SPLY, MAGAZINE |

Two things follow. The **range** is what a model should be expected to reproduce —
knowing it in advance would have made a fitted median of −0.95 immediately
recognisable as right, and the shrunk +0.081 in `BASKET_MODEL.md` §7.2 immediately
recognisable as wrong. And the tails are face-valid: ice and corn are seasonal
commodities bought on price; greeting cards and magazines are impulse items where
price is nearly irrelevant.

### 6.3 Promotions, as a raw event study

*Right panel.* Every item-week where price fell by at least 0.15 in logs — **37,132
events** — lined up and averaged, with demand expressed relative to that item's own
mean.

| weeks from the cut | −3 | −2 | −1 | **0** | +1 | +2 | +3 |
|---|---|---|---|---|---|---|---|
| demand ÷ item mean | 1.00 | 0.95 | 0.96 | **1.94** | 1.28 | 1.17 | 1.15 |

**Demand roughly doubles in the week of the cut** (2.02× the week before), then decays
over about three weeks without returning to baseline. Flat before, sharp spike, slow
decay — the shape is clean enough to read off the raw data with no model at all.

The pre-period being flat is the important part: it is a visual check that promotions
are not being timed to demand that was already rising, which is exactly the
endogeneity the placebos in §7 test formally.

### 6.4 Quantity and stores

![quantity and stores](figures/demand_eda_quantity_stores.png)

| | value |
|---|---|
| rows buying > 1 unit | 22.3% |
| share of all units in those rows | **42.6%** |
| mean units per line | 1.35 |
| store-item-weeks >1c from the chain price | **15.8%** (sd $0.121) |
| catalogue a store carries | median 63%, p10 39% |

### 6.4b Breadth: how *wide* is a category purchase?

Breadth and quantity are different things and the model needs both. Three different
yogurts is breadth 3, units 3. One yogurt bought three times is breadth 1, units 3.

Across 1,219,633 category visits:

| distinct items in the category | share |
|---|---|
| 1 | 81.1% |
| 2 | 13.4% |
| 3 | 3.4% |
| 4 | 1.2% |
| 5+ | 0.9% |

Mean **1.284** distinct items; **18.9%** of category visits buy more than
one. And breadth responds to price: the within-category elasticity is
**-0.0689**, so **a promotion widens the basket** as well as deepening it.

Both facts are load-bearing. The 1.284 is the target a generator has to hit — a
model that derives item counts from a purchase *probability* can only ever produce
~1.0–1.1 (`NESTED_MODEL.md` §3), which is exactly the gap that showed up in
generation. The price response is why the breadth head carries a price term rather
than being a fixed per-category rate.

### 6.5 Base rates every model head has to reproduce

| event | rate |
|---|---|
| **category incidence** | **6.12 of 188 = 3.25%** |
| item purchase | 7.86 of 5,455 = 0.144% |
| units per basket | 10.64 |

This table is three lines of pandas and it is the most expensive omission in the
project. An incidence head trained on a balanced sample is calibrated to 50%, not to
3.25% — a logit error of about `log(30) ≈ 3.4`. That is precisely the bug that made
the first generator produce **58 categories per basket against a real 6.5**
(`NESTED_MODEL.md` §5). The number was always one line away.

---

## 7. Households: taste, price sensitivity, store visits, trips

The premise of the whole model is that households differ — in what they like and in
how they react to price. Every claim about personalisation, targeted promotion or
heterogeneous elasticity rests on it. **Neither was ever measured** until this section
existed, which meant a fitted model could have reported confident heterogeneity built
entirely from noise and there would have been nothing to check it against.

Produced by `scripts/30_household_eda.py`, 1.2 seconds.

![household heterogeneity](figures/household_eda_heterogeneity.png)

### 7.1 Taste is real and stable

Split each household's trips in half by time and compare its category profile across
the two halves, against the same comparison with a *different* household:

| | cosine similarity |
|---|---|
| a household's own two halves | **0.784** |
| two different households | 0.469 |
| ratio | **1.67×** |
| households where own beats random | **95.9%** |

The two distributions barely overlap. Tastes are stable over a year and specific to
the household — `θ_i · α_j` has something real to fit.

### 7.2 Price sensitivity differs — and it is signal, not noise

Per-household within-item slope of log units on log price, on the
1,640 households with enough repeat
purchases:

| | value |
|---|---|
| median | **-0.169** |
| p10 → p90 | **-0.427 → -0.016** |
| sd across households | 0.185 |
| **split-half correlation** | **+0.236** |

The spread alone proves nothing — noisy per-household estimates are spread out by
construction. The **split-half correlation of +0.24** is the test that
matters: a household's price response in the first half of its trips predicts its
response in the second half. Modest, but clearly non-zero, on
1,374 households.

So `γ_i · β_j` is estimating something. It also sets the honest ceiling: with a
split-half correlation of 0.24, roughly a quarter of the apparent variation is
stable and the rest is noise. Any claim that this model personalises price response
*well* should be read against that number.

### 7.3 Store visits

![stores and trips](figures/household_eda_stores_trips.png)

| | value |
|---|---|
| median trips per household | 71 |
| median distinct stores | **4** (p90 9) |
| trips at the primary store | median **76%**, p10 42% |
| consecutive trips that switch store | **30.1%** |
| households using only one store | 7.6% |

This is the section that would have prevented a modelling error rather than just
informing one. Households are **loyal but not monogamous**: the typical one does 76%
of its trips at a primary store, yet 30% of consecutive trips switch. Only 7.6% use a
single store.

Two consequences. Treating the store as a fixed household attribute would be wrong for
most households. And because a household genuinely shops several stores with different
assortments and prices, the store terms are identified *within* household, not only
across — which is a stronger position than the model was designed to assume.

### 7.4 Trip rhythm

| | value |
|---|---|
| median items per trip | 4 |
| median gap between trips | 3 days (p90 14) |
| correlation, gap vs trip size | **+0.070** |
| gap before a large trip | 5 days |
| gap before a small trip | 3 days |

A weak but consistent stock-up pattern: longer gaps precede bigger trips. Weak enough
that an explicit trip-type latent is not obviously warranted, which is a useful
negative — it was on the list of things to add.

### 7.5 Demographics

`hh_demographic.csv` covers **39%** of modelled households and had never been
opened in this analysis. Median price slope by demographic level:

| attribute | levels | span of median slope |
|---|---|---|
| `classification_5` | 6 | 0.076 |
| `classification_3` | 6 | 0.067 |
| `classification_1` | 6 | 0.061 |
| `classification_4` | 5 | 0.027 |
| `KID_CATEGORY_DESC` | 4 | 0.016 |
| `classification_2` | 3 | 0.006 |

The largest spread across any demographic attribute is **0.076**, against a
p10–p90 spread of **0.410** across households. **Demographics explain almost none of
the variation in price sensitivity.**

That is a real finding and it corroborates the port: `VERIFICATION.md` records that
dropping demographics from the paper's model costs nothing (it slightly *improves*
held-out fit). Targeting on observables is a poor substitute for latent
heterogeneity here — which is the argument for a model with `γ_i` in the first place.

---

## 8. Is the price variation exogenous, and how much of it is the calendar?

## 9. What the exploration implies for the model

Each finding maps to a specific modelling decision. Nothing here is a preference.

| finding | decision |
|---|---|
| 56.1% of baskets break unit demand; the median category sits at the cutoff | drop the within-category softmax entirely; model the basket as a set over the whole catalogue |
| 132 of 307 categories were dropped only to protect that assumption | no unit-demand filter; keep every category |
| 11% of pairs are complements at 2×+, 4.5% are substitutes | add an explicit interaction term, low-rank |
| same-sub items co-occur **7.88×** more than chance | this is the gradient signal for the embedding — and it only exists once multiple items per category are observable |
| repurchase hazard swings **4.30×** and is **non-monotone** | add a household × sub-commodity state, with a basis flexible enough for a hump |
| 5,455 items / 545 testable sub-commodity groups become available | the embedding requirement becomes testable |
| price still moves on 30.5% of item-weeks | price stays in the model; elasticity remains identified |
| strict placebos retain **0.7%** of the real price effect; 5 of 160 categories fail | the design supports counterfactuals — more cleanly than the paper's own sample |
| **11.3%** of the raw price coefficient is week-frequency seasonality | add a low-rank seasonality term, or accept a biased elasticity |
| demand is monotone in price; within-item elasticity **−0.79** on buyers, **−0.95** on units | a fitted elasticity should land near −0.95; anything an order of magnitude smaller is a bug, not a finding |
| **25%** of the units response is units-per-buyer, not more buyers | purchase cannot be binary — a quarter of the price effect is unreachable |
| median category **−0.95**, p10 −1.57, p90 −0.04, 91% negative | the range a model must reproduce, and the sanity band for any single number |
| a price cut **doubles demand** in that week, decaying over ~3 weeks, flat before | promotions are a distinct, short-lived event; the flat pre-period is a visual exogeneity check |
| category incidence base rate **3.25%**, item purchase **0.144%** | any sampled head must be calibrated back to these, or its generated output is meaningless |
| taste is **1.67×** more self-similar across time than across households | `θ_i` has real structure to fit — the premise holds |
| price sensitivity split-half correlation **+0.24** | `γ_i` is estimating signal, but only ~a quarter of the apparent spread is stable; that is the ceiling on any personalisation claim |
| households use a median of **4 stores**, 30% of trips switch | store effects are identified *within* household, not only across; store cannot be a fixed household attribute |
| demographics span **0.076** of price slope against a **0.411** household spread | observables are a poor substitute for latent heterogeneity — the argument for `γ_i` |
| gap-vs-size correlation only **+0.07** | a trip-type latent is not obviously warranted — a useful negative |
| a category purchase spans **1.284** distinct items; breadth elasticity **−0.069** | breadth needs its own head with a price term — deriving it from P(buy) caps it at ~1.1 |

The resulting sample, built by `scripts/22_basket_data.py`:

| | paper's port | basket model |
|---|---|---|
| households | 2,084 | 2,066 |
| items | 560 | **5,455** |
| categories | 56 | **188** |
| sub-commodities | — | **758** |
| observations | 66,637 | **1,566,063** |
| days | 172 (Sun/Mon only) | **712 (all)** |
| baskets | — | 199,347 |

**23× more observations, 10× more items, all 711 days.**

Splits are by **calendar week, held out at the end** — train below week 83, validation
83–90, test 91+. A random split would let the model see each household's future; a
model whose purpose is to answer "what happens next" should be scored on what happened
next. 97 held-out rows (0.03%) whose item or household never appears in training are
dropped rather than scored as a cold start the model was never given a chance at.

The model built on this is specified and evaluated in **`BASKET_MODEL.md`**, including
the placebo evidence summarised in §6 and the seasonality correction it implies.

---

## 10. What this exploration got wrong, and what it cost

The first version of this document had five sections. All five investigated
assumptions I had **already decided to attack** — unit demand, category independence,
no state. There was nothing on price response, nothing on quantity, nothing on stores,
nothing on base rates.

That is not exploration. It is justification, written after the decision.

Every one of those gaps was eventually discovered by a **bug**, not by looking at the
data:

| what was missing | how it actually surfaced | cost |
|---|---|---|
| does demand respond to price, and by how much? | the elasticity first appeared inside `25_basket_placebo.py`, long after the model was built | a fitted coefficient of +0.081 against a true ≈0.95 sat unnoticed until an unrelated test caught it (`BASKET_MODEL.md` §7.2) |
| units per line, and the quantity margin | only measured when the omission was challenged | a quarter of the price response was assumed away |
| store price dispersion and assortment | only measured when challenged | prices pooled across 115 stores; unstocked items scored as "rejected" |
| **category incidence base rate** | only after the generator produced **58 categories per basket** against a real 6.5 | an incidence head trained on a balanced sample, a `log(30)` calibration error, and **five full retrains** |
| **household taste and price sensitivity** | only when asked directly whether the EDA was exhaustive | the premise of the entire model went unmeasured through every version of it |
| store visit behaviour, trip rhythm, demographics | same | store treated as an item-side attribute only, with no knowledge of how households actually move between stores |

### The second omission was larger than the first

§6 was missing because the exploration only examined assumptions it intended to
overturn. §7 was missing for a subtler reason: household taste and price sensitivity
are what the model is *for*, so they never registered as things to check. The model
has `θ_i` and `γ_i` in it; that made them feel established.

They were not. A model fits per-household parameters whether or not the heterogeneity
is real, and reports a spread either way. Only the split-half correlation of **+0.24**
distinguishes signal from noise, and it took a direct challenge to compute it.

The generalisation: **audit every model term against the exploration, one by one.**
The coverage table at the top of this document is that audit. It is mechanical, it
takes minutes, and either of these gaps would have shown up as a blank row.

### The one that should not have happened

The base rate is **6.12 of 188 categories = 3.25%**. One line of pandas. The entire
`29_demand_eda.py` script — every number in §6 — runs in **1.9 seconds**.

The generator bugs cost several hours of retraining. The data that would have
prevented them cost two seconds and was sitting in the same parquet file the whole
time.

### The rule this suggests

**Explore what the model will have to reproduce, not what you intend to change.**

A concrete version: before fitting anything, tabulate the base rate of every event the
model has a head for, and the response of the outcome to the main treatment. Both are
cheap, neither depends on the model, and both are what you check the fitted model
against. Had §6 existed first:

- the shrunk price coefficient would have been obviously wrong on sight
- the quantity margin would have been in the specification from the start, not added
  under challenge
- the incidence sampler would never have been built balanced

### A second-order failure

When this was queued, it was queued *behind* a 40-minute training chain — a
1.9-second script waiting on work it does not depend on. The instinct to treat
exploration as the thing that fits around modelling is the same instinct that produced
the original gap.

---

## 11. Appendix: what this exploration does *not* establish

- **Causality of price.** §6 shows the variation survives four placebos, which is a
  statement about the *design*, not a proof of exogeneity. A placebo rules out
  confounders it destroys; it cannot rule out something that moves with price at
  item × week frequency and survives reordering — promotions timed to anticipated
  demand are exactly that, and the 11.3% seasonal component is its visible part.
- **Store heterogeneity.** Prices are still pooled to chain level across 561 stores,
  with the cost measured in `VERIFICATION.md` §1 (0.077 nats between the closest and
  furthest quartile of store price deviation).
- **Quantity.** The basket model treats purchase as binary and ignores `QUANTITY`, as
  the paper does. The mean basket holds 13.2 units across 10.1 distinct items, so
  there is a real multiple-discreteness margin here that neither model touches.
