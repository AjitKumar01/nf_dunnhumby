# Data exploration: dunnhumby "The Complete Journey"

What 2,500 households actually did over two years of grocery shopping, measured
directly from the transaction file. This document describes the data. It does not
argue for or against any model, and nothing in it depends on one having been fitted.

**Source.** `transaction_data.csv`: 2,553,406 purchase lines, 2,500 households, 91,856
products, 253,183 baskets, 711 days, 307 commodities, 2,373 sub-commodities. Prices
are reconstructed from `SALES_VALUE` and the three discount columns; that
reconstruction is audited separately and is not repeated here.

Where a figure is quoted for a filtered catalogue rather than the raw file, the filter
is stated. The working catalogue used from §4 onward is the 5,455 items with at least
100 purchase lines — 65.7% of volume — which is the level at which per-item quantities
are estimable.

**Scripts.** §§1–5 `21_basket_eda.py`, §6 `29_demand_eda.py`, §7 `30_household_eda.py`,
§8 `25_basket_placebo.py`. Every number and figure comes from one of them; none is
typed by hand.

### What is measured here

| question | section |
|---|---|
| how many items does a basket hold, and how many from one category? | §1 |
| which products get bought together, and which never are? | §2 |
| how long between repeat purchases, and does that predict the next one? | §3 |
| how many products have enough purchases to analyse? | §4 |
| how much do prices move? | §5 |
| **does demand respond to price, and through which margin?** | **§6** |
| **do households differ — in taste, in price response, in where they shop?** | **§7** |
| **is the price variation exogenous?** | **§8** |

## 1. What a basket contains

![unit demand](figures/basket_eda_unit_demand.png)

A basket is not one item, and it is often not one item per category either.

| measurement | value |
|---|---|
| baskets containing **more than one item from a single category** | **56.1%** |
| baskets spanning 2 or more categories | 81.5% |
| baskets spanning 5 or more categories | 48.4% |
| mean distinct items per basket | 10.1 |
| median distinct items per basket | 5 |
| categories where >15% of category-trips buy 2+ items | **132 of 307** |
| median category's multi-item share | 13.1% |

Read the middle panel carefully. Buying two or more items from the same category is
not a fringe behaviour concentrated in a few odd categories — the **median category
does it on 13.1% of its category-trips**, and the distribution is smooth. There is no
natural cut point separating "single-item categories" from the rest; the behaviour is
continuous across the catalogue.

Multi-item category purchases are therefore the norm rather than the exception at the
basket level: **56.1%** of baskets contain at least one.

---

## 2. Which products are bought together

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

So most pairs sit near independence, with a structured tail at both ends: roughly one
pair in nine is a genuine complement at twice chance or better, and one in twenty is
avoided. The bulk is unstructured; the tails are not.

### Within a sub-commodity, shoppers seek variety

Items from the **same sub-commodity** are

> **7.88× more likely to appear in the same basket than chance**
> (observed 3.23% of within-basket pairs, expected 0.41%)

This is the opposite of what a pure substitution story predicts. If items within a
sub-commodity were substitutes, buying one would make buying another *less* likely and
the ratio would be below 1.

Shoppers do not pick one yogurt. They pick three — different flavours, same
sub-commodity. They buy two apple varieties. **Within a sub-commodity the dominant
behaviour is variety-seeking, not substitution.** Substitution shows up between
sub-commodities, not inside them.

---

## 3. Repurchase timing

![state](figures/basket_eda_state.png)

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

**A 4.30× swing**, and the shape matters as much as the size. The hazard *rises* from
0–3 days to a peak at 7–14 days, then decays. That hump is a consumption cycle: you do
not rebuy milk the day after, you rebuy it the week after.

So recency is informative about the next purchase, and informative in a non-monotone
way — the relationship cannot be summarised by a single "days since" number.

---

## 4. How much of the catalogue is analysable

![catalogue](figures/basket_eda_catalogue.png)

How many items have enough purchases to say anything reliable about them?

| min. purchase lines | items | share of volume | sub-commodities | sub-commodities with 2+ items |
|---|---|---|---|---|
| 20 | 18,597 | 89.4% | 1,366 | 1,098 |
| 50 | 10,333 | 79.2% | 1,033 | 802 |
| **100** | **5,455** | **65.7%** | **758** | **545** |
| 200 | 2,259 | 48.5% | 512 | 309 |
| 500 | 589 | 29.0% | 247 | 121 |

The last column counts sub-commodities holding at least two items — the groups within
which any similarity question can be asked at all. A singleton sub-commodity cannot
tell you whether similar things behave similarly.

**The working catalogue is the ≥100-line cut**: 5,455 items, 65.7% of volume, 758
sub-commodities of which 545 are testable. The 20-line threshold covers more volume
(89.4%) but an item seen 20 times across 2,066 households supports very little
per-item estimation. The 100-line cut trades a quarter of the volume for quantities
that can actually be measured.

---

## 5. How much prices move

Prices move often enough to study. For the 5,455-item catalogue:

| | value |
|---|---|
| share of item-weeks where the price moves | **30.5%** |
| median within-item coefficient of variation | **0.136** |

The daily price panel is 5,455 × 712 with **24.7% of item-days directly observed**,
the rest carried forward from the last observed day — the cost of working at daily
rather than weekly resolution.

Two price series are distinguished throughout: `unit_price`, the loyalty (card-holder)
price a shopper actually faces, and `base_price`, the regular posted price. Price
*movement* is measured within item, so an item being expensive in general is separated
from its price changing.

---

## 6. Does demand actually respond to price?

The central question for a dataset built around prices. Measured on the item × week
panel by `scripts/29_demand_eda.py`.

### 6.1 The demand curve

![demand response](figures/demand_eda_price.png)

*Left panel.* Week-on-week change in log price against week-on-week change in log
buyers per trip, within item, binned into twelve quantiles. Monotone across the whole
range and passing through the origin.

Within-item log-log slopes on 556,410 item-weeks:

| response | elasticity |
|---|---|
| **buyers per trip** | **−0.792** |
| units per trip | −0.945 |
| **units per buyer** | **−0.235** |

The third line separates two channels that the first two combine. A price cut brings
in more buyers *and* makes each buyer take more, and the second channel is **25% of
the total units response**. Whether demand is counted in buyers or in units therefore
changes the answer by a quarter.

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

The tails are face-valid, which is a useful check on the measurement itself: frozen
ice and corn are seasonal commodities bought largely on price, while greeting cards
and magazines are impulse purchases where price is close to irrelevant.

### 6.3 Promotions, as a raw event study

*Right panel.* Every item-week where price fell by at least 0.15 in logs — **37,132
events** — lined up and averaged, with demand expressed relative to that item's own
mean.

| weeks from the cut | −3 | −2 | −1 | **0** | +1 | +2 | +3 |
|---|---|---|---|---|---|---|---|
| demand ÷ item mean | 1.00 | 0.95 | 0.96 | **1.94** | 1.28 | 1.17 | 1.15 |

**Demand roughly doubles in the week of the cut** (2.02× the week before), then decays
over about three weeks without returning to baseline.

The pre-period is flat, which matters: demand is not already rising in the weeks
before a price cut. Had it been, the spike could have reflected promotions being timed
to demand rather than causing it. §8 tests that formally.

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

Breadth and quantity are different quantities and move separately. Three different
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

So a promotion changes *what* a shopper takes from a category, not only whether they
visit it and how many units they take. Breadth, incidence and depth are three
distinguishable responses to the same price change.

### 6.5 Base rates every model head has to reproduce

| event | rate |
|---|---|
| **category incidence** | **6.12 of 188 = 3.25%** |
| item purchase | 7.86 of 5,455 = 0.144% |
| units per basket | 10.64 |

These are the rates any statement about the data has to be consistent with. A
household buys from **3.25%** of categories on a given trip and **0.144%** of the
catalogue — both small numbers, and both easy to get wrong by an order of magnitude if
they are assumed rather than measured.

---

## 7. Households: taste, price sensitivity, store visits, trips

Do households differ from one another in ways that persist, or does aggregate
behaviour describe everyone? Two questions: do they like different things, and do they
react differently to price. Measured by `scripts/30_household_eda.py`.

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
the household: knowing what a household bought last year tells you much more about
what it buys this year than knowing what an average household buys.

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

The spread alone proves nothing: per-household estimates are noisy, and noisy
estimates are spread out even when the underlying quantity is identical for everyone.
The **split-half correlation** is the test that separates the two — a household's
price response in the first half of its trips against its response in the second half.
It is modest but clearly non-zero, on
1,374 households.

So price sensitivity genuinely differs across households — but only about a quarter of
the observed spread is stable, and the rest is estimation noise. That ratio is the
ceiling on how well any method, model or otherwise, could sort households by price
sensitivity from this data.

### 7.3 Store visits

![stores and trips](figures/household_eda_stores_trips.png)

| | value |
|---|---|
| median trips per household | 71 |
| median distinct stores | **4** (p90 9) |
| trips at the primary store | median **76%**, p10 42% |
| consecutive trips that switch store | **30.1%** |
| households using only one store | 7.6% |

Households are **loyal but not monogamous**: the typical one does 76% of its trips at
a primary store, yet 30% of consecutive trips switch, and only 7.6% ever use a single
store.

So a household is not well described by one store. And because the same household
shops several stores that differ in price and assortment, store differences can be
observed *within* a household rather than only by comparing different households —
which is the cleaner comparison, since it holds taste fixed.

### 7.4 Trip rhythm

| | value |
|---|---|
| median items per trip | 4 |
| median gap between trips | 3 days (p90 14) |
| correlation, gap vs trip size | **+0.070** |
| gap before a large trip | 5 days |
| gap before a small trip | 3 days |

A weak but consistent stock-up pattern: longer gaps precede bigger trips. The
correlation is +0.07 — real but small, so trips do not separate cleanly into distinct
"top-up" and "stock-up" types; the variation is continuous.

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

The largest spread across any demographic attribute is **0.076**, against a p10–p90
spread of **0.411** across households. **Demographics explain almost none of the
variation in price sensitivity** — under a fifth of it.

Two households in the same demographic cell differ from each other about as much as
two households drawn at random. Whatever drives price sensitivity here, it is not the
recorded observables.

---

## 8. Is the price variation exogenous, and how much of it is the calendar?

## 9. Summary of findings

| finding | section |
|---|---|
| 56.1% of baskets hold 2+ items from one category; the median category does it on 13.1% of its category-trips | §1 |
| ~11% of sub-commodity pairs are complements at 2×+ chance, ~4.5% are avoided | §2 |
| items in the same sub-commodity co-occur **7.88×** more than chance — variety-seeking, not substitution | §2 |
| repurchase hazard swings **4.30×** with recency and is non-monotone, peaking at 7–14 days | §3 |
| 5,455 items have ≥100 purchases, covering 65.7% of volume and 545 testable sub-commodity groups | §4 |
| prices move on **30.5%** of item-weeks | §5 |
| demand is monotone in price: **−0.79** on buyers, **−0.95** on units | §6.1 |
| **25%** of the units response is units-per-buyer rather than more buyers | §6.1 |
| elasticity by category: median **−0.95**, p10 −1.57, p90 −0.04, 91% negative | §6.2 |
| a price cut **doubles demand** that week, decaying over ~3 weeks; flat beforehand | §6.3 |
| a category purchase spans **1.284** distinct items; breadth elasticity **−0.069** | §6.4b |
| base rates: category incidence **3.25%**, item purchase **0.144%** | §6.5 |
| taste is **1.67×** more self-similar within household than across; 95.9% of households | §7.1 |
| price sensitivity differs across households, split-half correlation **+0.24** | §7.2 |
| households use a median of **4 stores**; 30% of consecutive trips switch | §7.3 |
| demographics span **0.076** of price slope against a **0.411** household spread | §7.5 |
| strict price placebos retain **0.7%** of the real effect; 5 of 160 categories fail | §8 |
| **11.3%** of the raw price–demand association is week-frequency seasonality | §8 |

The working sample these are measured on, built by `scripts/22_basket_data.py`:

| | value |
|---|---|
| households | 2,066 |
| items | 5,455 |
| categories | 188 |
| sub-commodities | 758 |
| (basket, item) rows | 1,566,063 |
| baskets | 199,347 |
| days | 712 |
| stores | 115 |

## 10. Scope: what this document does *not* establish

- **Causality.** §8 shows the price variation survives four placebos. That is evidence
  about the *design*, not proof of exogeneity: a placebo rules out the confounders it
  destroys, and cannot rule out one that moves with price at item × week frequency and
  survives reordering. Promotions timed to anticipated demand are exactly such a
  confounder, and the 11.3% seasonal component is the visible part of it.
- **Store-level prices are sparse.** Only 2.3% of the item × store × week grid is ever
  observed, so most store-level price statements rest on thin data. Chain-level
  figures are the reliable ones.
- **Availability is a proxy.** dunnhumby has no stock-out feed. "This store carries
  this item" is inferred from having sold it, which cannot distinguish a temporary
  stock-out from a deliberate delisting.
- **Weighed goods.** `QUANTITY` is unreliable for items sold by weight, so unit counts
  are capped at 12 and the far tail of bulk lines is not analysed.
- **Panel, not population.** 2,500 loyalty-card households of one retailer over two
  years. Nothing here generalises to non-cardholders, other retailers or other periods.
- **No causal claim about anything except price.** Seasonality, store and household
  differences are described as they appear in the data; none is identified against a
  counterfactual.
