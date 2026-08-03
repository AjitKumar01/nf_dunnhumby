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

Every number and figure here is produced by `scripts/21_basket_eda.py`. Nothing is
typed in by hand.

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

## 6. What the exploration implies for the model

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

The model built on this is specified and evaluated in **`BASKET_MODEL.md`**.

---

## Appendix: what this exploration does *not* establish

- **Causality of price.** Nothing here re-opens the price-endogeneity question. The
  placebo tests in `PREPROCESSING.md` §9 were run on the paper's sample and found 34
  of 56 categories failing at least one placebo. They have **not** been re-run on the
  expanded catalogue, so every elasticity from the basket model should be read as
  predictive, not causal.
- **Store heterogeneity.** Prices are still pooled to chain level across 561 stores,
  with the cost measured in `VERIFICATION.md` §1 (0.077 nats between the closest and
  furthest quartile of store price deviation).
- **Quantity.** The basket model treats purchase as binary and ignores `QUANTITY`, as
  the paper does. The mean basket holds 13.2 units across 10.1 distinct items, so
  there is a real multiple-discreteness margin here that neither model touches.
