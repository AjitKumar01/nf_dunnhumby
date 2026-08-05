# Data exploration: dunnhumby "The Complete Journey"

What 2,500 households did over two years of grocery shopping, measured from the
transaction file. This document describes the data. It does not argue for or against
any model, and nothing in it depends on one having been fitted.

**Source.** `transaction_data.csv`: **2,553,406** purchase lines, **2,500** households,
**91,856** products, **253,183** baskets, **711** days, **307** commodities, **2,373**
sub-commodities. Prices are reconstructed from `SALES_VALUE` and three discount
columns; that reconstruction is audited separately.

**Working catalogue.** From §4 onward, analysis uses the **5,455** items with at least
100 purchase lines — **65.7%** of all volume. §4 gives the reason and the cost.

**Scripts.** §§1–5 `21_basket_eda.py`, §6 `29_demand_eda.py`, §7 `30_household_eda.py`,
§7.6 `32_reliability.py`, §8 `25_basket_placebo.py`. Every number and figure is
produced by one of them; none is typed by hand.

---

## How to read this document

Each finding is presented in three parts, kept separate on purpose:

| part | what it contains |
|---|---|
| **Measurement** | the unit of observation, the filters applied, and the formula. Enough to recompute the number independently. |
| **Finding** | the number itself, and nothing else. |
| **Reading** | what it does and does not imply. This is interpretation and is labelled as such. |

Two conventions follow from that:

- **Every derived quantity is given with its arithmetic.** Terms like *lift*,
  *elasticity* and *hazard* mean different things depending on what was conditioned on
  and what the reference is, so the formula is stated rather than the name. Where it
  helps, a worked example on real rows is included.
- **Every figure carries a "Reading it" note** giving each axis, what one observation
  represents, and any transform — logs, clipping, binning, normalisation. Several
  panels are not raw data (binned scatters, survival curves, cumulative distributions,
  split-half constructions), and the construction has to be understood before the
  shape means anything.

### What is measured

| question | section |
|---|---|
| How many items does a basket hold, and how many from one category? | §1 |
| Which products are bought together, and which are avoided? | §2 |
| How long between repeat purchases, and does that predict the next one? | §3 |
| How many products have enough purchases to analyse? | §4 |
| How much do prices move? | §5 |
| Does demand respond to price, and through which margin? | §6 |
| Do households differ — in taste, price response, where they shop? | §7 |
| Is the price variation exogenous? | §8 |

---

## 1. What a basket contains

![unit demand](figures/basket_eda_unit_demand.png)

**Measurement.** One observation per basket (`BASKET_ID`). Within each, count
distinct `PRODUCT_ID`s and distinct `COMMODITY_DESC`s. A basket "buys multiple from a
category" when its distinct-item count exceeds its distinct-category count. Whole file,
no item filter.

**Finding.**

| quantity | value |
|---|---|
| baskets containing **more than one item from a single category** | **56.1%** |
| baskets spanning 2 or more categories | 81.5% |
| baskets spanning 5 or more categories | 48.4% |
| mean distinct items per basket | 10.1 |
| median distinct items per basket | 5 |
| categories where >15% of category-trips buy 2+ items | **132 of 307** |
| median category's multi-item share | 13.1% |

**Reading the middle panel.** It starts from a table with **one row per category** —
307 rows, two columns:

| category | category-trips | rate of buying 2+ items |
|---|---|---|
| EGGS | 27,600 | 0.022 |
| FLUID MILK PRODUCTS | 67,700 | 0.215 |
| SOFT DRINKS | 68,000 | 0.415 |
| … | | |

Sort those 307 rows by the last column and walk along it. That is the x-axis: **a
category's own rate of buying more than one item**. Both curves are cumulative — y is
the share *at or below* x — so both rise from 0 to 1 by construction.

The two curves count the same 307 rows differently, and that is the whole point:

- **Grey — each category counts once.** y = "what fraction of *categories* have a rate
  at or below x". At x = 0.13 it reads 0.50, so half of all categories buy multiple
  items on 13% or fewer of their trips.
- **Blue — each category counts by how many category-trips it gets.** y = "what
  fraction of *category-trips* happen in a category with a rate at or below x". At
  x = 0.13 it reads only 0.28.

The gap between them says the rate is **higher in the categories shoppers visit
most**. EGGS and SOFT DRINKS both count once in grey, but SOFT DRINKS is met 2.5×
more often, and it buys multiple items 19× as readily.

Reading the medians off the 50% line: **half of categories sit below 13.1%, but half
of category-trips happen in categories above 21.5%.** The pooled rate over all
category-trips is **23.1%** — which is the number that describes a shopper's
experience, and nearly double the per-category median.

| category | category-trips | buys 2+ items |
|---|---|---|
| SOFT DRINKS | 68,000 | **41.5%** |
| BAG SNACKS | 41,100 | **39.5%** |
| CHEESE | 46,100 | **37.8%** |
| BAKED BREAD/BUNS/ROLLS | 59,300 | 29.6% |
| BEEF | 36,200 | 25.0% |
| FLUID MILK PRODUCTS | 67,700 | 21.5% |
| TROPICAL FRUIT | 32,200 | 5.5% |
| EGGS | 27,600 | 2.2% |

Drinks, snacks, cheese and bread are bought several at a time; eggs and tropical fruit
essentially never are.

**Reading.** Multi-item category purchases concentrate in high-traffic categories
rather than being spread evenly, and at basket level they compound: 56.1% of baskets
contain at least one category bought two or more times.

---

## 2. Which products are bought together

![interaction](figures/basket_eda_interaction.png)

**Measurement.** Lift compares how often two sub-commodities share a basket against
how often they would if unrelated.

Everything is counted inside one **stratum**: baskets holding between 5 and 20
distinct sub-commodities, after items with fewer than 100 purchase lines are dropped.
That leaves **n = 83,928 baskets**. Every count below is a count of baskets in that
set, and a basket either contains a sub-commodity or does not — buying three yogurts
counts once, so this measures co-*occurrence*, not co-volume.

For a pair of sub-commodities `x` and `y`:

```
solo[x]  = baskets containing x                        (out of n)
solo[y]  = baskets containing y
k        = baskets containing BOTH x and y

observed = k / n
expected = (solo[x] / n) × (solo[y] / n)      if x and y were independent
lift     = observed / expected
```

**Worked example — the highest-lift pair in the data:**

| | value |
|---|---|
| x = SQUASH ZUCCHINI | in **700** of the 83,928 baskets |
| y = YELLOW SUMMER SQUASH | in **236** |
| both together | **k = 106** |

```
observed = 106 / 83,928                     = 0.001263
expected = (700/83,928) × (236/83,928)      = 0.0000235
lift     = 0.001263 / 0.0000235             = 53.85
```

So courgettes and yellow squash land in the same basket **54 times more often than
they would if the two were unrelated**. That is the far right edge of the left panel.

**Two filters, and what each is for:**

- **The 5–20 stratum.** A household buying 30 sub-commodities co-buys everything, so
  without this, lift would largely measure basket size. Restricting to a band of
  comparable baskets removes that. The cost is that a pair's lift here is *not* what
  you would get on all 253,183 baskets — every number in this section lives inside
  the stratum.
- **Pairs need k ≥ 30.** A pair seen 3 times together can show a lift of 40 by
  accident. This leaves **29,117 pairs** of the ~287,000 possible.

**Reading the panels.** *Left*: one observation per pair, x = `log2(lift)`, y = how
many pairs. Logs because lift is a ratio — "twice as often" (2) and "half as often"
(0.5) are equally far from independence, but on a raw scale 2 sits 1.0 above 1 and 0.5
only 0.5 below it, which would squash the left half of the distribution. On the log
scale **0 is independence**, +1 is twice chance, −1 is half. Clipped to lift ∈
[0.05, 20] so the 53.85 example above does not stretch the axis.

*Middle and right*: the 12 pairs with the highest and lowest lift, one bar each, x =
lift on a plain linear scale — these are read as individual values, not as a
distribution, so no log is needed.

**Finding.** On 83,928 baskets and 29,117 sub-commodity pairs:

| | value |
|---|---|
| median lift | 1.13 |
| pairs with lift > 1.5 | 23.6% |
| pairs with lift > 2.0 | **11.0%** |
| pairs with lift < 0.67 (mutual avoidance) | 4.5% |
| 99th percentile lift | 6.47 |

**Reading.** Most pairs sit near independence, with a structured tail at both ends:
one pair in nine is a complement at twice chance or better, one in twenty is avoided.
The bulk is unstructured; the tails are not.

### Within a sub-commodity, shoppers seek variety

**Measurement.** Items from the same sub-commodity, compared against a
catalogue-composition reference rather than against the pair lift above.

**Finding.** Items from the same sub-commodity are

> **7.88× more likely to appear in the same basket than chance**

Computed differently from the pair lifts above, so here is that arithmetic too. Take
every basket with 2–30 distinct items and enumerate every **item pair** inside it.
Each pair either shares a sub-commodity or does not:

```
observed = same-sub pairs / all within-basket pairs        = 0.0323
```

For the reference, ask what that share would be if items were drawn at random from the
catalogue. With `c_s` items in sub-commodity `s` and `T` items in total:

```
expected = Σ_s c_s(c_s − 1) / (T(T − 1))                   = 0.0041
ratio    = 0.0323 / 0.0041                                 = 7.88
```

The expected value is small because most sub-commodities hold only a handful of the
5,455 items, so two randomly chosen items rarely share one.

**Reading.** This is the opposite of what a pure substitution story predicts. If items
within a sub-commodity were substitutes, buying one would make buying another *less*
likely and the ratio would fall below 1.

Shoppers do not pick one yogurt. They pick three — different flavours, same
sub-commodity. They buy two apple varieties. **Within a sub-commodity the dominant
behaviour is variety-seeking, not substitution.** Substitution shows up between
sub-commodities, not inside them.

---

## 3. Repurchase timing

![state](figures/basket_eda_state.png)

**Reading it.** *Left*: one observation per repeat-purchase **event** — every time a
household bought a sub-commodity it had bought before. x = days since that household's
previous purchase of it, y = how many events. Clipped at 120 days.

*Middle* is the one that needs care, because it is not a distribution of anything. The
unit is a **(household, trip, sub-commodity) opportunity**: for every trip a household
made, and every sub-commodity it has ever bought, ask "how long since they last bought
this, and did they buy it now?". Group those opportunities by days-since and take the
share that ended in a purchase. So y is a **conditional probability**, not a count, and
the x bins have unequal widths (0–3, 3–7, 7–14, …) so the axis is categorical rather
than linear — the visual spacing is not proportional to elapsed time. The current trip
is excluded from its own history; without that, every point would read 100%.

*Right*: one observation per household, x = how many distinct days it shopped, y = how
many households.

**Measurement.** A repeat-purchase event is any occasion a household bought a
sub-commodity it had bought before; the gap is days since its previous purchase of
that sub-commodity. The hazard construction is set out below.

**Finding.** Across **1,082,615** repeat-purchase events — every occasion a household
bought a sub-commodity it had bought before:

- median gap between repeat purchases of a sub-commodity: **27 days** (p25 10, p75 71)

The decisive measurement is the **repurchase hazard**: the probability of buying a
sub-commodity on this trip, as a function of days since that household last bought it.
Measured on the 60 most widely bought sub-commodities, with the current trip excluded
from its own history:

**How the hazard is computed.** The question is: *given that a household is in the
store today, how likely are they to buy milk — and does that depend on how long it has
been since they last bought milk?*

Answering it needs one row per **opportunity**, not per purchase. An opportunity is a
household walking into a store on a day when it *could* have bought the
sub-commodity. Most opportunities end in no purchase, and those rows are the whole
point — a hazard is purchases ÷ opportunities, so leaving out the misses would make
every value 1.

Here is one real household's trips, for FLUID MILK WHITE ONLY. Every row is a
shopping trip; `bought?` says whether milk was in that basket:

| DAY | bought? | buy_day | ffilled | prev_buy | since |
|---|---|---|---|---|---|
| 42 | yes | 42 | 42 | — | — |
| 49 | yes | 49 | 49 | 42 | **7** |
| 53 | yes | 53 | 53 | 49 | **4** |
| 54 | yes | 54 | 54 | 53 | **1** |
| 58 | yes | 58 | 58 | 54 | **4** |
| 63 | yes | 63 | 63 | 58 | **5** |
| 64 | no | — | 63 | 63 | **1** |
| 66 | no | — | 63 | 63 | **3** |
| 69 | yes | 69 | 69 | 63 | **6** |
| 73 | no | — | 69 | 69 | **4** |
| 82 | no | — | 69 | 69 | **13** |
| 86 | yes | 86 | 86 | 69 | **17** |

Read the four derived columns left to right:

1. **`buy_day`** — the day, but only on rows where milk was actually bought. Blank
   otherwise.
2. **`ffilled`** — forward-fill that column: carry the last non-blank value downward.
   Now every row knows the most recent milk purchase *including today's*. On day 64
   it reads 63, which is right — the last purchase was day 63.
3. **`prev_buy`** — shift `ffilled` down by one row. **This is the step the original
   text failed to explain.** Without it, day 63 would say its last purchase was day
   63 — itself. Every buying row would report 0 days since, and every buying row is
   by definition a purchase, so the 0-day bin would be 100% purchases. The shift
   makes each row look at the state of the world *as it was on arrival*, before
   today's basket existed.
4. **`since`** = `DAY − prev_buy`. Day 49 is 7 days after the day-42 purchase; day 82
   is 13 days after the day-69 one.

The first row has no `prev_buy` and is dropped — there is no prior purchase to measure
from.

Now pool these rows across all buyer households and the 60 most widely bought
sub-commodities, group by `since`, and within each group take the share that bought:

```
hazard(bin) = rows in bin with bought? = yes / all rows in bin
```

From the twelve rows above, the 1-day bin holds day 54 (bought) and day 64 (not), so
it contributes 1 purchase out of 2 opportunities. The 13-day bin holds day 82 alone,
contributing 0 out of 1. Millions of such rows give the curve below.

Two consequences worth being explicit about. Only households that **ever** buy the
sub-commodity are included, so the curve is about repurchase timing among buyers, not
about the population at large. And the bins have unequal widths (0–3, 3–7, 7–14, …),
so on the chart horizontal distance is not proportional to elapsed days.

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

**Reading.** The ratio of the highest bin to the lowest is
`0.149 / 0.035 = 4.30`. The shape matters as much as the size: the hazard *rises* from
0–3 days to a peak at 7–14, then decays. That hump is a consumption cycle — milk is not
rebought the next day but the next week.

Two consequences. Recency carries information about the next purchase; and the
relationship is **non-monotone**, so it cannot be summarised by a single "days since"
coefficient. Any functional form imposed on it needs at least enough freedom to turn
around once.

---

<!-- ## 4. How much of the catalogue is analysable

![catalogue](figures/basket_eda_catalogue.png)

**Reading it.** These two panels have **no unit of observation**, which is what makes
them different from every other figure here — nothing is being counted or averaged
over rows. Neither is a distribution. Both answer "if I demanded at least *x*
purchases per item, how much catalogue would survive?". So x is a **threshold I
choose**, not a measured value, and each point is a fresh re-count of the whole
catalogue at that threshold. Both curves fall by construction, and the left panel is
log–log because both quantities span orders of magnitude.

*Left*: y = items surviving. *Right*: y = sub-commodities surviving. The dashed green
line on each counts only what is *usable* — items sitting in a sub-commodity that
still has at least two members, since a sub-commodity reduced to one item can no
longer support any comparison between similar products.

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

--- -->

## 4. How much of the catalogue is analysable

**Measurement.** For each candidate threshold `m`, count the items with at least `m`
purchase lines in the whole file, and what share of all lines they account for. The
last column counts sub-commodities that still hold **two or more** surviving items —
groups within which any comparison between similar products remains possible.

![catalogue](figures/basket_eda_catalogue.png)

**Reading it.** Neither panel is a distribution; there is **no unit of observation**,
which makes these different from every other figure here — nothing is counted or
averaged over rows. Both answer "if I demand at least *m* purchases per item, how much
catalogue survives?". So x is a **threshold chosen**, not a value measured, and each
point is a fresh re-count of the entire catalogue. Both curves fall by construction,
and the left panel is log–log because both quantities span orders of magnitude. The
dashed green line counts only items in a sub-commodity that still has ≥2 members.

**Finding.**

| min. purchase lines | items | share of volume | sub-commodities | with ≥2 items |
|---|---|---|---|---|
| 20 | 18,597 | 89.4% | 1,366 | 1,098 |
| 50 | 10,333 | 79.2% | 1,033 | 802 |
| 100 | 5,455 | 65.7% | 758 | 545 |
| 200 | 2,259 | 48.5% | 512 | 309 |
| 500 | 589 | 29.0% | 247 | 121 |

**Reading.** The trade is volume against per-item precision. At **≥100 lines**, 5,455
items carry **65.7%** of volume and 545 sub-commodities remain comparable. Dropping to
≥20 recovers volume (89.4%) but an item seen 20 times across 2,066 households supports
very little per-item estimation; raising to ≥500 leaves only 589 items and 29% of
volume. **≥100 is the working catalogue** used from here on, and everything downstream
inherits that choice.

---

## 5. How much prices move

**Measurement.** For each item, take the median `unit_price` of its transactions in
each week, then compare **consecutive** weeks — pairs where the item sold in both week
`w` and week `w+1`, so a gap in trading is not counted as a price change:

```
moves = share of consecutive item-week pairs with |price(w+1) − price(w)| > $0.01
```

on **335,491** such pairs. The coefficient of variation is `sd / mean` of an item's
transacted prices over the whole panel.

**Finding.** For the 5,455-item catalogue:

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

## 6. Demand response to price

The central question for a dataset built around prices. Measured on the item × week
panel by `scripts/29_demand_eda.py`.

### 6.1 The demand curve

![demand response](figures/demand_eda_price.png)

**Reading it — the left panel.** Twelve dots. Here is exactly what each one is.

Start from the item × week panel: one row per (item, week), holding that item's mean
log price that week and its log buyers-per-trip. For each row compute the change from
the **same item's previous week**:

```
Δ log price  = logp(item, w) − logp(item, w−1)
Δ log buyers = lbuy(item, w) − lbuy(item, w−1)
```

Keep rows where the price actually moved, `|Δ log price| > 0.01`. That leaves
**178,444 rows** — every row is one item in one week, paired with what it did the week
before.

Now sort those 178,444 rows by `Δ log price` and cut into **twelve equal-count bins**
of about 14,870 rows each. Each dot is one bin:

```
x = mean Δ log price   of the rows in that bin
y = mean Δ log buyers  of the rows in that bin
```

The twelve plotted points are:

| bin | x = mean Δ log price | y = mean Δ log buyers/trip | rows |
|---|---|---|---|
| 1 | −0.486 | **+0.701** | 14,871 |
| 2 | −0.236 | +0.355 | 14,934 |
| 3 | −0.142 | +0.189 | 14,806 |
| 4 | −0.083 | +0.006 | 14,870 |
| 5 | −0.043 | −0.069 | 14,871 |
| 6 | −0.012 | −0.079 | 14,894 |
| 7 | +0.026 | −0.153 | 14,848 |
| 8 | +0.054 | −0.190 | 14,872 |
| 9 | +0.092 | −0.180 | 15,029 |
| 10 | +0.147 | −0.202 | 14,708 |
| 11 | +0.235 | −0.245 | 14,885 |
| 12 | +0.470 | **−0.358** | 14,856 |

Read bin 1: in those 14,871 item-weeks the price fell by 0.486 in logs (about 39%
cheaper), and buyers rose by 0.701 in logs (about twice as many). Bin 12: price rose
0.470 in logs (about 60% dearer), buyers fell 0.358 (about 30% fewer).

Two choices worth naming. **Differencing within item** means an item's own price level
and popularity cancel, so the curve is about a price *changing* rather than expensive
items versus cheap ones. **Equal-count rather than equal-width bins** means every dot
rests on ~14,870 rows, so the extremes are as well-supported as the middle.

**Note the panel and the regression are not the same calculation.** The panel uses
week-on-week *differences* on 178,444 rows where the price moved. The elasticity below
is a *within-item* regression on all 556,410 item-weeks, including the ones where the
price held steady. The two agree in sign and rough magnitude, but the slope of the
plotted dots is not the reported elasticity.

**Measurement.** Build an item × week panel: for every item and week,
`buyers` = how many purchase lines it got, `units` = how many units, `trips` = how many
baskets happened that week chain-wide. Then

```
lbuy   = log((buyers + 0.5) / trips)      the 0.5 keeps zero-purchase weeks in
lunits = log((units  + 0.5) / trips)
logp   = mean log price of the item that week
```

The elasticity is an OLS slope of the outcome on `logp` **after subtracting each
item's own mean from both** — a within-item regression. That is what makes it a
statement about a price *changing* rather than about expensive items versus cheap
ones:

```
elasticity = Σ (logp − mean_item logp)(y − mean_item y) / Σ (logp − mean_item logp)²
```

on 556,410 item-weeks.

**Finding.**

| response | elasticity |
|---|---|
| **buyers per trip** | **−0.792** |
| units per trip | −0.945 |
| **units per buyer** | **−0.235** |

**Reading.** The third line separates two channels that the first two combine. A price cut brings
in more buyers *and* makes each buyer take more, and the second channel is **25% of
the total units response**. Whether demand is counted in buyers or in units therefore
changes the answer by a quarter.


```
share from the quantity margin = −0.235 / −0.945 = 24.9%
```

### 6.2 Elasticity by category

*Middle*: a histogram of **160 numbers**, one per category.

Each number is that category's own elasticity, computed by running the §6.1 regression
using **only that category's rows**. Worked through for SOFT DRINKS:

1. Take every (item, week) row belonging to the category — 225 items, **22,950**
   item-weeks.
2. For each row, subtract that **item's** own average from both variables:
   `xd = logp − mean_item(logp)`, `yd = lbuy − mean_item(lbuy)`.
3. The slope is the ratio of two sums over those 22,950 rows:

```
slope = Σ(xd · yd) / Σ(xd²) = −592.58 / 576.95 = −1.027
```

So SOFT DRINKS contributes the single value **−1.027** to the histogram. Repeat for
each of the 160 categories with at least 3 items and 200 item-weeks, and plot the
resulting 160 numbers.

x = a category's slope. y = how many of the 160 categories have a slope in that range.
The extremes come from small categories: FRZN ICE (4 items) sits at −10.1, MAGAZINE
(17 items) at +1.08.

**Measurement.** The §6.1 regression run separately per category, on the 160 with at
least 3 items and 200 item-weeks.

**Finding.**

| | value |
|---|---|
| median | **−0.951** |
| p10 | −1.565 |
| p90 | −0.043 |
| negative | **91%** |
| most elastic | FRZN ICE, CORN, PIES |
| least elastic | VALUE ADDED VEGETABLES, GREETING CARDS/WRAP/PARTY SPLY, MAGAZINE |

**Reading.** The tails are face-valid, which is a check on the measurement itself: frozen
ice and corn are seasonal commodities bought largely on price, while greeting cards
and magazines are impulse purchases where price is close to irrelevant.


The spread is also the point: p10 −1.565 to p90 −0.043 is a factor of 36, so the
−0.792 of §6.1 is an average over categories that behave nothing alike. Price response
has to be estimated per item, not assumed common.

### 6.3 Promotions, as a raw event study

*Right panel.* Every item-week where price fell by at least 0.15 in logs — **37,132
events** — lined up and averaged, with demand expressed relative to that item's own
mean.

**Measurement**, in five steps.

**Step 1 — find the events.** Scan the item × week panel for weeks where an item's log
price fell by at least 0.15 against the previous week (≈14% cheaper). There are
**37,132** such weeks, spread over **4,712 items**, so an item contributes about 8
events on average. Each event is one (item, week) pair.

**Step 2 — normalise demand.** Raw buyer counts cannot be averaged across items: a
staple gets hundreds a week, a niche item gets two. So each item's weekly buyer count
is divided by **that item's own mean over all 102 weeks**:

```
norm_buy(item, week) = buyers(item, week) / mean_all_weeks buyers(item)
```

1.0 means "a normal week for this item", 2.0 means "twice its usual". This is what
lets a 5,000-buyer item and a 50-buyer item be averaged into one curve: both are
expressed as "relative to normal for me".

**Step 3 — line them up.** For each of the 37,132 events, read `norm_buy` at offsets
−3 to +3 weeks around it. Take one real event: item 76, CHOICE BEEF, price cut in week
26. That item averages **5.14** buyers a week across the panel:

| week | offset | buyers | norm_buy = buyers / 5.14 |
|---|---|---|---|
| 23 | −3 | 8 | 1.557 |
| 24 | −2 | 4 | 0.779 |
| 25 | −1 | 1 | 0.195 |
| **26** | **0** | **3** | **0.584** |
| 27 | +1 | 6 | 1.168 |
| 28 | +2 | 5 | 0.973 |
| 29 | +3 | 0 | 0.000 |

One event is pure noise — this one's demand *fell* at the cut. The signal only appears
on averaging.

**Step 4 — average across events at each offset.** Each dot is the mean of `norm_buy`
over all events that have data at that offset:

| offset | mean norm_buy | events contributing |
|---|---|---|
| −3 | 1.003 | 37,018 |
| −2 | 0.948 | 37,104 |
| −1 | 0.959 | 37,132 |
| **0** | **1.940** | 37,132 |
| +1 | 1.284 | 36,716 |
| +2 | 1.169 | 36,338 |
| +3 | 1.151 | 35,964 |

The counts fall slightly away from 0 because an event near the start or end of the
panel has no week to look back or forward to.

**A complication: events are not independent.** An item is cut many times — 37,132
events across only **4,712 items**, about 8 each. CHOICE BEEF alone is flagged in 19
different weeks: 6, 14, 22, 24, 26, 33, 37, 38, … Weeks 22 and 24 are two apart, so
week 24 is simultaneously its own event *and* the "+2" of the week-22 event.

Across all items, **57.7%** of consecutive cuts for the same item fall less than 7
weeks apart, so their ±3 windows overlap. The consequence runs both ways: a week
labelled "−1" may already be lifted by a previous promotion, and the decay at +1..+3
may be the *next* promotion arriving rather than this one persisting.

Both profiles are therefore reported. **Isolated** events are those with no other cut
of the same item within ±3 weeks — **20,725** of 37,132, or 55.8%.

**Step 5 — a reference level to divide by.** Neither 1.0 nor the pre-period works.
`norm_buy` averages exactly 1.0 across *all* of an item's weeks by construction, and
those weeks include its own promotions, so an ordinary week must sit below 1.0. The
pre-period is worse: an event is *defined* as a price drop, so the weeks before one are
mechanically at an above-average price and therefore at below-average demand.

The reference used here is a **quiet week** — an item-week more than 3 weeks from any
cut of that same item. There are **368,173** such item-weeks, and in them

| | value |
|---|---|
| demand ÷ the item's own mean | **0.896** |
| log price − the item's mean log price | **+0.011** |

A quiet week is at the item's usual price and runs at 89.6% of the item's average
demand. That 0.896, not 1.0, is the "nothing happening" line.

**Finding.**

| offset | — all 37,132 events — | | — isolated 20,725 events — | | |
|---|---|---|---|---|---|
| | demand | price | demand | price | demand ÷ quiet |
| −3 | 1.003 | +0.047 | 0.925 | +0.050 | 1.03 |
| −2 | 0.948 | +0.089 | 0.859 | +0.098 | 0.96 |
| −1 | 0.959 | +0.128 | 0.870 | +0.123 | 0.97 |
| **0** | **1.940** | **−0.200** | **2.136** | **−0.194** | **2.38** |
| +1 | 1.284 | −0.130 | 1.337 | −0.139 | 1.49 |
| +2 | 1.169 | −0.061 | 1.117 | −0.055 | 1.25 |
| +3 | 1.151 | −0.029 | 1.065 | −0.003 | 1.19 |
| quiet week | 0.896 | +0.011 | 0.896 | +0.011 | 1.00 |

*demand* = mean `norm_buy`; *price* = mean of `logp` minus the item's own mean `logp`,
so +0.123 is 13% above the item's usual price and −0.194 is 18% below it. Every offset
rests on at least 19,938 item-weeks in the isolated column and 35,964 in the all column.

**Reading, one row at a time.**

*The pre-period does not ramp up.* Demand at −3, −2, −1 is 0.925, 0.859, 0.870 — within
3% of the quiet-week level of 0.896, with no rise heading into the cut. This is the
result that matters for causality: if retailers timed promotions to demand that was
already climbing, the spike at 0 would be partly the reason for the cut rather than its
effect. Demand is flat-to-falling instead.

*What the pre-period does do is get more expensive.* Price runs +0.050, +0.098, +0.123
above the item's mean over the same three weeks. That is not a coincidence, it is the
event definition: a cut of ≥0.15 in logs is easiest to record when the price started
high. It also means the mild demand decline into the cut is exactly what elasticity
predicts. From −3 to −1 price rises 0.123 − 0.050 = **+0.073** in logs and demand falls
ln(0.870 / 0.925) = **−0.061**, an implied slope of −0.061 / +0.073 = **−0.84** — next to
the −0.792 headline elasticity of §6.1.

*The promotion week is far steeper than elasticity alone.* From −1 to 0 price moves
−0.194 − 0.123 = **−0.317** in logs while demand moves ln(2.136 / 0.870) = **+0.898**,
an implied slope of **−2.83**. That is 3.4× the −0.792 estimated on ordinary week-to-week
price variation. A promoted price does not arrive alone — it comes with an endcap, a
shelf tag, a circular — and none of that is in the price variable. A model given only
price will therefore under-predict promotion weeks and over-predict quiet ones. The
−0.792 is an average over two different regimes, not a single constant.

*Most of the post-cut tail is the promotion still running.* This is the correction the
contamination check was built for. Read the two columns at +1: price is still −0.139
below the item's mean, so demand at 1.337 is not persistence, it is a discount that has
not ended. By +3 the price has fully returned — −0.003 against a quiet-week +0.011 —
and demand is 1.065, which is **1.19×** the quiet-week level. That 19% is the genuine
residual. The all-events column puts it at 1.151 / 0.896 = 1.28×, and the extra 9 points
are the next promotion arriving inside the window, not this one persisting.

*No stockpiling is visible within three weeks.* Pull-forward predicts a **dip below**
the quiet-week line after the promotion ends: households bought ahead and stay out of
the market. No offset dips. The lowest post-cut reading, +3 at 1.19×, is 19% above
quiet, and every earlier offset is higher.

**Two limits on that last claim.** The window is ±3 weeks, so stockpiling that unwinds
in month two is invisible. And isolation requires a 4-week clear gap, which drops the
44.2% of events on the most frequently promoted items — precisely where a shopper has
most reason to hold inventory. The finding is "no dip on items promoted less often than
monthly, within three weeks", not "no stockpiling".

**What changed against the earlier version of this section.** It reported a 2.02× lift
against week −1 and a pre-period that was "flat" at 0.970, and read the slow post-cut
decay as demand persisting. All three were artefacts of overlapping windows. The 0.970
pre-period was propped up by neighbouring promotions (0.885 once they are removed,
0.99× quiet), the lift against a quiet week is 2.38× rather than 2.02×, and the
persistence was mostly an unfinished discount — 1.19×, not 1.28×, once the price is
actually back.

### 6.4 Quantity and stores

![quantity and stores](figures/demand_eda_quantity_stores.png)

**Reading it.** *Left*: the unit is one **purchase line** — one item on one receipt.
x = how many units of it were bought (clipped at 6), y = the share of all lines, so the
bars sum to 1 rather than counting.

*Middle*: the unit is an **(item, store, week)** cell where that store's price was
actually observed. x = that store's price minus the chain-wide price for the same item
and week, in dollars, clipped to ±$1. The tall spike at 0 is stores charging exactly
the chain price; the spread either side is genuine cross-store price variation.

*Right*: one observation per **store**. x = the share of the 5,455-item catalogue that
store ever sold, y = how many stores. A store at 0.4 stocks 40% of the catalogue.

**Measurement.** Units per purchase line from `QUANTITY`, capped at 12. Store price
deviation is a store's median price for an item-week minus the chain median for the
same item-week. Assortment is the share of the 5,455-item catalogue a store ever sold.

**Finding.**

| | value |
|---|---|
| rows buying > 1 unit | 22.3% |
| share of all units in those rows | **42.6%** |
| mean units per line | 1.35 |
| store-item-weeks >1c from the chain price | **15.8%** (sd $0.121) |
| catalogue a store carries | median 63%, p10 39% |

### 6.5 Breadth: distinct items per category purchase

**Measurement.** Breadth is the number of distinct items bought from one category on
one trip; quantity is units per line. They move separately. Three different
yogurts is breadth 3, units 3. One yogurt bought three times is breadth 1, units 3.

Across 1,219,633 category visits:

| distinct items in the category | share |
|---|---|
| 1 | 81.1% |
| 2 | 13.4% |
| 3 | 3.4% |
| 4 | 1.2% |
| 5+ | 0.9% |

**Finding.** Mean **1.284** distinct items per category visit;
**18.9%** of visits take two or more different products from the same
category. Those wider visits are also much bigger: a breadth-1 visit averages 1.328
units, a breadth>1 visit averages 3.499.

**Breadth responds to price.** For each category visit take `log(breadth)` and the
**faced** price — the mean log price of *every* item in that category that week, not
of the items the shopper happened to pick — then regress within category:

```
elasticity = Σ (lp − mean_c lp)(log breadth − mean_c log breadth) / Σ (lp − mean_c lp)²
           = -0.1146     on 1,219,633 category visits
```

Using the price of the items actually bought instead gives −0.067. That version is
contaminated: a shopper who adds a second item changes the average by choosing it, so
the regressor moves with the outcome. The faced price is the one reported.

**Three margins, measured separately.** The regression above is fitted only on visits
where the category *was* bought, so it cannot say anything about whether the category
is entered; and units never appear in it, so it cannot say anything about how much is
taken. All three are estimated here from one (category, week) panel — 15,309
category-weeks with at least 5 visits — using the same estimator and the same faced
price:

| margin | quantity regressed | elasticity |
|---|---|---|
| **incidence** | log(visits ÷ baskets that week) — is the category entered at all | **-0.6747** |
| **breadth** | log(item lines ÷ visits) — how many different products | **-0.0919** |
| **depth** | log(units ÷ item lines) — how many of each | **-0.1581** |

**Reading.** A 10% price cut in a category raises the chance a basket enters that
category by about 6.7%, the number of different products taken by 0.9%, and the units
per product by 1.6%. Incidence dominates — roughly 7× breadth and 4× depth. Breadth is
real but it is the smallest of the three.

The three are not three ways of saying the same thing, and the arithmetic shows it.
The identity

```
units per basket = (visits ÷ baskets) × (lines ÷ visit) × (units ÷ line)
```

is exact, so in logs the three elasticities must add to the total-units elasticity:

```
(-0.6747) + (-0.0919) + (-0.1581) = -0.9248
```

against **-0.9453** measured independently in §6.1 on the item × week panel — a gap of
0.021, which is the price of the two panels differing (category-week
aggregates demeaned within category here, item-weeks demeaned within item there). The
decomposition therefore accounts for essentially all of the units response, and says
that about 73% of it is households deciding to enter the category at all.

### 6.6 Base rates

**Measurement.** Counts per basket over the working sample, divided by the number of
categories or items available.

**Finding.**

| event | rate |
|---|---|
| **category incidence** | **6.12 of 188 = 3.25%** |
| item purchase | 7.86 of 5,455 = 0.144% |
| units per basket | 10.64 |

**Reading.** These are the rates any statement about the data has to be consistent
with. A
household buys from **3.25%** of categories on a given trip and **0.144%** of the
catalogue — both small numbers, and both easy to get wrong by an order of magnitude if
they are assumed rather than measured.

---

## 7. Households

Do households differ from one another in ways that persist, or does aggregate
behaviour describe everyone? Two questions: do they like different things, and do they
react differently to price. Measured by `scripts/30_household_eda.py`.

![household heterogeneity](figures/household_eda_heterogeneity.png)

**Reading it.** *Left* is a comparison of two similarity scores, and the construction
is the argument. Take one household. Split its trips in half by date. Turn each half
into a 188-long vector counting how many purchases fell in each category, then scale
each vector to unit length. Compare the two halves with cosine similarity: 1.0 means
identical shopping mix, 0 means no overlap.

Now do that 2,066 times and plot the results as **blue**. Then repeat, but compare
each household's first half against a *different, randomly chosen* household's second
half — that is **grey**. One observation per household in each; y = how many
households.

The comparison is the point: blue is a household against itself over time, grey is a
household against a stranger. If tastes were not stable and personal, the two
distributions would sit on top of each other. Cosine rather than raw counts so that a
household buying 500 items and one buying 50 with the *same mix* count as similar
rather than different.

*Middle*: one observation per household, x = its own estimated price slope (how much
its purchasing falls when prices rise), clipped to [−4, 2], y = how many households.

*Right*: one point per household, x = its slope estimated on the **first half** of its
trips, y = the same household's slope on the **second half**. If the spread in the
middle panel were pure estimation noise, this would be a formless blob centred on the
mean. The visible upward tilt is the +0.24 correlation: households that look
price-sensitive early look price-sensitive later.

### 7.1 Taste stability within household

Does a household keep buying the same kinds of things? Split its history in half by
date and see whether the first half looks like the second half — then check that
against how much it looks like *someone else's* second half.

**Measurement.** For one household:

1. Find the middle day of its shopping and cut its purchases into an earlier half and
   a later half.
2. In each half, count how many purchases fell into each of the 188 categories. That
   gives two lists of 188 numbers. Slot 5 is BEEF for every household, so the lists
   line up.
3. Divide each list by its own length `√(Σ counts²)`. Now only the *mix* is left, not
   the size — a household buying 500 items and one buying 50 in the same proportions
   end up with the same list.
4. Multiply the two lists slot by slot and add up:

```
similarity = Σ v1ᵢ · v2ᵢ        (both lists already scaled to length 1)
```

1.0 means the same mix of categories, 0 means nothing in common. Note that a category
counts only if it appears in *both* halves — 8 purchases in the first half times 0 in
the second contributes nothing.

**The comparison.** Doing this for all 2,066 households gives the blue curve. Grey is the
same arithmetic with one change: each household's earlier half is matched against a
*different* household's later half. Grey uses **every** such pairing —
4,266,290 of them — not a sample, so it does not depend on a
random draw.

**Finding.**

| | cosine similarity |
|---|---|
| a household's own two halves | **0.7838** |
| two different households, averaged over all 4,266,290 pairs | 0.4710 |
| ratio | **1.66×** |
| households scoring above their own average against strangers | **97.6%** |

**Reading.** A household resembles its own later self far more than it resembles
anyone else. That is the point of the comparison: 0.78 on its own would prove nothing,
because everybody buys milk and bread, and two strangers already score 0.47 for that
reason alone. The gap between the two numbers is what is personal.

**Why both curves must be read together.** Self-similarity also rises simply because a
household with more purchases has a fuller, less noisy list — the correlation between
own similarity and log purchase count is +0.643. But the stranger baseline rises with
volume too, and the gap does not move:

| purchases in the first half | households | own | stranger | ratio |
|---|---|---|---|---|
| under 50 | 61 | 0.569 | 0.338 | 1.68× |
| 50–150 | 404 | 0.650 | 0.407 | 1.60× |
| 150–400 | 842 | 0.781 | 0.472 | 1.65× |
| 400+ | 759 | 0.875 | 0.508 | 1.72× |

So the absolute height of the blue curve is partly a data-volume effect. The distance
between blue and grey is not.

### 7.2 Price sensitivity across households

The question is whether *this particular* household buys less when prices rise — one
number per household, not an average over shoppers. That is what a coupon would be
targeted on.

**Measurement.** Comparing a household's beef purchase against its yogurt purchase
says nothing about price; beef and yogurt simply cost different amounts. The only
useful comparison is the **same household buying the same product twice at different
prices**.

1. Keep only (household, product) pairs bought **3 or more times** — 872,715
   purchase rows across 135,055 pairs.
2. For each purchase, work out two things: how far the price was from **what that
   household usually pays for that product**, and how far the units were from **what
   it usually buys of it**. Writing them `xd` and `yd`:

```
xd = logp − (that household's mean logp for that product)
yd = logu − (that household's mean logu for that product)
```

3. Multiply them row by row, add up over all of that household's rows, and divide by
   the summed square of the price deviations:

```
slope_i = Σ xd · yd  /  Σ xd²
```

Subtracting the per-product mean is what keeps beef-versus-yogurt out of it — every
number is relative to that product's own normal for that household. A negative slope
means the household buys less when the price is above its usual level.

**A worked example.** Household 3, product 1054 (CHEESE CRACKERS), bought 10 times:

| day | price | units | xd (price vs its usual) | yd (units vs its usual) | xd·yd | xd² |
|---|---|---|---|---|---|---|
| 104 | 3.59 | 1 | +0.0115 | −0.3466 | −0.0040 | 0.0001 |
| 140 | 3.59 | 1 | +0.0115 | −0.3466 | −0.0040 | 0.0001 |
| 154 | 3.39 | 1 | −0.0459 | −0.3466 | +0.0159 | 0.0021 |
| 199 | 3.59 | 1 | +0.0115 | −0.3466 | −0.0040 | 0.0001 |
| 244 | 3.59 | 2 | +0.0115 | +0.3466 | +0.0040 | 0.0001 |
| 248 | 3.59 | 2 | +0.0115 | +0.3466 | +0.0040 | 0.0001 |
| 251 | 3.59 | 2 | +0.0115 | +0.3466 | +0.0040 | 0.0001 |
| 264 | 3.59 | 2 | +0.0115 | +0.3466 | +0.0040 | 0.0001 |
| 288 | 3.59 | 2 | +0.0115 | +0.3466 | +0.0040 | 0.0001 |
| 617 | 3.39 | 1 | −0.0459 | −0.3466 | +0.0159 | 0.0021 |
| **its own average** | **3.55** | | | | **+0.0397** | **0.0053** |

Summing that household's other 17 repeat-bought products the same way:

```
slope = Σ xd·yd / Σ xd²  =  0.1591 / 1.1430  =  +0.139
```

**Thresholds.** A household needs **80** qualifying rows for its overall slope, leaving
**1,640** of 2,066. The split-half runs the same arithmetic twice — once on the
household's earlier trips, once on its later ones — but with the threshold halved to
**40**, because each half holds roughly half the purchases. 1,374 households clear 40
in both halves, and their two slopes are correlated.

**Finding.**

| | value |
|---|---|
| median | **-0.169** |
| p10 → p90 | **-0.427 → -0.016** |
| sd across households | 0.185 |
| **split-half correlation** | **+0.236** |

**Reading.** The spread on its own proves nothing. Noisy estimates are spread out even
when the true value is the same for everybody. The split-half correlation is the test
that tells them apart, and at +0.236 it is modest but clearly not zero.

**Why it is only +0.236: most households never see the price move.** The slope divides
by `Σ xd²`, which is how much the price actually varied on the things that household
repeatedly buys. When that is near zero, small accidents become large slopes.

Look again at the worked example. The price was 3.59 on eight of the ten visits and
3.39 on two. The units went from 1 to 2 partway through — almost entirely **at the
same price of 3.59**. Something other than price changed, and the arithmetic charges it
to price anyway, because price is the only thing it looks at. That is how this
household ended up with a *positive* slope.

This is common, not exceptional:

| | share |
|---|---|
| rows from a (household, product) pair whose price never moved | **18.6%** |
| pairs with no price variation at all | **16.3%** |

Splitting the 1,640 households by how much price movement they had — `Σ xd²` above or
below its median of 9.35 — shows the effect directly:

| | spread of slopes (sd) | share with a positive slope |
|---|---|---|
| little price movement | **0.226** | 11.5% |
| much price movement | **0.129** | 3.5% |

Where prices genuinely moved, the estimates are tighter and almost all negative, which
is what demand should look like. Where prices barely moved, they fly apart and
11.5% come out positive. So a good part of the apparent
"households differ" is really "some households were measured badly" — which is what
§7.6 pursues.

### 7.3 Store visits

![stores and trips](figures/household_eda_stores_trips.png)

**Reading it.** *Left*: one observation per household. For each, find whichever store
it visits most often, then x = the share of all its trips made at that one store. A
household at 1.0 never shops anywhere else; at 0.4 its favourite store still accounts
for less than half its trips. y = how many households.

*Middle*: one observation per household, x = how many distinct stores it ever visited
(clipped at 15), y = how many households.

*Right*: the unit is a **trip**, but the plot is binned. For every trip, measure the
gap since that household's previous trip, group trips into gap bins (0–3 days, 3–7,
…), and plot the **mean basket size** within each bin. So y is an average over trips,
not a count of them, and the x bins have unequal widths — the axis is categorical, so
horizontal distance is not proportional to days.

**Measurement.** One observation per household. Primary store is whichever it visits
most; a switch is a trip at a different store from the previous trip.

**Finding.**

| | value |
|---|---|
| median trips per household | 71 |
| median distinct stores | **4** (p90 9) |
| trips at the primary store | median **76%**, p10 42% |
| consecutive trips that switch store | **30.1%** |
| households using only one store | 7.6% |

**Reading.** Households are loyal but not monogamous: the typical one does 76% of its trips at
a primary store, yet 30% of consecutive trips switch, and only 7.6% ever use a single
store.

So a household is not well described by one store. And because the same household
shops several stores that differ in price and assortment, store differences can be
observed *within* a household rather than only by comparing different households —
which is the cleaner comparison, since it holds taste fixed.

### 7.4 Trip rhythm

**Measurement.** One observation per trip. Gap is days since that household's previous
trip; a large trip is in the top quartile by item count.

**Finding.**

| | value |
|---|---|
| median items per trip | 4 |
| median gap between trips | 3 days (p90 14) |
| correlation, gap vs trip size | **+0.070** |
| gap before a large trip | 5 days |
| gap before a small trip | 3 days |

**Reading.** A weak but consistent stock-up pattern: longer gaps precede bigger
trips. The
correlation is +0.07 — real but small, so trips do not separate cleanly into distinct
"top-up" and "stock-up" types; the variation is continuous.

### 7.5 Demographics

**Measurement.** Join `hh_demographic.csv` to the per-household price slopes from
§7.2, and take the median slope within each level of each attribute, keeping levels
with at least 30 households.

**Finding.** Demographics cover 39% of modelled households.

| attribute | levels | span of median slope |
|---|---|---|
| `classification_5` | 6 | 0.076 |
| `classification_3` | 6 | 0.067 |
| `classification_1` | 6 | 0.061 |
| `classification_4` | 5 | 0.027 |
| `KID_CATEGORY_DESC` | 4 | 0.016 |
| `classification_2` | 3 | 0.006 |

**Reading.** The largest spread across any demographic attribute is 0.076, against a
p10–p90 spread of 0.411 across households. **Demographics explain almost none of the
variation in price sensitivity** — under a fifth of it.

Two households in the same demographic cell differ from each other about as much as
two households drawn at random. Whatever drives price sensitivity here, it is not the
recorded observables.

---

### 7.6 Reliability of the per-household estimate

**Measurement.** Each household's price slope (§7.2) estimated twice, on disjoint
halves of its own history split at its median shopping day, then correlated across
households. Four tests: against a null that shuffles household labels; corrected for
using half the data per side; recomputed at stricter minimum-data thresholds; and
used to rank households on one half and measure them on the other.

![reliability](figures/reliability.png)

**Reading it.** *Left*: x = a correlation value, y = how many of 200 shuffles produced
it. Each shuffle moves the second-half slopes onto the wrong households, keeping both
sets of numbers intact and destroying only which household each pair belongs to. The
grey pile is therefore **how far from zero a correlation lands by chance when there is
no relationship at all**, given 1,374 households — not a picture of measurement error.
It is centred at +0.003 with sd 0.025, matching the theoretical 1/√(n−1) = 0.027. The
red line is the real, unshuffled value.

Two different things get called noise in this section, and they do different work.
*Measurement noise* is each household's slope being imprecise, and it is what holds the
correlation down to 0.236. *Chance in the correlation itself* is what this panel
measures, and it is ±0.025. This panel establishes only that 0.236 is really there; the
other three ask whether it is large enough to use.
*Middle*: each point is the whole analysis re-run at a stricter minimum, x = purchase
rows a household must have **in each half** to be included, y = the resulting
correlation; `n=` labels how many households survive each cut. *Right*: households are
sorted by their **first-half** slope into three equal groups, and each bar is that
group's mean slope in the **second half** — data never used to form the groups.

**Finding 1 — it is real.** Shuffling household labels gives a null centred at **+0.003** with
sd **0.025**. The actual +0.236 sits **9.2 standard deviations above** it (p < 0.005).
Households genuinely resemble their past selves.

**Finding 2 — it understates the full history.** Split-half deliberately throws away half the
data on each side. The Spearman–Brown correction for that is `2r/(1+r)`:

```
2 × 0.236 / (1 + 0.236) = 0.381
```

So an estimate built on a household's **whole** record has reliability around
**0.38**, not 0.236. The lower number answers "how well do two halves agree", which is
not the question a targeting system asks.

The word *implied* matters: 0.381 is **predicted**, not measured. The formula rests on
two assumptions — that the two halves measure the same thing (the household's taste did
not drift across the year) and that error variance falls in proportion to data.

**Both are testable one level down.** Cut each household into four consecutive
quarters instead of two halves. Measure how well the quarters agree, extrapolate that
to halves with the same `2r/(1+r)`, then compare against the half-vs-half correlation
actually observed on those same households:

| | value |
|---|---|
| households with a usable slope in all four quarters and both halves | 1,045 |
| quarter-vs-quarter reliability (mean of the 6 pairs) | +0.1064 |
| what `2r/(1+r)` predicts for halves | +0.1924 |
| **what the halves actually give** | **+0.3199** |

The formula **understates** the gain by 40%. Doubling the data helped more than
proportionally, and the reason is visible in §7.2: the slope divides by `Σ xd²`, the
price movement a household actually saw. A quarter-length record does not just have
fewer rows — whole (household, item) pairs drop below the 3-purchase minimum and
contribute nothing at all. Lengthening the window adds usable pairs as well as rows.

So 0.381 is a conservative floor for the full-history reliability, not an optimistic
ceiling.

**Finding 3 — the limit is measurement noise.** Re-running with stricter data
requirements:

| minimum rows per half | households kept | split-half r | implied full-history |
|---|---|---|---|
| ≥ 20 | 1,641 | **−0.136** | — |
| ≥ 40 | 1,374 | +0.236 | 0.381 |
| ≥ 80 | 1,052 | +0.313 | 0.477 |
| ≥ 160 | 701 | +0.403 | 0.574 |
| ≥ 320 | 314 | **+0.506** | **0.672** |

Reliability more than **doubles** as households provide more data, and at ≥20 rows it
is actually *negative* — pure noise. If households were genuinely similar to one
another, more data per household would not help; the ceiling would stay put. It rises
steeply, so the constraint is **how precisely each household is measured**, not how
much they differ.

For a household with a long record, price sensitivity is a reasonably stable trait
(0.67). For a light shopper it is barely measurable at all.

**Finding 4 — it supports ranking.** Sort households into thirds by their first-half
slope, then measure the second half:

| group (ranked on first half) | mean slope in the held-out half |
|---|---|
| most sensitive | **−0.273** |
| middle | −0.193 |
| least sensitive | **−0.136** |

The most-sensitive third really is about **twice** as price-responsive as the
least-sensitive third, on data that played no part in forming the groups. The gap is
**0.14, or 0.53 standard deviations** — a real, monotone separation, and a modest one.

**Reading.** Sorting households by price sensitivity works, and it is not an
artefact. But the ordering is noisy for light shoppers and much sharper
for heavy ones, so a targeting policy should weight by how much is known about each
household rather than treating every estimate as equally trustworthy. The earlier
claim that "only a quarter is real" was too pessimistic; a claim that this cleanly
separates customers would be too optimistic.

## 8. Exogeneity of the price variation

§5 and §6 establish that prices move and that demand moves against them. Whether that
association is **causal** is a separate question: prices might move *because* demand
was expected to move. This section tests it without a model.

**Measurement.** Build an item × week panel of log purchase rate on log price, absorb
item fixed effects always and week fixed effects optionally, cluster standard errors
by item. Then refit the identical regression on four **fake** price series, per
category, on the 160 categories large enough to estimate:

| placebo | how the fake series is built | what it destroys |
|---|---|---|
| forward shift | the item's own series moved +6 weeks | timing, partially |
| backward shift | moved −6 weeks | timing, partially |
| **weeks reordered** | the item's own weeks randomly permuted | all time alignment |
| **another item's series** | item *j* given item *k*'s prices | time *and* item alignment |

A fake price series cannot cause demand. Any coefficient it produces is what the design
manufactures from nothing, so it is the null the real estimate must be judged against.

**Finding.** With week fixed effects:

| price series | median coefficient | categories significant at 1% |
|---|---|---|
| **real prices** | **-0.844** | 52.5% |
| shifted +6 weeks | -0.082 | 10.6% |
| shifted −6 weeks | -0.048 | 9.4% |
| **weeks reordered** | **-0.006** | 3.1% |
| **another item's series** | **+0.002** | 4.4% |

Per-category verdicts:

| | categories |
|---|---|
| scored | 160 |
| real effect significantly negative | 82 |
| fail at least one placebo | 26 |
| **fail a strict placebo** (reorder or swap) | **5** |

**Reading.** The two strict placebos retain
`0.7%` and
`0.2%` of the real
coefficient, and only 5 of 160 categories
fail one. The shift placebos behave worse (10.6% and
9.4%), but a series shifted six weeks stays
correlated with the real one, so they were never clean nulls — their failure is
expected and not informative.

### How much of the association is the calendar?

**Measurement.** The same regression run twice, with and without week fixed effects.
Week effects absorb anything that moves all items together at week frequency —
seasons, holidays, chain-wide promotional calendars.

**Finding.**

| specification | median coefficient |
|---|---|
| no week effects | -0.951 |
| week effects | -0.844 |

```
share removed = 1 − (-0.8438 / -0.9511) = 0.113
```

**Reading.** **11.3%** of the raw price–demand association is
week-frequency seasonality — prices and demand moving together over the year rather
than one moving the other. Small enough to be missed, large enough to matter for any
quantitative claim about price response. It is a **lower bound** on confounding: week
effects catch what moves all items together, not what moves one item at one store.

---

## 9. Summary of findings

Every row links to the section carrying the measurement and formula.

### Basket composition

| finding | value | section |
|---|---|---|
| baskets holding 2+ items from one category | **56.1%** | §1 |
| baskets spanning 2+ categories | 81.5% | §1 |
| distinct items per basket | mean 10.1, median 5 | §1 |
| categories where >15% of category-trips buy 2+ items | 132 of 307 | §1 |
| distinct items per **purchased category** | **1.284** | §6.5 |
| units per purchase line | mean 1.35; 22.3% buy >1, carrying 42.6% of units | §6.4 |

### Product relationships

| finding | value | section |
|---|---|---|
| sub-commodity pairs measured | 29,117 on 83,928 baskets | §2 |
| median lift | 1.13 | §2 |
| pairs at 2× chance or more | 11.0% | §2 |
| pairs at 0.67× or less (avoided) | 4.5% | §2 |
| **same-sub-commodity items co-occurring vs chance** | **7.88×** | §2 |

### Timing

| finding | value | section |
|---|---|---|
| repeat-purchase events | 1,082,615 | §3 |
| median gap between repeats | 27 days (p25 10, p75 71) | §3 |
| **repurchase hazard swing, peak to floor** | **4.30×**, non-monotone | §3 |
| median gap between a household's trips | 3 days | §7.4 |
| correlation, trip gap vs trip size | +0.070 | §7.4 |

### Prices and demand

| finding | value | section |
|---|---|---|
| items with ≥100 purchase lines | 5,455, 65.7% of volume | §4 |
| consecutive item-weeks where price moves >1¢ | **30.5%** | §5 |
| median within-item coefficient of variation | 0.136 | §5 |
| **elasticity of buyers** | **-0.792** | §6.1 |
| **elasticity of units** | **-0.945** | §6.1 |
| elasticity of units per buyer | -0.235 = 25% of the units response | §6.1 |
| elasticity by category | median -0.951, p10 -1.565, p90 -0.043; 91% negative | §6.2 |
| demand at a price cut | **2.38×** a quiet week, on the 20,725 non-overlapping events of 37,132 | §6.3 |
| residual once the price is back (+3 wks) | **1.19×** a quiet week; no post-promotion dip | §6.3 |
| implied slope in the promotion week | **-2.83**, 3.4× the -0.792 average — price alone does not explain a promotion | §6.3 |
| elasticity of incidence, breadth, depth | **-0.675**, -0.092, -0.158; they sum to -0.925 against the -0.945 of §6.1 | §6.5 |

### Households

| finding | value | section |
|---|---|---|
| **taste self-similarity vs cross-household** | **1.66×** (0.784 vs 0.471 over all 4,266,290 pairs); 97.6% of households above their own stranger average | §7.1 |
| price sensitivity across households | median -0.169, p10 -0.427, p90 -0.016 | §7.2 |
| stores per household | median 4, p90 9 | §7.3 |
| trips at the primary store | median 76% | §7.3 |
| consecutive trips switching store | 30.1% | §7.3 |
| demographic coverage | 39% of households | §7.5 |
| **split-half reliability of price sensitivity** | **+0.236**, 9.2 sd above a shuffled null | §7.6 |
| implied full-history reliability | +0.381 | §7.6 |
| reliability at ≥320 rows per half | **+0.505** — noise-limited, not similarity-limited | §7.6 |
| held-out separation, top vs bottom third | 0.53 sd | §7.6 |

### Causality

| finding | value | section |
|---|---|---|
| strict placebos retain | 0.7% and 0.2% of the real effect | §8 |
| categories failing a strict placebo | **5 of 160** | §8 |
| **share of the association that is seasonality** | **11.3%** | §8 |

### Base rates

| event | rate | section |
|---|---|---|
| category incidence | 6.12 of 188 = **3.25%** | §6.6 |
| item purchase | 7.86 of 5,455 = 0.144% | §6.6 |
| units per basket | 10.64 | §6.6 |

### The sample these are measured on

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

---

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
