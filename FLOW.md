# The complete flow: from dunnhumby's raw CSVs to fitted model and results

This is the map. It follows one household's shopping trip all the way through, names
every file, and gives the row count at each step so you can check your own run against
it. No statistics background assumed.

If you only read one thing, read §1 (the idea) and §3 (the worked example).

---

## 1. What we are trying to build, in one paragraph

We want a model that can answer: *if this specific household walked into the store on
Monday and the price of this specific product had gone up 30 cents, would they still
have bought it — and if not, what would they have bought instead?*

To answer that we need, for every shopping trip, a list of the products the shopper
could have bought, the price of **each** of them (not just the one they picked), and
what they actually chose. dunnhumby gives us receipts. A receipt tells you what someone
bought and what they paid. It does not tell you the price of the things they *didn't*
buy — and that is exactly what a choice model needs. Most of the preprocessing is
reconstructing those unobserved prices, and then deciding which products and shoppers
have enough clean price movement to learn from.

---

## 2. Vocabulary

Terms used throughout the code and the other documents.

| term | plain meaning |
|---|---|
| **household** | one shopper family, identified by `household_key`. Also called a "user" in the model code. |
| **trip** | everything one household bought on one calendar day. The unit of choice. |
| **item / product / UPC** | one specific sellable product, e.g. a particular brand and size of paper towel. `PRODUCT_ID`. |
| **category** | a set of products a shopper picks *at most one of* — e.g. PAPER TOWELS. We use dunnhumby's `COMMODITY_DESC`. |
| **sub-commodity** | a finer grouping inside a category, e.g. "PAPER TOWELS & HOLDERS". **Deliberately hidden from the model** so we can use it later to test whether the model figured out product similarity on its own. |
| **choice set** | the products available in a category on a trip — here, the 10 most popular items in that category. |
| **outside good** | the option of buying nothing from the category. Chosen on ~98% of category-trip opportunities. |
| **session** | a block of time within which prices don't change. Here: one calendar day, chain-wide. |
| **pair-week** | our unit of "before and after a price change": the Sunday of one week paired with the Monday of the next. Prices reset in between. |
| **posted / shelf price** | the price on the tag, the same for everyone. What we want. |
| **loyalty price** | the shelf price for card holders. Every household in this panel has a card, so this *is* the relevant shelf price. |
| **held-out / test data** | trips deliberately hidden during fitting, used to check the model predicts things it has never seen. |
| **log-likelihood** | how surprised the model is by what actually happened. Closer to zero is better; −4.3 is better than −4.7. |
| **elasticity** | percentage change in how much of something gets bought when its price goes up 1%. −1.2 means a 1% price rise cuts sales 1.2%. |
| **placebo test** | move price changes to weeks when prices didn't actually change. If the model still "finds" an effect, the effect was never really about price. |

---

## 3. One trip, end to end

Household **1142**, calendar **day 54** (a Sunday), store 335. This is real; you can
reproduce every line.

### Step 1 — the raw receipt

`transaction_data.csv` has **57 lines** for this household on this day. Twelve of them
are for products that survive into the final sample; eleven are shown here (the twelfth
is a second facial tissue, which matters in step 3):

| PRODUCT_ID | QUANTITY | SALES_VALUE | RETAIL_DISC | what it is |
|---|---|---|---|---|
| 824005 | 1 | 1.99 | 0.00 | sliced white mushrooms |
| 854405 | 1 | 6.99 | −3.00 | boneless chicken breast |
| 946839 | 1 | 2.79 | 0.00 | beef, primal cut |
| 963835 | 1 | 1.99 | 0.00 | paper towels |
| 995965 | **2** | 2.99 | **−2.99** | salad mix — a 2-for-1 |
| 1004906 | 1 | 1.99 | 0.00 | russet potatoes |
| 1044676 | 1 | 0.16 | 0.00 | yellow onions |
| 1090701 | 1 | 1.50 | −0.49 | frozen garlic bread |
| 1108094 | 1 | 11.99 | 0.00 | beer |
| 5585510 | 1 | 2.79 | 0.00 | vine-ripe tomatoes |
| 10121610 | 1 | 1.09 | 0.00 | facial tissue |

There is **no price column**. `SALES_VALUE` is what the retailer banked.

### Step 2 — turn money into a price

For the salad mix: two units for $2.99 total, with a $2.99 loyalty discount applied.

```
price this shopper faced   = SALES_VALUE / QUANTITY          = 2.99 / 2 = $1.495
price without a loyalty card = (2.99 + 2.99) / 2             = $2.99
```

Every household in this panel has a loyalty card, so **$1.495 is the price that
mattered**. (§1 of `PREPROCESSING.md` shows why, and why the two other discount columns
are handled differently.)

### Step 3 — one row per category, not per product

The model asks "which paper towel?", not "did you buy these eleven things?". So the
trip becomes **one row per category the household bought from**.

This household illustrates the rule nicely. Twelve of its 57 lines are for products we
model — but two of them are both facial tissue (products 976864 at $0.99 and 10121610
at $1.09, bought on the same trip). The model assumes you pick *at most one* item per
category, so one of the two is kept at random and the other is treated as not chosen.
That is the paper's own rule, and it is why twelve lines become **eleven rows**.

### Step 4 — what the model actually receives

`model_input/train.tsv`, four tab-separated integers per line:

```
963   384   8   1        household 963, item 384, session 8, chose it
963   410   8   1
963   501   8   1
...
```

* `963` is household 1142 renumbered 0…2083.
* `384` is product 963835 (paper towels) renumbered 0…559.
* `8` is day 54 renumbered 0…171.

Alongside, `model_input/item_sess_price.tsv` carries the price of **all 560 items on
all 172 days** — including the 549 items this household did *not* buy that day. That
completed grid is what makes the counterfactual question answerable.

For contrast, here is what the eleven raw lines above became — one row per category:

| model item | product | category | sub-commodity (hidden from the model) |
|---|---|---|---|
| 58 | 946839 | BEEF | PRIMAL |
| 63 | 1108094 | BEERS/ALES | BEER/ALE/MALT LIQUORS |
| 214 | 10121610 | FACIAL TISS/DNR NAPKIN | FACIAL TISSUE |
| 233 | 1090701 | FROZEN BREAD/DOUGH | FRZN GARLIC BREAD |
| 310 | 854405 | MEAT - MISC | BREAST — BONELESS |
| 320 | 824005 | MUSHROOMS | MUSHROOMS WHITE SLICED |
| 369 | 1044676 | ONIONS | ONIONS YELLOW |
| 384 | 963835 | PAPER TOWELS | PAPER TOWELS & HOLDERS |
| 410 | 1004906 | POTATOES | POTATOES RUSSET |
| 442 | 995965 | SALAD MIX | GARDEN PLUS |
| 501 | 5585510 | TOMATOES | TOMATOES VINE RIPE |

The remaining 45 lines are products we do not model at all. Exactly why, for this trip:

| lines | reason |
|---|---|
| 36 | their category failed the one-item-per-trip test — shoppers routinely buy two or three at once, so "which one did you pick?" is not a meaningful question |
| 5 | their category has too few purchases in this window to estimate anything |
| 3 | their category *is* modelled, but the product is not among its 10 most popular |
| 1 | their category is too seasonal |

Which is a fair snapshot of the whole dataset: the one-item-per-trip rule is by far the
biggest reason things get dropped.

### Step 5 — what the model learns from it

For this trip and the PAPER TOWELS category, the model computes a score for each of the
10 paper-towel products, using this household's learned taste, this household's learned
price sensitivity, and each product's price that day. It compares those scores against
what was actually chosen, and nudges the numbers. Then it does the same for whether the
household bought paper towels *at all*.

---

## 4. The pipeline, stage by stage

Run everything with `bash scripts/run_all.sh` (~35 minutes). Or run the numbered
scripts in order.

### Inputs (untouched)

| file | rows | what it gives us |
|---|---|---|
| `transaction_data.csv` | 2,595,732 | every receipt line: who, what, when, where, how much |
| `product.csv` | 92,353 | the product hierarchy and brand |
| `hh_demographic.csv` | 801 | age, income, household size — **only 801 of 2,500 households** |
| `causal_data.csv` | 36,786,524 | was this product on display / in the mailer, per store per week |
| `coupon.csv` + `campaign_table.csv` + `campaign_desc.csv` | 124,548 + 7,208 + 30 | which household could redeem which coupon, and when |
| `coupon_redempt.csv` | 2,318 | coupons actually redeemed |

### `01_build_base.py` — clean receipts, compute prices

* Reconstructs the shelf price on every line (§3 step 2).
* Drops bulk lines and absurd prices: 2,595,732 → **2,553,408** lines.
* Works out the day of week from the shopping-volume pattern (dunnhumby anonymises
  dates, so Saturday and Sunday are identified as the two busiest weekday slots).
* Groups lines into **213,961 trips** (household × day).
* Builds two price tables: product × week chain-wide, and product × store × week.

Outputs: `data/tx.parquet` (2.55M), `data/trips.parquet` (213,961),
`data/price_week.parquet` (1.07M), `data/price_store_week.parquet` (2.35M).

### `02_select_sample.py` — choose who, what and when to model

Seven filters, in order. Each one is justified in `PREPROCESSING.md`.

| step | rule | effect |
|---|---|---|
| keep the price-change window | Sundays and Mondays only | 30% of trips |
| drop holiday weeks | chain spend >12% off its local trend | 87 of 101 pair-weeks left |
| drop broken days | a day with <50 baskets chain-wide is a hole in the panel, not a quiet day | removes 1 pair-week, leaving **86** |
| keep regular shoppers | 20–300 trips over two years | 2,137 households; **2,084** of them go on to buy from at least one surviving category and end up in the model |
| drop non-merchandise | fuel, postage, lottery, coupons | — |
| drop scale-weighed items | their "unit price" is really weight × price-per-pound | 2,142 of 30,452 priced products |
| the paper's category filters | enough items, one-per-trip holds, enough price movement, not too seasonal | 285 → **56 categories** |

Outputs: `data/sample_trips.parquet` (49,729), `data/sample_choices.parquet` (66,638),
`data/items.parquet` (560), `data/filter_audit.csv` (why each category was kept or cut).

### `03_make_model_inputs.py` — write the files the model reads

Renumbers households, items and days to 0-based integers, splits the data, and writes
the exact file formats the authors' C++ expects.

| file | lines | contents |
|---|---|---|
| `train.tsv` | 46,431 | household, item, session, 1 |
| `validation.tsv` | 6,471 | same, used to decide when to stop |
| `test.tsv` | 13,736 | same, never touched during fitting |
| `item_sess_price.tsv` | 96,320 | **every** item × session price = 560 × 172 |
| `itemGroup.tsv` | 560 | which category each item belongs to |
| `sess_days.tsv` | 172 | which pair-week and weekday each session is |
| `obsUser.tsv` | 2,084 | nine demographic columns per household |
| `obsItem.tsv` | 560 | four product attributes per item |
| `events.csv` | 48,160 | which item-weeks had a price change — the natural experiments |
| `id_maps/` | — | model numbers ↔ real dunnhumby ids |

**The split is by household × pair-week**, not by random trip. A household appears in
training, but specific *weeks* of its behaviour are hidden. That forces the model to
predict a household it knows, in a week it has never seen, on the day a price moved.

### `04_extras.py` — the signals dunnhumby has and the paper's data didn't

* `item_sess_display.tsv`, `item_sess_mailer.tsv` — was this product on an end-cap or
  in the weekly mailer, as a share of shopper traffic exposed.
* `coupon_campaigns.npz` — which household could redeem a coupon on which product and
  when. Stored as 23 campaign membership lists rather than a 12-million-cell table.
* `redemptions.csv` — coupons actually used, held back for validation only.

### `11_placebo_tests.py` — **before believing anything**

Moves each price change to a week when prices didn't actually change, and refits. If
the "price effect" survives that, it was never about price. Run before the models are
fitted, because it decides whether the whole design is credible.
Results: `PREPROCESSING.md` §9.

### `05_train_nf.py` — fit the model

Two stages, as in the paper.

1. **Which product, given you're buying from this category.** Learns a taste vector
   and a price-sensitivity vector per household, and a matching pair per product.
2. **Whether you buy from the category at all.** Takes a summary of stage 1 (how
   attractive the category's products look to *you*, today) and predicts purchase.

Four versions are fitted so the comparisons mean something: the full model, one with
no personalisation, one where every category is learned in isolation, and one with the
display/mailer/coupon extras.

### `06`–`09`, `12`–`17` — selection, evaluation, verification

| script | what it answers |
|---|---|
| `06_hyperparam_sweep.py` | how many latent dimensions, chosen on price-change weeks only |
| `07_evaluate.py` | the paper's full evaluation battery on held-out data |
| `08_data_report.py` | descriptive statistics |
| `09_counterfactual_checks.py` | do the households the model calls price-sensitive actually respond more? |
| `10_price_definition_audit.py` | is the price reconstruction right? |
| `12_preprocessing_figures.py` | every figure in `PREPROCESSING.md` |
| `13_placebo_followup.py` | which categories fail, and does dropping them change anything |
| `14_verify_model.py` | does our code recover parameters it is given? |
| `15_cpp_crosscheck.py` | does it agree with the authors' original C++? |
| `16_inspect_embeddings.py` | what did the model actually learn? |
| `17_store_diagnostics.py` | does pooling 561 stores cost us anything? |

---

## 5. The shape of the data at each stage

```
transaction_data.csv          2,595,732 receipt lines
        |  01: clean, price, group into trips
        v
data/tx.parquet               2,553,408 lines  ->  data/trips.parquet   213,961 trips
        |  02: Sun/Mon window, holidays, households, categories, items
        v
data/sample_trips.parquet        49,729 trips
data/sample_choices.parquet      66,638 (trip, category, chosen item)
data/items.parquet                  560 items in 56 categories
        |  03: renumber, split, complete the price grid
        v
model_input/train.tsv            46,431      \
model_input/validation.tsv        6,471       |  the model's view
model_input/test.tsv             13,736       |
model_input/item_sess_price.tsv  96,320      /   = 560 items x 172 days
        |  05: fit
        v
out/nf_stage1.pt, out/nf_stage2.pt
        |  07, 09, 14-17: evaluate and verify
        v
out/evaluation_summary.csv, figures/*.png
```

The two numbers worth staring at: **66,638 observed choices**, and **96,320 prices**.
The second is bigger than the first, and that is the point — most of the work is
constructing prices for choices that were never made.

---

## 6. Where the numbers shrink, and why

Starting from 2,500 households and 92,353 products, we model 2,084 households and 560
products. That looks like heavy loss, so here is where it goes and whether it was
necessary.

| we go from | to | because |
|---|---|---|
| 2,595,732 receipt lines | 2,553,408 | bulk quantities and impossible prices |
| all 7 days | Sun + Mon (30%) | those two days straddle the weekly price reset — this is the whole identification strategy |
| 101 pair-weeks | 86 | 14 lost to holidays, 1 to a day with almost no baskets chain-wide |
| 2,500 households | 2,137, of which **2,084** appear in the final model | very light and very heavy shoppers behave differently (the paper's rule); the last 53 never buy from a surviving category on a Sunday or Monday |
| 92,353 products | 30,452 with a usable price in this window | most products barely sell |
| 30,452 | 28,310 | scale-weighed items have no posted price to learn from |
| 285 categories | 56 | mostly "too thin": not enough purchases in this window to estimate anything |
| all items in a category | top 10 | the paper's choice set; the 10 most popular cover most purchases |

The binding constraint is **sample size**, not judgement. The paper had 100,504 trips
in one store; we have 49,729 spread over 561 stores. That single fact explains why we
keep 56 categories where they kept 123.

---

## 7. What to look at first when reading the results

1. `PREPROCESSING.md` §9 — the placebo tests. If those failed completely, nothing else
   matters.
2. `REPORT.md` §B.5 — the head-to-head model comparison on held-out data.
3. `VERIFICATION.md` §5 — the limit on how much can be believed about any single
   household's price sensitivity.

And the honest one-line summary: the model works and beats its simpler alternatives on
data it has never seen, the *ranking* of households by price sensitivity is real, and
individual households' elasticity numbers are too noisy at this sample size to use as
numbers.
