# From raw dunnhumby CSVs to model input

What the version-4 basket model actually reads, where each file comes from, and how to
rebuild any of it. Every path and number here was read off the files on disk, not from
memory.

`PREPROCESSING.md` documents a **different, older** pipeline (`data/` → `model_input/`)
that served the paper's multinomial model. The current model in `scripts/v3_codex/` does
not read those files. It reads `basket_input/`. If you are looking for how the model that
runs today is fed, this is the document.

---

## 1. The chain in one view

```
  dunnhumby CSVs                    (external, read-only)
        |
        |  scripts/pipeline/01_build_base.py
        v
  data/tx.parquet, trips.parquet, price_week.parquet, price_store_week.parquet
        |
        |  scripts/pipeline/22_basket_data.py      (+ 23_promo_data.py for promotions)
        v
  basket_input/   baskets.parquet  items.parquet  log_price.npy  log_price_dev.npy
                  store_price.npz  state.npz  promo.npz  meta.json
        |
        |  scripts/v3_codex/data.py :: build()      cached, once
        v
  basket_input/v3_index*.npz        the ragged assortment index + trip table
        |
        |  scripts/v3_codex/features.py :: Features      lazily, at batch time
        |  scripts/v3_codex/fit.py     :: Batcher
        v
  RaggedIndex + context tensors  ->  RaggedModel
```

Stages 1–2 are run once by hand. Stage 3 is cached automatically the first time any
script calls `build()`. Stage 4 happens per minibatch and is never written to disk.

---

## 2. Where the raw data lives

```
../dunnhumby_The-Complete-Journey/dunnhumby_The-Complete-Journey CSV/
```

relative to the repo root — i.e. a sibling of `nf_dunnhumby/`. Override with the
`NF_RAW_DIR` environment variable (`scripts/pipeline/01_build_base.py:38`).

| file | size | used for |
|---|---:|---|
| `transaction_data.csv` | 142 MB | every purchase line: basket, household, product, units, sales value, discounts, store, day, week |
| `product.csv` | 6.4 MB | product hierarchy — department, commodity, sub-commodity, brand, manufacturer |
| `causal_data.csv` | 696 MB | in-store display and mailer placement per product/store/week |
| `hh_demographic.csv` | 43 KB | household demographics (not used by the current model) |
| `campaign_*.csv`, `coupon*.csv` | small | coupon campaigns (not used by the current model; see TODO — different ID space) |

The raw data is never modified.

---

## 3. Stage 1 — `01_build_base.py` → `data/`

Reads `transaction_data.csv` and builds the base tables. The important derivation here is
**price**, because the raw file has no price column: it has `SALES_VALUE` (what the
retailer received) plus three discount columns. The shelf price is reconstructed so that
the model conditions on what every shopper saw, including those who did not buy.
`PREPROCESSING.md` §1 documents that derivation and its audit in full.

Outputs used downstream: `data/tx.parquet` (40 MB, the cleaned transaction log).

---

## 4. Stage 2 — `22_basket_data.py` → `basket_input/`

This is the stage that defines the modelling universe. It deliberately **replaces** the
older `02_`/`03_` scripts rather than extending them, because each of their filters
protects an assumption this model drops:

| the paper's pipeline | here |
|---|---|
| at most one item per category per trip → 132 of 307 categories dropped | a basket is a multiset; several items and several units |
| categories independent | items interact through a shared embedding |
| Sunday + Monday only (172 sessions) | all 711 days, because household state needs time |
| → 560 items, 56 categories | → 5,455 items, 758 sub-commodities |
| price pooled to chain level | store-level price where observed |
| no assortment | per-store availability, so an item a store never carries is not scored as "rejected" |

### Selection cuts (from `meta.json`)

```
  min_lines  100     a product needs >= 100 purchase lines to be kept
  min_trips   20     a household needs >= 20 trips
  max_trips  300     and <= 300
```

giving **5,455 products, 2,066 households, 115 stores, 712 days, 199,345 baskets,
1,566,063 purchase lines**.

### Temporal split

By household × week, so a household contributes to more than one split:

```
  train        weeks  < 83     1,228,695 rows
  validation   weeks 83..90      134,665 rows
  test         weeks >= 91       202,703 rows
```

This is a **temporal** split — the reason recency features are non-stationary across it.

### Files written

| file | shape / size | what it is |
|---|---|---|
| `baskets.parquet` | 1,566,063 × 10 | one row per (basket, item): `BASKET_ID, user_id, DAY, WEEK_NO, item_id, sub_id, units, price, store_id` |
| `items.parquet` | 5,455 × 14 | `item_id ↔ PRODUCT_ID` plus held-out labels the model never sees (sub-commodity, commodity, brand, manufacturer, department) |
| `log_price.npy` | [5455, 712] | log price by item and day, carried forward over gaps |
| `log_price_dev.npy` | [5455, 712] | the same, as a deviation from each item's own mean |
| `store_price.npz` | 244,880 cells | store-level price deviations where observed (24.7% of cells), plus `carried[5455,115]` — the per-store availability mask |
| `state.npz` | 1.36 M keys | the "days since this household last bought this sub-commodity" structure |
| `promo.npz` | 6.52 M keys | display and mailer flags (written by `23_promo_data.py` from `causal_data.csv`) |
| `meta.json` | — | all sizes, split boundaries and cuts quoted above |

### Why `state.npz` looks the way it does

The obvious implementations are both wrong at this scale: materialising
(household × day × sub-commodity) is ~32 million rows, and a per-sample Python lookup is
~35 million dict hits per epoch. Instead purchase days are stored once as a globally
sorted key array with `key = group_id * 1024 + day`, where `group_id` indexes a
(household, sub-commodity) pair. One vectorised `np.searchsorted` answers the question for
a whole batch, and because the stride (1024) exceeds any day index the ordering never
crosses a group boundary.

---

## 5. Stage 3 — `data.py :: build()` → `v3_index*.npz`

Everything above is still a transaction table. The model's normaliser sums over **every
subset of the store's assortment**, so it needs the assortment as an indexed structure,
and it needs each observed basket expressed in assortment-local coordinates.

Two facts shape the layout:

- **The assortment is ragged.** Categories at the median store hold 9 products; the largest
  holds 182. Padding every category to the maximum would waste ~20× the work. So items are
  kept in one flat array with a row index, and only the *category* axis is padded — it is
  short (≤ 188) and a scan over it is unavoidable.
- **A purchased product is a position within its (store, category) block**, not a global id,
  because that is what the elementary symmetric polynomials index.

### Arrays in `v3_index_affinity.npz`

```
  store_cat_ptr   [32201]        where each (store, category) block starts and ends
  store_items     [590551]       product ids, grouped by (store, category)
  item_slot       [115, 5455]    position of a product inside its block, -1 if not carried
  n_store 115   n_cat 280   n_item 5455   n_user 2066

  trip_user / trip_store / trip_day / trip_week / trip_split / trip_nlines   [198690]
  line_ptr        [198691]       CSR-style pointer into the line arrays
  line_item / line_cat / line_slot / line_units   [1558093]
```

**Integrity is asserted, not assumed.** Every purchased product must lie in its store's
assortment — if it does not, the likelihood would be evaluated on a support that excludes
the observed basket and would be silently wrong. `build()` checks this and reports any
repair where the assortment definition and the transaction log disagree.

### Which partition — this matters

The **category partition defines the ragged row structure**, so it sets `C`, the cache
file, and the shape of `rho_c`. Checkpoints built under different partitions are **not
comparable**, and loading one under the wrong partition fails with a `rho_c` shape
mismatch.

| env | cache file | `n_cat` | notes |
|---|---|---:|---|
| *(default)* | `v3_index.npz` | 188 | merchandiser commodities |
| `V3_AFFINITY=1` | `v3_index_affinity.npz` | **280** | co-purchase affinity partition — **what the current runs use** |
| `V3_PARTITION=items_subcom.parquet` | `v3_index_items_subcom.npz` | 758 | sub-commodity granularity |

The affinity partition is rebuilt around what shoppers buy together rather than how a
merchandiser filed things, so the exact `rho_c` term carries complementarity. Any parquet
of `(item_id, cat_id)` can be named via `V3_PARTITION`, so a partition can be swapped
without a code change.

**All current results require `V3_AFFINITY=1`.** Omitting it silently selects a
different model.

### Rebuilding

```python
from data import build
D = build(force=True)       # deletes nothing; rewrites the cache for the active partition
```

Takes a few minutes. It is cached, so every other script just calls `build()`.

---

## 6. Stage 4 — `Features` and `Batcher` → tensors

Nothing here is written to disk; it is assembled per minibatch.

`features.py :: Features` memory-maps the four context sources and answers batch queries by
vectorised `searchsorted`:

| source | gives |
|---|---|
| `log_price_dev.npy` | `dlp` — each item's log-price deviation on the trip's day |
| `store_price.npz` | store-level price deviation where observed |
| `state.npz` | days since the household last bought the item's sub-commodity |
| `promo.npz` | display and mailer flags for (item, store, week) |

`fit.py :: Batcher.make(trip_ids)` returns:

```
  ix     RaggedIndex   the assortment slots for these trips, grouped by (trip, category)
  ctx    dict          per-slot context: dlp, dlp_bar, disp, mail, week, store, recency
  lctx   dict          the same, restricted to the purchased lines
  hh     [B]           household id per trip
  li, lt, lc, lq       purchased item / trip / category / units
```

`ix` plus `ctx` is exactly what `RaggedModel.b_flat` and the normaliser consume.

---

## 7. Where to find things

| you want | look in |
|---|---|
| raw CSVs | `../dunnhumby_The-Complete-Journey/…CSV/` (or `$NF_RAW_DIR`) |
| cleaned transaction log | `data/tx.parquet` |
| **model input** | **`basket_input/`** |
| the built index | `basket_input/v3_index_affinity.npz` |
| sizes, splits, cuts | `basket_input/meta.json` |
| trained checkpoints | `out/`, `codex_checked/out/` |
| the price derivation and its audit | `PREPROCESSING.md` §1 |
| the older paper pipeline | `PREPROCESSING.md`, and `model_input/` |

### Derived inputs that are *not* part of the main chain

Built by separate one-off scripts and read only when the corresponding flag is passed:

| file | built by | used by |
|---|---|---|
| `items_affinity.parquet` | co-purchase clustering | `V3_AFFINITY=1` |
| `v3_degree.npy` | co-purchase degree per product | `--phi-deg` |
| `v3_phimask_*.npy` | interaction support masks | `--phi-mask` |
| `v3_beta_target.npz` | `beta_target.py` | `--beta-cal-w` |

---

## 8. Reproducing from scratch

```bash
export NF_RAW_DIR=/path/to/dunnhumby_The-Complete-Journey CSV   # optional

python scripts/pipeline/01_build_base.py        # raw CSVs -> data/
python scripts/pipeline/22_basket_data.py       # data/    -> basket_input/
python scripts/pipeline/23_promo_data.py        # adds promo.npz

# stage 3 is automatic on first use:
cd scripts/v3_codex
V3_AFFINITY=1 python -c "from data import build; build()"
```

Stages 1–2 take tens of minutes, mostly parsing the 696 MB `causal_data.csv`. Stage 3 takes
a few minutes and is then cached.
