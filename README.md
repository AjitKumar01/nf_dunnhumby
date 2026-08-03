# Consumer choice on dunnhumby "The Complete Journey"

Two things live here.

1. **A port** of Donnelly, Ruiz, Blei & Athey (2023), *Counterfactual Inference for
   Consumer Choice Across Many Product Categories*, from the Che et al. single-store
   scanner panel to dunnhumby's loyalty panel — including a cross-check against the
   authors' own C++.
2. **A replacement model**, built after the exploration showed the paper's three
   structural assumptions do not hold on this data. It models whole baskets rather
   than one item per category, learns product interactions and household state, and
   its item embeddings recover the retailer's sub-commodity hierarchy without ever
   being shown it.

The second is the current work. Start at **[`DATA_EXPLORATION.md`](DATA_EXPLORATION.md)**
and **[`BASKET_MODEL.md`](BASKET_MODEL.md)**.

## Headline results

| | paper's port (`nf`) | basket model (`one`) |
|---|---|---|
| items / categories | 560 / 56 | **5,455 / 188** |
| observations | 66,637 | **1,566,063** |
| days | 172 (Sun + Mon only) | **712 (all)** |
| multiple items per category | assumed away | modelled |
| product interactions | IIA within category | learned, low-rank |
| household state | none | recency per sub-commodity |
| **embedding recovers sub-commodity** | **1.0× chance** (AUC 0.379) | **70.6× chance** (AUC 0.823) |
| price coefficient | — | **+0.840**, against a reduced-form 0.844 |
| strict price placebo retains | 20% of the real effect | **0.7%** |

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# The dunnhumby CSVs default to a sibling directory; point elsewhere with:
export NF_RAW_DIR="/path/to/dunnhumby_The-Complete-Journey CSV"

bash scripts/run_all.sh          # ~2.5 h end to end; 22 model fits
```

Just the basket model, without the paper replication (~1 h):

```bash
cd scripts
python3 21_basket_eda.py && python3 22_basket_data.py && python3 25_basket_placebo.py
python3 23_basket_model.py --label one --tie-context --K 64 \
        --l2 1e-2 --l2-price 1e-4 --lr 0.005 --lr-decay --iters 12000
python3 24_embedding_eval.py --primary one
```

Results land in `out/` (`evaluation_summary.csv` for the port, `embedding_eval.json`
and `price_causal.json` for the basket model); figures in `figures/`.
**[`RUNNING.md`](RUNNING.md) is the full manual** — prerequisites, per-stage
timings, troubleshooting, and recipes for fitting variants.

## Pipeline

```
run_all.sh                  everything below, in dependency order (~20 min,
                            plus ~30 for the optional C++ cross-check)

01_build_base.py            clean transactions, unit prices, trips, price panels
02_select_sample.py         pair-week window, household filter, category filters
03_make_model_inputs.py     emit the exact bemb_loc input files + counterfactual events
04_extras.py                display / mailer / coupon-eligibility panels (dunnhumby only)
05_train_nf.py              fit both stages (PyTorch, variational Bayes + reparam. SGD)
06_hyperparam_sweep.py      grid search, selected on validation *price-change* weeks
07_evaluate.py              the paper's evaluation battery
08_data_report.py           descriptive statistics -> out/data_report.md
09_counterfactual_checks.py model-free validation of the estimated heterogeneity
10_price_definition_audit.py  audit of the price reconstruction vs the user guide
11_placebo_tests.py         price-endogeneity placebo tests (paper sec. 5)
12_preprocessing_figures.py all figures for PREPROCESSING.md
13_placebo_followup.py      per-category verdicts + fit on the clean subset
14_verify_model.py          analytic checks, degenerate cases, parameter recovery
15_cpp_crosscheck.py        this port vs the authors' C++ on identical files
16_inspect_embeddings.py    what the latent vectors encode; ablations
17_store_diagnostics.py     does pooling 561 stores cost anything
18_substitution_eda.py      re-asks category selection for a substitution kernel
19_substitution_test.py     does the kernel learn similarity it was never shown
20_simulate.py              the fitted model as a simulator; emits transitions
run_bemb_loc.sh             run the authors' C++ stage-1 binary on the same files

--- the basket model (replaces 02/03/05 rather than extending them) ---
21_basket_eda.py            exploration behind DATA_EXPLORATION.md
22_basket_data.py           full 711-day basket dataset, no unit-demand filter
23_basket_model.py          basket model: interactions, household state, price
24_embedding_eval.py        do the embeddings recover sub-commodity clusters?
25_basket_placebo.py        price-endogeneity placebos on the 188-category catalogue
26_price_causal.py          structural placebo; fit where the counterfactual lives
```

## Where to start

| document | what it is | read it if |
|---|---|---|
| **`DATA_EXPLORATION.md`** | what the data looks like once the paper's assumptions are dropped: unit demand, category independence, no state — plus whether the price variation is exogenous | **start here for the basket model** |
| **`BASKET_MODEL.md`** | the replacement model — 5,455 items, 188 categories, product interactions, household state — its embedding test, its placebos, and what it gives up | you want the current results |
| **`FLOW.md`** | the end-to-end map of the *paper's* port: vocabulary, one real shopping trip followed through, every file and row count | you want the paper replication |
| `RUNNING.md` | how to install, run and debug the pipeline | you want to execute it |
| `PREPROCESSING.md` | every data decision, with figures and the evidence for each | you want to know why the data looks the way it does |
| `REPORT.md` | what the paper does, and how the model performs here | you want the results |
| `VERIFICATION.md` | what has and has not been checked, including the cross-check against the authors' own C++ | you want to know how much to believe |

**Run the placebo tests before believing any elasticity** — stage 11 for the paper's
port, stage 25 for the basket model. They are what decide whether the identifying
variation is credible, and `run_all.sh` puts each of them before the fits they
qualify.

## What maps onto what

| Paper | dunnhumby |
|---|---|
| one store, 23 months | 561 stores, 102 weeks; prices pooled to chain level |
| UPC | `PRODUCT_ID` |
| "category" (unit of substitution) | `COMMODITY_DESC` |
| "class"/"subclass" (held out of the model) | `SUB_COMMODITY_DESC`, `BRAND`, `MANUFACTURER` |
| trip = household × calendar day | same |
| price change at Tuesday midnight → Tue/Wed sample | price change at the Sunday→Monday week boundary → Sun/Mon sample |
| stock-out feed | not available; and store assortment differs (the median store sells 67% of the retained catalogue) |
| placebo tests: 13 of 123 categories fail at 1% | port: 34 of 56 fail at least one of six, 7 fail the randomised one. Basket model: **5 of 160** fail a strict placebo |
| — | in-store display and weekly mailer, product × store × week |
| — | coupon campaigns: household × product × date eligibility, and realised redemptions |

The Sunday→Monday boundary is not an assumption: `WEEK_NO` runs Monday→Sunday, and
the probability that a product's price moves between consecutive days is **51.9%**
across that boundary versus **26.3%** within a week. That is the same
identification device the paper uses, on a different day pair.

## Model input files

### `basket_input/` — the basket model (`22_basket_data.py`)

| file | what it is |
|---|---|
| `baskets.parquet` | one row per (basket, item) purchase, with the train/validation/test label |
| `items.parquet` | `item_id` ↔ `PRODUCT_ID`, plus the labels the model never sees (sub-commodity, brand, manufacturer, department) |
| `log_price.npy`, `log_price_dev.npy` | item × day log price, raw and centred within item |
| `state.npz` | sorted purchase-day keys for the vectorised "days since this household last bought this sub-commodity" lookup |
| `meta.json` | sizes, split boundaries, the frequency cut |

### `model_input/` — the paper's port

Formats follow `src/bemb_loc/emb_io.hpp` exactly (fscanf, tab separated, integer ids).

| file | columns |
|---|---|
| `train.tsv`, `validation.tsv`, `test.tsv` | user, item, session, units |
| `item_sess_price.tsv` | item, session, price (complete item × session grid) |
| `itemGroup.tsv` | item, category |
| `userGroup.tsv` | user, spend quintile (only needed by the `hpf` binary) |
| `sess_days.tsv` | session, pair-week, weekday (0=Sun, 1=Mon), hour |
| `obsUser.tsv` | user, 9 demographic columns |
| `obsItem.tsv` | item, 4 product columns |
| `item_sess_display.tsv`, `item_sess_mailer.tsv` | item, session, promotion intensity |
| `coupon_campaigns.npz`, `user_item_sess_coupon.tsv` | coupon eligibility |
| `events.csv` | item × pair-week flags for own-/cross-price-change events |
| `id_maps/` | model ids ↔ dunnhumby ids |

## Two things the shipped C++ cannot do

1. **The category stage.** `bemb_loc` stores prices as one item × session matrix —
   prices common to all users (its own README says so). The paper's stage 2 feeds
   the inclusive value `IV_ict` into that slot, and `IV` varies across households.
   That run used the user-varying TTFM build, which is not in this repository.
2. **More than one time-varying item attribute.** Price is the only such slot, so
   display, mailer and coupon eligibility have nowhere to go.

`05_train_nf.py` implements both stages and the extra attributes; `run_bemb_loc.sh`
still drives the original binary for stage 1 so the two can be cross-checked.

## Two changes to the paper's specification that turned out to be necessary

These concern the port (`05_train_nf.py`), not the basket model.

1. **Seed the price factors away from zero.** `gamma_i . lambda_j` is bilinear; if both
   factors start at zero the gradient in each is zero and the model learns no price
   response at all: the fitted price coefficient ends at 0.014 instead of 0.643.
   `bemb_loc` exposes `-meangamma` and `-meanbeta` for this; `--price-prior-mean` is
   the equivalent here.
2. **Centre the inclusive value.** Fed raw, `IV_ict`'s level is collinear with
   `vartheta_i . beta_c`, and the nesting coefficient collapses to **0.08** — the
   inclusive value carrying almost no weight at all (on an earlier sample it went
   outright negative, −0.16). Subtracting the household-category mean leaves only the
   part that moves with prices; the coefficient then estimates at **1.00**.
   `--no-center-iv` restores the raw version.

## Provenance and licensing

* **Paper**: Donnelly, Ruiz, Blei & Athey (2023), *Counterfactual Inference for
  Consumer Choice Across Many Product Categories*, Quantitative Marketing and
  Economics. Not redistributed here.
* **Authors' code**: <https://github.com/rodonn/nested-factorization>, MIT
  licensed. Not vendored — `scripts/run_bemb_loc.sh` compiles it from a clone you
  supply, and `15_cpp_crosscheck.py` compares against it.
* **Data**: dunnhumby *The Complete Journey*, free on registration at
  <https://www.dunnhumby.com/source-files/>. Redistribution is governed by
  dunnhumby's terms, so no raw or derived transaction data is committed here —
  `.gitignore` excludes `data/`, `model_input/`, `basket_input/` and the fitted
  parameters, all of which `scripts/run_all.sh` rebuilds. The small text results in
  `out/` and the figures are aggregate statistics and are committed.
