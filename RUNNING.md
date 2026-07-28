# Running the code

Everything in this repository is built from one input — the dunnhumby *Complete
Journey* CSVs — by one command. This file tells you how to get from a fresh clone
to the numbers quoted in `REPORT.md`, and what to do when something goes wrong.

If you want to know *what* the pipeline does rather than *how to run it*, read
[`FLOW.md`](FLOW.md) first. This document assumes you just want it to run.

---

## 1. What you need

| | |
|---|---|
| **Python** | 3.11 or newer (developed and verified on 3.13.4) |
| **Disk** | ~800 MB for the raw CSVs, ~200 MB for everything the pipeline generates |
| **RAM** | 8 GB is comfortable. The largest single object is the 2.6 M-row transaction table |
| **Time** | ~20 minutes for the full run on a laptop CPU, or ~50 with the optional C++ cross-check (see §6) |
| **GPU** | Not needed. Optional, and see the warning in §8 |
| **A C++ compiler + GSL** | Only for the optional cross-check against the authors' own binary (§9) |

No internet access is needed once the CSVs are downloaded.

## 2. Get the data

The dataset is free but requires a registration form:
<https://www.dunnhumby.com/source-files/> → *The Complete Journey*.

Unzip it so that the eight CSVs sit together in one directory:

```
dunnhumby_The-Complete-Journey/
├── dunnhumby - The Complete Journey User Guide.pdf
└── dunnhumby_The-Complete-Journey CSV/
    ├── transaction_data.csv     2,595,732 lines — the receipts
    ├── product.csv              the product hierarchy
    ├── hh_demographic.csv       demographics for 801 of the 2,500 households
    ├── causal_data.csv          36.8 M lines — in-store display and weekly mailer
    ├── campaign_desc.csv        coupon campaign dates
    ├── campaign_table.csv       which household got which campaign
    ├── coupon.csv               which products each coupon covers
    └── coupon_redempt.csv       realised redemptions
```

**Where the code looks for it.** By default, a sibling of this repository:

```
Causal/
├── dunnhumby_The-Complete-Journey/     ← the CSVs
├── nested-factorization/               ← the authors' code (optional, §9)
└── nf_dunnhumby/                       ← this repository
```

If your copy lives anywhere else, set one environment variable — nothing else in
the code has an absolute path:

```bash
export NF_RAW_DIR="/path/to/dunnhumby_The-Complete-Journey CSV"
```

Check it resolves before you start:

```bash
python3 -c "import os; p=os.environ['NF_RAW_DIR']; print(os.path.exists(os.path.join(p,'transaction_data.csv')))"
```

## 3. Install

```bash
git clone <this repo> && cd nf_dunnhumby
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Six packages: numpy, pandas, pyarrow, scipy, matplotlib, torch. Nothing is
compiled from source and nothing needs a GPU build of torch.

## 4. Run it

```bash
bash scripts/run_all.sh
```

That is the whole thing: it reads the CSVs, builds the sample, runs the placebo
tests, fits four models, evaluates them, regenerates every figure, and runs the
verification suite. Roughly 20 minutes, or 50 if GSL is installed and the C++
cross-check runs too. Progress prints as it goes; per-stage logs land in
`out/log_*.txt`.

Two useful variants:

```bash
SKIP_BASE=1 bash scripts/run_all.sh    # reuse data/tx.parquet — saves the slowest read
bash scripts/run_all.sh 2>&1 | tee run.log
```

**Order matters, in one place especially.** The placebo tests (stage 11) run
*before* any model is fitted. They are what decides whether the Sunday/Monday
price-change window identifies anything at all; if they fail, no elasticity from
stages 05–09 is worth reading. That is why they sit where they do, and you should
not reorder them out of convenience.

## 5. What you get

```
data/                cleaned transactions, price panels, the selected sample   (~43 MB)
model_input/         the files the model reads — the exact bemb_loc format     (~33 MB)
model_input_clean/   the same, restricted to categories that pass the placebo  (~7 MB)
out/                 fitted parameters (.pt), evaluation tables, logs          (~110 MB)
figures/             every figure in PREPROCESSING.md and VERIFICATION.md      (~2 MB)
```

All five are regenerated from scratch by `run_all.sh` and all five are
git-ignored, except the small text results in `out/` (`*.csv`, `*.json`,
`out/data_report.md`, `out/log_*.txt`) which are committed so the reports can be
read without a rerun.

The headline results table is `out/evaluation_summary.csv`; the descriptive
statistics are `out/data_report.md`.

## 6. Stage by stage

Run individually if you want; each stage only needs the ones above it. Times are
wall-clock on an Apple M5 Pro, CPU only.

| | script | time | what it produces |
|---|---|---|---|
| 01 | `01_build_base.py` | 6 s | cleaned transactions, unit prices, trips, two price panels |
| 02 | `02_select_sample.py` | 37 s | the pair-week window, the household filter, the five category filters |
| 03 | `03_make_model_inputs.py` | 1 s | `model_input/` — the files the model actually reads |
| 04 | `04_extras.py` | 6 s | display / mailer / coupon panels (dunnhumby-only extras) |
| 10 | `10_price_definition_audit.py` | 5 s | audit of the price reconstruction against the user guide |
| 11 | `11_placebo_tests.py` | 3.5 min | **run this before believing any elasticity.** `run_all.sh` runs it twice (a second time under the paper's own specification), so budget 5 min |
| 05 | `05_train_nf.py` | 15 s – 4 min per model | fits both stages. `nf` 1 min, `logit` 15 s, `nf_promo` 1.5 min, `nf_nopool` 4 min (it carries a latent vector per household *and* category) |
| 07 | `07_evaluate.py` | 26 s | the paper's evaluation battery |
| 13 | `13_placebo_followup.py` | 2 s | per-category placebo verdicts, fit re-aggregated on the clean subset |
| 09 | `09_counterfactual_checks.py` | 18 s | model-free validation of the estimated heterogeneity |
| 08 | `08_data_report.py` | 1 s | `out/data_report.md` |
| 12 | `12_preprocessing_figures.py` | 34 s | the nine figures in `PREPROCESSING.md` |
| 14 | `14_verify_model.py` | 83 s | gradients, degenerate cases, parameter recovery |
| 15 | `15_cpp_crosscheck.py` | **~35 min** | this port vs the authors' C++ on identical files. Almost all of it is the C++ binary itself; needs GSL, and `run_all.sh` skips it silently without one |
| 16 | `16_inspect_embeddings.py` | 11 s | what the latent vectors encode, plus ablations |
| 17 | `17_store_diagnostics.py` | 6 s | whether pooling 561 stores costs anything |
| — | clean-subset retrain + restore | 2.5 min | refits on placebo-surviving categories, then puts the default sample back |
| 06 | `06_hyperparam_sweep.py` | ~30 min | **not in `run_all.sh`** — the grid search that picked K=40 |

Stage 06 is excluded from the default run because its only output is the choice
of hyperparameters, and that choice is already baked into the defaults of stage
05. Run it if you want to re-derive them.

Stage 15 dominates the wall clock. `ITERS=500 bash scripts/run_bemb_loc.sh` cuts it
to about 8 minutes and still covers the peak, which lands near iteration 300.

## 7. Checking it worked

After a full run, these should hold. Small last-digit drift across platforms is
normal — torch's CPU reductions are not bit-identical across architectures — but
anything that moves the third decimal place means something is wrong.

```bash
column -s, -t out/evaluation_summary.csv | cut -c1-120
```

**Sample sizes** (printed by stages 02 and 03, and tabulated in `out/data_report.md`):

| | |
|---|---|
| households / items / categories | 2,084 / 560 / 56 |
| sessions | 172 days = 86 pair-weeks |
| trips | 49,729 |
| category purchases | 66,638 |
| train / validation / test | 46,431 / 6,471 / 13,736 |
| price grid | 96,320 = 560 × 172 |

**Held-out log-likelihood** (`out/evaluation_summary.csv`, higher is better):

| model | test log-lik | test MSE |
|---|---|---|
| `nf` | −4.239 | 0.938 |
| `nf_promo` | −4.242 | 0.935 |
| `nf_nopool` | −4.279 | 0.937 |
| `logit` | −4.715 | 0.980 |

The three latent models must beat `logit` by roughly 0.45 nats, and `nf`/`nf_promo`
must beat `nf_nopool`. `nf` and `nf_promo` sit within 0.004 of each other, so which
of those two prints first is not meaningful.

**Two sanity numbers that catch the failure modes documented in `README.md`:**

* the nesting coefficient (`nesting_coef_mean`) should be ≈ **1.0**, not negative —
  a negative value means the inclusive value is not being centred;
* the median own-price elasticity should be ≈ **−1.2**, not ≈ 0 — a near-zero value
  means the bilinear price term collapsed and `--price-prior-mean` is not taking
  effect.

**Placebo tests** (`out/placebo_summary.json`): the real price series should give a
typical effect near **−0.61** with ~75% of categories significant at 1%; the fully
randomised placebo should collapse that to about **+0.01** with ~12% significant.
If the fake looks like the real thing, the identification window is broken.

If the placebo tests report failures, that is not a bug: read §9 of
`PREPROCESSING.md`. Ten of the 56 categories fail a fully randomised placebo, and
that finding is a result, not an error.

## 8. Common problems

**`FileNotFoundError: .../transaction_data.csv`**
`NF_RAW_DIR` is unset and the CSVs are not a sibling of the repo. See §2.

**`FileNotFoundError: data/tx.parquet`**
You ran a later stage before stage 01, or used `SKIP_BASE=1` on a clean checkout.
Run `python3 scripts/01_build_base.py` first.

**`ModuleNotFoundError: pyarrow`**
pandas needs it for parquet. `pip install -r requirements.txt`.

**The run is much slower than the table in §6.**
Torch defaults to one thread in some environments. `export OMP_NUM_THREADS=8`.

**MPS / CUDA gives different numbers, or errors.**
Every stage that fits a model defaults to `--device cpu` deliberately. On this
sample the model is small enough that a GPU is not faster, and MPS changes
results in the last decimal because its float32 reductions differ. `--device
mps`, `--device cuda` and `--device auto` all work if you ask for them; use
`cpu` for anything you intend to quote.

**`gsl-config not found`**
Only stage 15 needs it, and `run_all.sh` skips that stage cleanly if it is
missing. `brew install gsl` (macOS) or `apt install libgsl-dev` (Debian) if you
want the cross-check.

**Stage 15 takes forever.**
It is meant to: most of that time is the authors' C++ binary, which is the point
of the stage. `ITERS=500 bash scripts/run_bemb_loc.sh` cuts it to ~8 minutes and
still covers the peak. Stages 14–17 are the only ones you can skip entirely
without affecting any number reported in `REPORT.md` — they verify the code
rather than produce results.

## 9. Optional: the authors' C++ binary

`15_cpp_crosscheck.py` fits the authors' own stage-1 implementation on exactly
the files in `model_input/` and compares the trajectory to this port's. It needs
their source tree:

```bash
git clone https://github.com/rodonn/nested-factorization.git ../nested-factorization
brew install gsl
bash scripts/run_bemb_loc.sh          # compiles src/bemb_loc/emb.cpp and runs it
python3 scripts/15_cpp_crosscheck.py  # writes figures/cpp_crosscheck.png
```

Override `NF_NF_SRC` if the clone is elsewhere. What the comparison can and
cannot establish is set out in `VERIFICATION.md` — in short, it validates stage 1
and says nothing about stage 2, which the shipped C++ cannot express.

## 10. Recipes

**Fit one model and stop.**

```bash
python3 scripts/05_train_nf.py --label mine --device cpu
```

The defaults are the selected configuration (K=40, Kp=20, price prior variance
0.25, 3000 iterations per stage), so this reproduces the `nf` row of the results
table. Add `--stage1-only` to skip the category stage.

**The baselines and ablations that appear in the report.**

```bash
python3 scripts/05_train_nf.py --label logit     --homogeneous          # no heterogeneity
python3 scripts/05_train_nf.py --label nf_nopool --no-pool              # each category alone
python3 scripts/05_train_nf.py --label nf_promo  --extras display mailer coupon
python3 scripts/07_evaluate.py --labels nf logit nf_nopool nf_promo --device cpu
```

**Restrict to categories that pass the placebo test.**

```bash
python3 scripts/02_select_sample.py --exclude-placebo-failures
python3 scripts/03_make_model_inputs.py --outdir model_input_clean
python3 scripts/05_train_nf.py --label nf_clean --indir model_input_clean
python3 scripts/07_evaluate.py --labels nf_clean --indir model_input_clean --tag _clean
```

Requires stage 11 to have run first — that is where the failure list comes from.
Afterwards, re-run stages 02–04 with no arguments to put the default sample back
in place, as `run_all.sh` does at the end.

**Change the sample.** Every filter in stage 02 is a flag with the default shown
by `--help`; `data/filter_audit.csv` records why each category was kept or
dropped, so you can see what a change did:

```bash
python3 scripts/02_select_sample.py --top-j 15 --min-cat-trips 400
python3 scripts/03_make_model_inputs.py   # rebuild the model files afterwards
```

**Two flags worth knowing about**, because turning them off breaks the model in
instructive ways documented in `README.md`:

| flag | what it does | what happens without it |
|---|---|---|
| `--price-prior-mean 0.5` | seeds the bilinear price term away from zero | no price response is learned at all; the price coefficient ends at 0.014 instead of 0.643 |
| (default: centred IV) | `--no-center-iv` feeds the raw inclusive value to stage 2 | the nesting coefficient collapses to 0.08 instead of ≈1.0 |

## 11. Reproducibility

Every stage takes `--seed` and defaults to a fixed one, so a rerun on the same
machine reproduces the same sample and the same fit. Two caveats:

- The train/validation/test split is drawn in stage 03. Re-running stage 03 with
  a different `--seed` invalidates every fitted model in `out/`.
- Floating-point reductions differ across CPU architectures and across torch
  versions. Expect agreement to about three decimal places in the
  log-likelihoods, not to the last bit.
