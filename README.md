# Basket model — dunnhumby "The Complete Journey"

A probabilistic model of the **whole basket**: it scores a set of products jointly, answers
price counterfactuals, and can be sampled from, so it doubles as a retail environment for
coupon and markdown policies.

This branch contains **only** what is needed to reproduce and evaluate the shipped model.
Superseded experiments live on the `adaptive-quadrature` branch.

---

## 1. What the model is

A basket `S` drawn from a store's assortment has energy

```
E(S) = sum_{j in S} b_j(x)                 utility: price, promotion, season, store, taste
     + sum_{j<k in S} phi_j' phi_k          pairwise interaction (low rank)
     - sum_c rho_c C(n_c, 2)                within-category complementarity
     - rho_0(|S|)                           basket-size potential
```

with `P(S) ∝ exp E(S)`. The normaliser sums over **every subset of the assortment**, which
is tractable because the quadratic term is linearised by Hubbard–Stratonovich —
`exp(½‖v‖²) = E_z[exp(v'z)]` with `v_S = sum_{j in S} phi_j` — leaving, for each draw of
`z`, a product over independent items that elementary symmetric polynomials evaluate
exactly. The `z` integral is done on a Smolyak sparse grid (`q=8`, 681 nodes at `Kz=4`).

`log f(z)` is the cumulant generating function of `v_S`, so `∇log f(0) = E[v_S]` and
`∇²log f(0) = Cov(v_S)`. The quadrature is valid while
`lambda_max = sum_j pi_j ‖phi_j‖² < 1` — a **budget**, which is why the interaction is
deliberately sparse (30 products, rank 4).

---

## 2. Results (run413, validation split)

| quantity | model | reference |
|---|---|---|
| set log-likelihood / basket | **−50.336** | −51.572 at the start of this work |
| MRR (complete-the-basket) | **0.0761** | 0.0703 |
| median rank | **380** | 468 |
| MRR@5 / Recall@5 | 0.0660 / 0.1234 | 0.0575 / 0.0918 |
| MRR@20 / Recall@20 | 0.0720 / 0.1835 | 0.0667 / 0.1772 |
| own-price elasticity | **−0.681** | **−0.7725** measured from the data |
| cross-price elasticity (same sub-commodity) | **+0.044** | **+0.1351** measured from the data |
| aggregate elasticity | −0.127 | −0.121 measured from the data |
| E[n] per basket | 8.0 | 7.82 observed |

The two elasticity references are estimated independently of the model, from an item-week
panel with item fixed effects and display/mailer controls (`docs/` describes the panel).
They are targets the model is checked against, not quantities fitted to.

The ranking rows are exactly what `src/run/evaluate.sh` prints for the shipped checkpoint
on 316 held-out cases; the likelihood row is the training-time validation figure over 384
trips. Small differences between the two are sample size, not disagreement.

**Known gaps**, stated plainly:

* Cross-price reaches a third of its target. The reference group's width is the knob —
  measured, cross-price is −0.162 (trip-wide reference), +0.044 (category), +0.502
  (sub-commodity), and the target sits at an effective width of ~85 products, between the
  last two. Closing it needs a blended reference, which buys elasticity fidelity and no
  likelihood.
* Household price sensitivity carries ~0.1% of ranking: taste personalises, price does not.
  Coupon targeting needs the second.
* Segment-level generation reproduces basket size well but compresses between segments,
  with 1–4/10 top-product overlap.
* run413 was still improving when its iteration budget ran out (+0.008 nats/1,000 it).

---

## 3. Repository layout

```
src/pipeline/     raw dunnhumby CSVs  ->  data/  ->  basket_input/
  01_build_base.py     transactions and the reconstructed shelf price
  22_basket_data.py    the modelling universe (products, households, splits)
  23_promo_data.py     display and mailer panel

src/basket/       the model, its training loop, and every evaluation
  ragged.py            energy, normaliser, sampler, projections     <- the kernel
  fit.py               training loop, objectives, schedules, logging
  data.py              builds/caches the ragged assortment index
  features.py          price, promotion, recency and store features
  evalall.py           checkpoint loading -- ALWAYS load through this
  downstream.py        counterfactual, personalisation, segmentation, generation
  eval_mrr_cutoffs.py  MRR and MRR@k
  stamp_flags.py       backfills model_flags onto pre-2026-08 checkpoints
  pairmask.py          picks the 30 interaction products by co-purchase lift
  phi_spectral_init.py spectral placement of phi from the empirical log-lift matrix
  beta_target.py       per-product price-coefficient calibration target

src/run/          the three things you actually run
  prepare.sh  train.sh  evaluate.sh

docs/             DATA_TO_MODEL_INPUT.md   raw CSVs -> tensors, every file explained
                  PREPROCESSING.md         the price reconstruction and its audit
```

Everything under `basket_input/`, `data/` and `out/` is regenerated, not committed — except
two small auxiliary inputs (`v3_beta_target.npz`, `v3_phimask_lift30.npy`) that the training
flags reference, which are committed so a fresh clone can train without re-running the
co-purchase scans.

**Scripts in `src/basket/` and `src/pipeline/` resolve paths as `<script dir>/../..`.**
They must stay two levels below the repository root, and `src/basket/` must stay one
directory, because its modules import each other by bare name.

---

## 4. Running it on a new machine

### Prerequisites

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt              # numpy, torch, pandas, pyarrow
```

Download **dunnhumby "The Complete Journey"** and place the CSV folder next to this
repository, or point `NF_RAW_DIR` at it:

```bash
export NF_RAW_DIR="/path/to/dunnhumby_The-Complete-Journey CSV/"
```

`transaction_data.csv` (142 MB), `product.csv`, and `causal_data.csv` (696 MB) are the ones
that are read. The raw data is never modified.

### Step 1 — build the model inputs (~35 min, once)

```bash
./src/run/prepare.sh
```

Produces `data/`, then `basket_input/` (5,455 products, 2,066 households, 115 stores,
199,345 baskets), then the cached ragged index `basket_input/v3_index_affinity.npz`.

### Step 2 — train

```bash
OMP_NUM_THREADS=8 ./src/run/train.sh my_run 40000
```

About 0.7 s/iteration on 8 threads, so ~8 h for 40,000 iterations. Writes
`out/v3_my_run.pt`, `out/v3_my_run_best.pt`, and one JSON object per evaluation to
`out/my_run_metrics.jsonl` (37 fields including per-block gradient norms, `lam_max`,
elasticity and wall clock — a stall can be told from a slowdown after the fact).

### Step 3 — evaluate

```bash
./src/run/evaluate.sh v3_my_run_best.pt
```

Prints MRR/MRR@k, then price counterfactuals, personalisation ablations, customer
segmentation and segment-level generation, and writes a JSON alongside the checkpoint.

---

## 5. Three things that will silently give you wrong numbers

These each cost real debugging time. They are all guarded now, but know them.

**`V3_AFFINITY=1` is load-bearing.** It selects the 280-category co-purchase affinity
partition. Omitting it silently builds a *different model* (188 merchandiser commodities),
and checkpoints are not comparable across partitions — `rho_c` changes shape. All three
scripts in `src/run/` export it.

**Never load a checkpoint with a bare `load_state_dict`.** Use `evalall.load_any`. A
checkpoint carries `model_flags` recording things the tensors cannot express:

* `price_soft` — whether `gamma`/`beta` are the price coefficients or softplus pre-images.
  `gamma = +0.0207` is valid under both readings; guessing wrong uses
  `softplus(0.0207) = 0.7036`, **34× too large**, and reported MRR 0.0044 for a model whose
  training log said 0.0705.
* `price_ref` — whether `dlp` is referenced to the trip's whole assortment or its category.
  Scoring a category-referenced model against a trip mean deletes its substitution channel.
* `poly_degree` — see below.

**The polynomial truncation degree is a numerical-validity constraint, not a speed knob.**
`exp(-rho_c C(n,2))` at `rho_c = -0.337, n = 120` is `10^1045`, past float64's `10^308`, so
the untruncated per-row polynomial returns NaN. Degrees just below overflow are *finite and
meaningless*: at degree 64, `sum_j pi_j = 120.00 = n_max` ("every product certain") when the
truth is 7.6. The safe degree depends on `rho_c` and therefore on the checkpoint.
`downstream.py::safe_degree` calibrates **upward** from the largest per-category count in
the data (26) — calibrating downward from the untruncated polynomial cannot work, because
that reference is the overflowing one.

---

## 6. How the shipped model was reached

`train.sh` reproduces the configuration, but the published checkpoint came from three staged
runs, because the price block converges far more slowly than the rest of the model.

| run | change | set LL | own-price | cross-price |
|---|---|---|---|---|
| baseline | — | −51.572 | −0.118 | — |
| run409 | `--price-soft` with the projection fixed for the unconstrained form | −50.523 | −0.313 | −0.162 |
| run411 | `--kappa-init 44` from the data-implied elasticity | −50.416 | −0.737 | −0.162 |
| run413 | `--price-ref category` | **−50.336** | −0.681 | **+0.044** |

`kappa` moves ~1.4 units per 1,000 iterations even at 20× the structural rate — its gradient
is small and sign-noisy over 24-trip minibatches, so Adam averages it away. **Where it
starts decides where it ends**, which is why `--kappa-init` exists: the own-price elasticity
is identified from the data without the model, and the likelihood's own optimum
independently agrees with it.

To continue from the shipped checkpoint rather than train from scratch, add
`--resume out/<ckpt>.pt --fresh-sched 1` and set `--lr` to the rate the previous run
*ended* at. The scheduler counter is per-process: resuming without `--fresh-sched` restores
the old rate, and resuming *with* new milestones discards decays the previous run applied.
Both mistakes cost a run here.
