# Architecture

What each module does, the exact shapes flowing between them, and the facts that are not
guessable from reading the code. Mathematics is in [`THEORY.md`](THEORY.md); the raw-CSV
lineage is in [`DATA_TO_MODEL_INPUT.md`](DATA_TO_MODEL_INPUT.md).

**Contents**

1. [Module map](#1-module-map)
2. [Data flow and shapes](#2-data-flow-and-shapes)
3. [`RaggedIndex`: the central structure](#3-raggedindex)
4. [`RaggedModel`: parameters and methods](#4-raggedmodel)
5. [The forward pass, step by step](#5-the-forward-pass)
6. [Checkpoints and `model_flags`](#6-checkpoints)
7. [The training loop](#7-the-training-loop)
8. [Optimiser groups and the 400× scale spread](#8-optimiser-groups)
9. [Guards](#9-guards)
10. [Resuming](#10-resuming)
11. [Environment variables](#11-environment-variables)

---

## 1. Module map

```
src/basket/
  paths.py              Root discovery.  REPO is the checkout (where code lives); ROOT is
                        where artifacts go and follows NF_ROOT; RAW is an INPUT and stays
                        anchored to REPO.  Nothing else hardcodes a relative path, so the
                        tree can be rearranged and scripts run from any directory.

  data.py               build() -> the ragged assortment index, cached to
                        basket_input/v3_index*.npz.  Asserts every purchased product lies
                        in its store's assortment: otherwise the likelihood is evaluated on
                        a support excluding the observed basket and is silently wrong.

  features.py           Features: memory-maps price, store price, recency and promotion and
                        answers batch queries by vectorised searchsorted.

  ragged.py             THE KERNEL.  Energy, normaliser, marginals, sampler, projections.
  fit.py                Batcher, training loop, objectives, schedules, checkpointing.
  evalall.py            load_any() -- the ONLY correct way to load a checkpoint.

  downstream.py         counterfactual / personalisation / segmentation / generation
  eval_mrr_cutoffs.py   MRR and MRR@k on the fixed holdout
  elasticity_targets.py estimates the external targets the model is calibrated to
  diagnose_bucket_coverage.py   comparison against the historical truncated kernel
  stamp_flags.py        backfills model_flags onto pre-2026-08 checkpoints

  pairmask.py           selects the interaction products by co-purchase lift
  phi_spectral_init.py  places phi by eigendecomposing the empirical log-lift matrix
  beta_target.py        per-product price-coefficient calibration target

src/pipeline/           raw dunnhumby CSVs -> data/ -> basket_input/
src/run/                prepare.sh, train.sh, evaluate.sh
```

`src/pipeline/*` import `paths` from `src/basket/` through a two-line bootstrap; that is the
only cross-directory dependency.

---

## 2. Data flow and shapes

```
  dunnhumby CSVs                                  (read-only; NF_RAW_DIR)
        |  01_build_base.py         reconstructs shelf price from SALES_VALUE + discounts
        v
  data/tx.parquet (39 MB), trips, price_week, price_store_week
        |  22_basket_data.py + 23_promo_data.py
        v
  basket_input/   baskets.parquet   1,566,063 x 10
                  items.parquet     5,455 x 14
                  log_price.npy     [5455, 712]      log price by item and day
                  log_price_dev.npy [5455, 712]      deviation from the item's own mean
                  store_price.npz   244,880 cells    store-level deviations (0.53% of grid)
                  state.npz         1.36 M keys      days since last purchase, by sub-commodity
                  promo.npz         6.52 M keys      display and mailer flags
        |  data.py :: build()                        cached once, ~3 min
        v
  basket_input/v3_index_affinity.npz  (8.4 MB)
        |  features.py + fit.py :: Batcher           per minibatch, never written
        v
  RaggedIndex + ctx/lctx dicts  ->  RaggedModel
```

**Scale.** 5,455 products, 2,066 households, 115 stores, 712 days. Stage 2 reports 199,345
baskets; the index then drops held-out lines outside the training support, leaving
**198,690 trips and 1,558,093 lines**. Split temporally by household-week: train
`week < 83`, validation `83..90`, test `>= 91`.

---

## 3. `RaggedIndex`

Represents "the assortments of the trips in this batch", grouped into
$(\text{trip},\text{category})$ **rows**.

| field | shape | dtype | meaning |
|---|---|---|---|
| `item` | `[T]` | int64 | product id of every assortment slot |
| `row_of` | `[T]` | int64 | which row each slot belongs to |
| `item_trip` | `[T]` | int64 | which trip each slot belongs to = `row_trip[row_of]` |
| `row_trip` | `[n_rows]` | int64 | the trip each row belongs to |
| `row_cat` | `[n_rows]` | int64 | the category each row is |
| `row_size` | `[n_rows]` | int64 | products in that row |
| `n_rows` | scalar | | rows in the batch |
| `B` | scalar | | trips in the batch |

For a 24-trip batch: $T \approx 127{,}000$ slots, $n_{\text{rows}} \approx 5{,}176$.
`row_size` has **median 3, max 1,773**, and **median 128 weighted by where purchases fall** —
shoppers buy from the big rows, which is why the purchase-weighted figure is the one that
matters for the price reference (THEORY §11.4).

Everything downstream is a segment operation over `row_of` or `item_trip`.

**Context** arrives as two parallel dicts with identical keys — `ctx` over assortment slots
`[T]`, `lctx` over purchased lines `[L]` — so `energy()` and `log_Z()` score a product
identically. Keys: `dlp`, `dlp_bar`, `disp`, `mail`, `week`, `store`, `rec`.

---

## 4. `RaggedModel`

### Parameters

| name | shape | role | THEORY |
|---|---|---|---|
| `lam` | `[J]` | product intercept $\lambda_j$ | §2 |
| `theta` | `[N, 32]` | household taste $\theta_h$ | §2 |
| `phi` | `[J, Kz]` | interaction $\phi_j$, $K_z=4$, masked to 30 products | §4, §8 |
| `gamma` | `[N, Kp]` | household price loading, $K_p=8$ | §11.1 |
| `beta` | `[J, Kp]` | product price loading | §11.1 |
| `price_kappa` | scalar | idiosyncratic/aggregate split $\kappa$ | §11.5 |
| `rho_c` | `[C]` | within-category $\rho_c$, $C=280$ | §2, §9 |
| `rho_0_free` | `[nmax]` | size potential $\rho_0(n)$ | §2 |
| `xi` | `[S, Ks]` | store effect | §2 |
| `a_q`, `gamma_q`, `beta_q`, `log_r` | | units model | |

### Methods

| method | returns | notes |
|---|---|---|
| `b_flat(ix)` | `[T]` | per-slot utility $b_j$ |
| `b_at(it, trip, c)` | | same, at arbitrary positions; **all** price sites route through `price_g()` / `price_b()` so the elasticity machinery cannot disagree with the utility |
| `log_Z(ix)` | `[B]` | the normaliser, THEORY §4–6 |
| `pi_quad(ix)` | `[T]` | $\pi_j$ by autograd through the quadrature, THEORY (7.1). **Requires grad enabled** — wrapping in `no_grad` raises. Guarantees $\sum_j\pi_j=\mathbb{E}[n]$ |
| `energy(...)` | `[B]` | $E(S)$ for the observed baskets |
| `loglik(...)` | | set likelihood, optionally with ESS and size |
| `size_dist(ix, ...)` | `[B, nmax]` | $P(n)$, exact given $z$ |
| `sample(ix, ...)` | list of lists | one basket per trip, THEORY §10 |
| `project(...)`, `project_price(...)`, `project_mean(...)`, `clamp_rho_c(...)` | | constraint steps applied after each optimiser step |

`set_quad(model, ...)` is the single place the integrator is chosen.

---

## 5. The forward pass

For one minibatch of $B$ trips:

1. **`Batcher.make(trips)`** gathers the assortment into a `RaggedIndex`, looks up
   `dlp`, `disp`, `mail`, `rec` per slot, and computes the price reference `dlp_bar`.
   Under `price_ref=category` the reference is a **per-slot** vector `rowbar[row_of]`;
   under `trip` it is a **per-trip** scalar. `b_at` detects which by shape.
2. **`b_flat(ix)`** $\to$ `[T]` utilities.
3. **`w_j(z)`** $= \exp(b_j - \tfrac12\lVert\phi_j\rVert^2 + \phi_j^{\top}z)$ at each of the
   681 Smolyak nodes.
4. **`esp_bucketed`** runs recursion (6.1) per row, bucketed by `row_size` so short rows do
   not pay for the 1,773-product one, truncated at `poly_degree`.
5. **Convolution across rows** $\to A_n(z)$, then $f(z)=\sum_n e^{-\rho_0(n)}A_n(z)$.
6. **Quadrature** over nodes $\to \log Z$.
7. **`energy()`** scores the observed basket; the loss is $E(S)-\log Z$.

---

## 6. Checkpoints

Written by `fit.py::save_ckpt` as a dict with `format = 2`:

| key | contents |
|---|---|
| `model` | `state_dict` |
| `opt`, `sched` | optimiser and schedule state — so `--resume` is a continuation, not a restart |
| `rng_np`, `rng_torch` | both RNG streams, so a resumed run draws what it would have drawn |
| `iter`, `cum_iter`, `best_vb`, `best_it`, `lz_strikes` | |
| `data` | `{partition, affinity, n_cat, R}` — guards against loading under the wrong data build |
| `quad` | how $\log Z$ was integrated |
| `model_flags` | `{price_soft, price_ref, poly_degree}` |

### Why `model_flags` exists

**Two different models can have byte-identical weights.**

- **`price_soft`** — whether `gamma`/`beta` *are* the price coefficients or softplus
  pre-images. $\gamma = +0.0207$ is valid under both readings. Reading it wrong uses
  $\mathrm{softplus}(0.0207) = 0.7036$ — **34× too large** — and reported MRR 0.0044 for a
  model whose training log said 0.0705.
- **`price_ref`** — whether $\bar\ell$ is the trip-wide or category mean. Scoring a
  category-referenced model against a trip mean deletes its substitution channel entirely
  (THEORY §11.4).
- **`poly_degree`** — recorded for diagnosis; evaluation re-derives a safe value (§9 of
  THEORY).

**Load every checkpoint through `evalall.load_any`.** It restores these flags, verifies the
data partition, and selects the integrator the checkpoint was trained under. A bare
`load_state_dict` gets none of that and fails *silently*. `stamp_flags.py` backfills older
checkpoints.

---

## 7. The training loop

`fit.py::main`, in order:

```
build data and features
construct RaggedModel
apply staged freezes (--interaction-stage, --size-stage, ...)
LOAD CHECKPOINT              (--resume / --warm-start)
calibrate poly_degree        <- against the LOADED weights
apply price_soft warm start  <- to the LOADED weights
set price_ref, kappa_init
apply phi mask / spectral init
build optimiser groups
train
```

**The order is load-bearing.** Two bugs came from getting it wrong:

- the `price_soft` warm start once ran *before* the load, converting weights the load then
  overwrote while leaving the flag set — initial eval $-270{,}161$;
- degree calibration once ran *before* the load, on a model whose $\rho_c \approx 0$ where
  every degree agrees, choosing 96 for a checkpoint whose safe ceiling was 32.

### Logging

`--metrics-jsonl` writes one JSON object per evaluation: 37 fields including per-block
gradient norms, `lam_max`, `phi_max`, `phi_zero_frac`, model and observed $\mathbb{E}[n]$
and $\mathrm{Var}(n)$, elasticity, ESS, learning rate and wall clock. Diagnosis is a query
over that file, not a regex over the 1,200-character human line — and the wall clock in it
is what distinguishes a stall from a slowdown after the fact.

---

## 8. Optimiser groups

One learning rate cannot serve this model. `optimizer_parameter_groups` splits it:

| group | scale | natural parameter scale | why |
|---|---|---|---|
| main | $1.0$ | — | structural parameters |
| `lam` | $0.05$ | — | already an exposure-corrected estimate over the whole training split; a 24-trip gradient should not move it a full step |
| `price` (`gamma`, `beta`) | $0.05$ | $\approx 0.02$ | unconstrained these *are* the coefficients, so an unscaled step moves them 10% of their own value — $51\times$ the constrained effective step $0.002\cdot\sigma(-3.92)=3.9\times10^{-5}$ — and diverges |
| `kappa` | $5$–$20$ | $\approx 40$ | the structural rate moves it 0.005% per step |

That is a **400× spread** between the price group and $\kappa$. Adam normalises the
*gradient*, so this is invisible as a gradient change and shows up only as divergence or as
a parameter that never arrives.

---

## 9. Guards

| guard | trigger | why |
|---|---|---|
| **divergence tripwire** | eval with model $\mathbb{E}[n] > 0.5\,n_{\max}$ while data sits at 8.61 | aborts with the diagnosis instead of clamping; caught a bad run at the first eval rather than after 25,000 iterations |
| **`rho_c` step scale** `0.05` | — | $\exp(-\rho_c n_c(n_c-1)/2)$ overflows float64 at $\rho_c\approx-0.10$ with $n_c=120$ |
| **`rho_c` floor** `-0.92` | — | a $2.5\times$ pair lift is $\rho_c=-0.92$; below that the term detonates |
| **partition guard** | `V3_AFFINITY=1` with `items_affinity.parquet` absent | previously fell through to the default 188-category partition **and cached it under the affinity filename** — a different model under the right name |
| **ESS gate** | low effective sample size | on the importance-sampling estimator |

---

## 10. Resuming

Two traps, both of which cost a run here.

The scheduler counter is **per process**, and `opt.load_state_dict` restores the optimiser's
learning rate. Therefore:

- resuming **without** `--fresh-sched` restores the *old* rate and silently ignores `--lr`;
- resuming **with** new `--lr-milestones` starts a fresh counter, discarding decays the
  previous run already applied — one run resumed a converged model at $4\times$ its rate and
  lost 0.59 nats before the cause was found.

**To continue a run:**

```bash
--resume <ckpt> --fresh-sched 1 --lr <the rate the previous run ENDED at>
```

The startup log prints every group's rate and what a fresh schedule would use, so this is
checkable in one line before the run gets going.

---

## 11. Environment variables

| variable | effect |
|---|---|
| **`V3_AFFINITY=1`** | selects the **280-category co-purchase affinity partition**. The default is 188 merchandiser commodities; `V3_PARTITION=items_subcom.parquet` gives 758 sub-commodities. |
| `NF_ROOT` | where `data/`, `basket_input/`, `out/` are read and written. Does **not** move `RAW`. |
| `NF_RAW_DIR` | where the raw dunnhumby CSVs live. Defaults to a sibling of the checkout. |
| `OMP_NUM_THREADS` | CPU threads. ~0.55 s/iteration at 12 threads. |

**The partition changes the model.** It defines the ragged row structure, so it sets $C$,
the cache filename, and the shape of `rho_c`. **Checkpoints are not comparable across
partitions**, and loading one under the wrong partition fails on a `rho_c` shape mismatch —
`load_any` turns that into an explicit message naming the partition to use.

All three scripts in `src/run/` export `V3_AFFINITY=1`. Anything run by hand must too.
