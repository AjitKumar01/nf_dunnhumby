# Architecture

What each module does, what flows between them, and the handful of facts that are not
guessable from reading the code.

For the mathematics see `THEORY.md`; for how raw CSVs become tensors see
`DATA_TO_MODEL_INPUT.md`.

---

## 1. Module map

```
src/basket/
  paths.py              where everything lives.  Discovers the repository root by walking
                        up from the file; NF_ROOT overrides it, NF_RAW_DIR points at the
                        raw CSVs.  Nothing else hardcodes a relative path, so the layout
                        can be moved and scripts run from any working directory.

  data.py               build() -> the ragged assortment index, cached to
                        basket_input/v3_index*.npz.  Asserts every purchased product lies
                        in its store's assortment; otherwise the likelihood would be
                        evaluated on a support that excludes the observed basket.

  features.py           Features: memory-maps the four context sources (price deviation,
                        store price, recency, promotion) and answers batch queries by
                        vectorised searchsorted.

  ragged.py             THE KERNEL.  Energy, normaliser, marginals, sampler, projections.
  fit.py                Training loop, objectives, schedules, checkpointing, logging.
                        Also Batcher, which assembles a minibatch's tensors.
  evalall.py            load_any() -- the single correct way to load a checkpoint.

  downstream.py         counterfactual / personalisation / segmentation / generation
  eval_mrr_cutoffs.py   MRR and MRR@k on the fixed holdout
  diagnose_bucket_coverage.py   compares against the historical truncated kernel
  stamp_flags.py        backfills model_flags onto pre-2026-08 checkpoints

  pairmask.py           picks the interaction products by co-purchase lift
  phi_spectral_init.py  places phi by eigendecomposing the empirical log-lift matrix
  beta_target.py        per-product price-coefficient calibration target

src/pipeline/           raw dunnhumby CSVs -> data/ -> basket_input/
src/run/                prepare.sh, train.sh, evaluate.sh
```

`src/pipeline/*` import `paths` from `src/basket/` via a two-line bootstrap; that is the
only cross-directory dependency.

---

## 2. Data flow

```
  dunnhumby CSVs                       (read-only, NF_RAW_DIR)
        |  src/pipeline/01_build_base.py            reconstructs shelf price
        v
  data/tx.parquet, trips, price panels
        |  src/pipeline/22_basket_data.py + 23_promo_data.py
        v
  basket_input/  baskets.parquet  items.parquet  log_price*.npy
                 store_price.npz  state.npz  promo.npz  meta.json
        |  src/basket/data.py :: build()            cached once
        v
  basket_input/v3_index_affinity.npz    ragged assortment index + trip table
        |  src/basket/features.py + fit.py :: Batcher    per minibatch, never written
        v
  RaggedIndex + context dicts  ->  RaggedModel
```

Scale: **5,455 products, 2,066 households, 115 stores, 712 days.** After dropping held-out
lines outside the training support: **198,690 trips, 1,558,093 purchase lines.** Split temporally by household-week: train `week < 83`,
validation `83..90`, test `>= 91`.

---

## 3. The central data structure

`RaggedIndex` (in `ragged.py`) represents "the assortments of the trips in this batch",
grouped by `(trip, category)` rows:

| field | shape | meaning |
|---|---|---|
| `item` | `[T]` | product id of every assortment slot in the batch |
| `row_of` | `[T]` | which `(trip, category)` row each slot belongs to |
| `item_trip` | `[T]` | which trip each slot belongs to (`row_trip[row_of]`) |
| `row_trip`, `row_cat` | `[n_rows]` | the trip and category each row is |
| `row_size` | `[n_rows]` | products in that row — median 3, max 1,773; purchase-weighted median 128 |
| `B` | scalar | trips in the batch |

Everything downstream is a segment operation over `row_of` or `item_trip`. `T` is about
127,000 slots for a 24-trip batch.

Context arrives as two parallel dicts with identical keys — `ctx` over assortment slots,
`lctx` over purchased lines — so `energy()` and `log_Z()` score a product identically.
Keys: `dlp`, `dlp_bar`, `disp`, `mail`, `week`, `store`, `rec`.

---

## 4. `RaggedModel`: parameters and the methods that matter

| parameter | shape | role |
|---|---|---|
| `lam` | `[J]` | product intercept (exposure-corrected incidence at init) |
| `theta` | `[N, 32]` | household taste embedding |
| `phi` | `[J, Kz]` | interaction embedding, `Kz = 4`, masked to 30 products |
| `gamma` | `[N, Kp]` | household price loading, `Kp = 8` |
| `beta` | `[J, Kp]` | product price loading |
| `price_kappa` | scalar | idiosyncratic/aggregate price split (see THEORY §6) |
| `rho_c` | `[C]` | within-category effect, `C = 280` under the affinity partition |
| `rho_0_free` | `[nmax]` | basket-size potential |
| `xi` | `[S, Ks]` | store effect |
| `a_q`, `gamma_q`, `beta_q`, `log_r` | | units model |

Methods:

* `b_flat(ix)` / `b_at(...)` — per-slot utility. All price sites route through `price_g()`
  and `price_b()` so the elasticity machinery cannot disagree with the utility.
* `log_Z(ix)` — the normaliser, by quadrature over the ragged structure.
* `pi_quad(ix)` — `pi_j = P(j in S)`, by autograd through the quadrature. **Needs grad
  enabled**; wrapping it in `no_grad` raises. Guarantees `sum_j pi_j = E[n]`.
* `energy(...)`, `loglik(...)`, `size_dist(...)`, `units_loglik(...)`
* `sample(ix, ...)` — one basket per trip (THEORY §5)
* `project(...)`, `project_price(...)`, `project_mean(...)`, `clamp_rho_c(...)`

`set_quad(model, ...)` is the single place the integrator is chosen.

---

## 5. Checkpoints

Written by `fit.py::save_ckpt` as a dict with `format = 2`:

```
model        state_dict
opt, sched   optimiser and schedule state -- so --resume is a continuation, not a restart
rng_np, rng_torch    both RNG streams
iter, cum_iter, best_vb, best_it, lz_strikes
data         {partition, affinity, n_cat, R}      guards against loading under the wrong
                                                  data build (rho_c changes shape)
quad         how log Z was integrated
model_flags  {price_soft, price_ref, poly_degree} properties the TENSORS CANNOT EXPRESS
```

`model_flags` exists because two different models can have byte-identical weights:

* **`price_soft`** — whether `gamma`/`beta` are the price coefficients themselves or
  softplus pre-images. `gamma = +0.0207` is valid under both readings. Reading it wrong uses
  `softplus(0.0207) = 0.7036`, **34× too large**, and reports MRR 0.0044 for a model whose
  training log says 0.0705.
* **`price_ref`** — whether `dlp` is referenced to the trip's whole assortment or to its
  category. Scoring a category-referenced model against a trip mean deletes its substitution
  channel (THEORY §6).
* **`poly_degree`** — recorded for diagnosis; evaluation re-derives a safe value.

**Load every checkpoint through `evalall.load_any`.** It restores these flags, verifies the
data partition matches, and selects the integrator the checkpoint was trained under. A bare
`load_state_dict` gets none of that and fails silently rather than loudly.

---

## 6. Training loop

`fit.py::main` in order: build data and features → construct the model → apply staged
freezes → **load the checkpoint** → calibrate the truncation degree *against the loaded
weights* → apply `phi` mask/init → build the optimiser groups → train.

The order matters. Two bugs came from getting it wrong: the `price_soft` warm start once ran
before the load (converting weights the load then overwrote), and degree calibration once ran
before the load (choosing 96 for a checkpoint whose safe ceiling was 32, because a fresh
model has `rho_c ≈ 0` where every degree agrees).

### Optimiser groups

One learning rate cannot serve this model. `optimizer_parameter_groups` splits it:

| group | scale | why |
|---|---|---|
| main | 1.0 | structural parameters |
| `lam` | 0.05 | already an exposure-corrected estimate over the whole training split; a minibatch gradient should not move it a full step |
| `price` (`gamma`,`beta`) | 0.05 | unconstrained these ARE the coefficients (~0.02), so an unscaled step moves them 10% of their own value — 51× the constrained effective step, and it diverges |
| `kappa` | 5–20 | natural scale ~40, so the structural rate moves it 0.005% per step |

That is a **400× spread** between the price group and `kappa`.

### Guards

* **divergence tripwire** — any eval with model `E[n] > 0.5·n_max` while the data sits at
  8.6 aborts with the diagnosis rather than clamping.
* **`rho_c` step scale** `0.05` — `exp(-rho_c·n_c(n_c-1)/2)` overflows float64 at
  `rho_c ≈ -0.10` with `n_c = 120`.
* **ESS gate** on the importance-sampling estimator.

### Logging

`--metrics-jsonl` writes one JSON object per evaluation: 37 fields including per-block
gradient norms, `lam_max`, `phi_max`, `phi_zero_frac`, model and observed `E[n]` and
`Var(n)`, elasticity, ESS, learning rate and wall clock. Diagnosis is a query over that
file, not a regex over the 1,200-character human log line — and wall clock in it is what
lets a stall be told from a slowdown after the fact.

---

## 7. Resuming: two traps

The scheduler counter is **per process**, and `opt.load_state_dict` restores the optimiser's
learning rate. So:

* resuming **without** `--fresh-sched` restores the *old* rate and ignores your `--lr`;
* resuming **with** new `--lr-milestones` starts a fresh counter, discarding decays the
  previous run already applied — a run here resumed a converged model at 4× its rate and
  lost 0.59 nats before the cause was found.

To continue a run: `--resume <ckpt> --fresh-sched 1 --lr <the rate the previous run ended at>`.
The startup log prints every group's rate and what a fresh schedule would use, so this is
checkable in one line before the run gets going.

---

## 8. The environment variable that changes the model

`V3_AFFINITY=1` selects the **280-category co-purchase affinity partition**. The default is
188 merchandiser commodities; a third option, `V3_PARTITION=items_subcom.parquet`, gives 758
sub-commodities.

The partition defines the ragged row structure, so it sets `C`, the cache filename, and the
shape of `rho_c`. **Checkpoints are not comparable across partitions** and loading one under
the wrong partition fails on a `rho_c` shape mismatch — `load_any` turns that into an
explicit message naming the partition to use, because it was once a bare shape error.

All three scripts in `src/run/` export it. Anything you run by hand must too.
