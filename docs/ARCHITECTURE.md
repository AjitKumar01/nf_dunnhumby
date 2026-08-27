# Architecture

## How to read this

This follows **one basket** from a row in a CSV to a number the optimiser can use, and
explains each piece of machinery at the point where the basket meets it. Reference tables —
exact shapes, every parameter, every flag — come afterwards, once you know where they sit.

The mathematics is in [`THEORY.md`](THEORY.md); this is about the code. Section references
like *(THEORY §5)* point there.

**Contents**

*Part I — orientation*
1. [The one-screen map](#1-the-one-screen-map)
2. [Follow one basket through the system](#2-follow-one-basket)

*Part II — the pieces*

3. [`RaggedIndex`: the central structure](#3-raggedindex)
4. [`RaggedModel`: parameters and methods](#4-raggedmodel)
5. [The forward pass, step by step](#5-the-forward-pass)

*Part III — running it*

6. [Checkpoints and `model_flags`](#6-checkpoints)
7. [The training loop](#7-the-training-loop)
8. [Optimiser groups and the 400× scale spread](#8-optimiser-groups)

*Part IV — what will bite you*

9. [Guards](#9-guards)
10. [Resuming](#10-resuming)
11. [Environment variables](#11-environment-variables)

---

# Part I — orientation

## 1. The one-screen map

**The question: what are the moving parts, and which one would I edit?**

Three stages, and you almost always want the middle one.

```
  RAW CSVs  ──►  src/pipeline/  ──►  basket_input/  ──►  src/basket/  ──►  out/
   142 MB +       run once,          the modelling      the model,        checkpoints,
   696 MB         ~10 min            universe           training, eval    metrics
```

| if you want to change... | edit |
|---|---|
| which products/households are in scope, how price is reconstructed | `src/pipeline/` — then re-run `prepare.sh` |
| the model, its energy, its normaliser, its sampler | `src/basket/ragged.py` — **the kernel** |
| the objective, schedules, what is constrained | `src/basket/fit.py` |
| how a checkpoint is scored | `src/basket/downstream.py`, `eval_mrr_cutoffs.py` |
| where files live | nothing — `src/basket/paths.py` discovers the root |

The rest of `src/basket/` is support: `data.py` builds the assortment index, `features.py`
serves context, `evalall.py` loads checkpoints correctly, and four small scripts build
auxiliary inputs (`pairmask.py`, `phi_spectral_init.py`, `beta_target.py`,
`elasticity_targets.py`).

---

## 2. Follow one basket

**The question: what actually happens between a CSV row and a gradient?**

Take one real trip: household 1042 visits store 367 on day 415 and buys four items.

### Stage 1 — it becomes a row in a table (`src/pipeline/`)

`01_build_base.py` reads `transaction_data.csv`. The raw file has no price column — it has
`SALES_VALUE` and three discount columns — so the shelf price is **reconstructed**, because
the model must condition on what every shopper saw, including those who did **not** buy.
That derivation is audited in [`PREPROCESSING.md`](PREPROCESSING.md).

`22_basket_data.py` then decides the modelling universe: products with ≥100 purchase lines,
households with 20–300 trips. Our trip survives as four rows in `baskets.parquet`.

### Stage 2 — it becomes an index entry (`data.py`)

The likelihood must sum over **every subset of the store's assortment** (THEORY §4), so the
assortment must be an indexed structure, and each purchased product must be expressed in
**assortment-local coordinates**.

`build()` produces, for store 367, the list of products it carries grouped by category, and
records each of our four purchases as *"position 7 within the (store 367, category 12)
row"*. That position is what the polynomial recursion of THEORY §7 indexes.

It also **asserts** every purchased product lies in its store's assortment. If one did not,
the likelihood would be evaluated on a support that excludes the observed basket — silently
wrong, rather than an error.

### Stage 3 — it becomes tensors (`Batcher.make`)

Our trip is batched with 23 others. `Batcher` gathers ~127,000 assortment slots and, for
each, looks up: the item's log-price deviation that day, whether it was on display or in the
mailer that week, how long since this household last bought its sub-commodity, and the
**price reference** $\bar\ell$ (THEORY §10.4 — the choice that decides whether the model can
substitute at all).

Two parallel views come out: `ctx` over all 127,000 slots, `lctx` over the 4 purchased
lines. They carry identical keys so that `energy()` and `log_Z()` score a product
*identically* — a product must not be worth one thing as a candidate and another as a
purchase.

### Stage 4 — it becomes a number (`ragged.py`)

1. `b_flat(ix)` → a utility for each of the 127,000 slots (THEORY 3.2).
2. For each of **681 Smolyak quadrature nodes** $z$, form the tilted weights
   $w_j(z)$ (THEORY 5.3).
3. `esp_bucketed` runs the elementary-symmetric recursion per category row — bucketed by row
   size so a 3-product row does not pay for the 1,773-product one.
4. Convolve across rows → $A_n(z)$, weight by the size potential → $f(z)$.
5. Integrate over the 681 nodes → $\log Z$.
6. `energy()` scores the *observed* four-item basket.

The loss for our trip is $E(S) - \log Z$: how good this basket is, minus how good all
possible baskets are. Backpropagation from there reaches every parameter — including through
the normaliser, which is why $\log Z$ has to be differentiable and not merely computable.

> **Where we are.** That is the whole path. The rest of this document is the detail of each
> piece, and the things that will bite you when you run it.

---

# Part II — the pieces

## 3. `RaggedIndex`

**The question: how is a ragged assortment held in a dense tensor?**

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

**The question: what does the model own, and what can I ask it?**

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

**The question: in what order does it all happen?**

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

# Part III — running it

## 6. Checkpoints

**The question: why can two checkpoints with identical weights be different models?**

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

**The question: what happens before the first gradient step, and why does the order matter?**

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

**The question: why can one learning rate not serve this model?**

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

# Part IV — what will bite you

## 9. Guards

**The question: what fails loudly, and what used to fail silently?**

| guard | trigger | why |
|---|---|---|
| **divergence tripwire** | eval with model $\mathbb{E}[n] > 0.5\,n_{\max}$ while data sits at 8.61 | aborts with the diagnosis instead of clamping; caught a bad run at the first eval rather than after 25,000 iterations |
| **`rho_c` step scale** `0.05` | — | $\exp(-\rho_c n_c(n_c-1)/2)$ overflows float64 at $\rho_c\approx-0.10$ with $n_c=120$ |
| **`rho_c` floor** `-0.92` | — | a $2.5\times$ pair lift is $\rho_c=-0.92$; below that the term detonates |
| **partition guard** | `V3_AFFINITY=1` with `items_affinity.parquet` absent | previously fell through to the default 188-category partition **and cached it under the affinity filename** — a different model under the right name |
| **ESS gate** | low effective sample size | on the importance-sampling estimator |

---

## 10. Resuming

**The question: why did my resumed run get worse?**

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

**The question: which switch silently changes the model?**

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
