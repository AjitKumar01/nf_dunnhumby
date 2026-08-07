# Handoff

Written at the end of a long session. Read this first, then `NESTED_MODEL.md` §12
(changelog) and `paper/nested_basket.tex`.

---

## 1. What this project is

A model of dunnhumby "Complete Journey" supermarket baskets, built to answer two
questions: **what happens if a price changes**, and **can it generate synthetic baskets
good enough to learn a pricing policy on**.

It separates three decisions a single-choice model conflates:

| decision | head | question |
|---|---|---|
| incidence | Bernoulli per (trip, category) | does this household buy from this category at all? |
| allocation | softmax over items | which product? |
| breadth | `1 + Poisson` | how many *different* products from that category? |
| quantity | `1 + Poisson` | how many units of each? |

All four share one item utility `u_ijt`. The "nest" is **one term** — `κ_c(IV_ict − IV̄_c)`
in the incidence logit — which is the only place category incidence sees how good the
category's items are. `κ` has a nested-logit reading: 0 = a price cut only moves share,
1 = category grows exactly as its items gained, >1 = it grows more.

**Do not describe this as a fitted Poisson–multinomial.** It is not. See §4.3 and §4.5 of
the paper; the factorisation motivates κ's meaning and nothing else.

---

## 2. Current state (as of this handoff)

**Branch `category-state-fix`**, three commits ahead of `eda-audit-fixes`:

```
470b67d  Delete the leakage flags
abf197a  Fix the category recency feature
203a595  paper: define every symbol and show how the price series is built
```

**12 models are training right now** (wave 1 of 2, launched via `bash /tmp/rerun.sh`
from `scripts/`). Logs at `out/log_x_<label>.txt`. When wave 1 finishes the script
starts wave 2 automatically and prints `MODELS_DONE`.

### What must be rerun after those finish

Everything downstream, because the category-state fix changes the incidence head:

```bash
cd scripts
# generator evaluations -- 15 labels, run ~5 at a time with OMP_NUM_THREADS=3
python3 34_generator_eval.py --label <L> --n-trips 24768 --top-items 500
# then
python3 28_nested_counterfactual.py --labels nested nested_pl
python3 31_benchmark.py
python3 24_embedding_eval.py --labels nested nested_noctx nested_both nested_nostate \
        --primary nested --suffix _nested
python3 36_strong_baselines.py --labels nested nested_both --n-baskets 4000 --iters 6000
python3 35_context_ablation.py --n-baskets 4000 --labels nested nested_both nested_hn75
python3 37_price_leak_test.py --n-baskets 3000
python3 33_verify_equations.py     # must print "all equations verified"
python3 29_demand_eda.py ; python3 30_household_eda.py ; python3 32_reliability.py
```

**Then regenerate the paper.** Its §5 onward still carries numbers from the previous
complete set. Every number in the paper was substituted programmatically from `out/*.json`
— do that again rather than editing by hand.

---

## 3. Hardware and how to run efficiently (measured, not guessed)

MacBook Pro M5 Pro, 15 cores, 24 GB unified.

| configuration | result |
|---|---|
| **MPS (GPU)** | **2× slower than CPU** — the hot path is `np.searchsorted` and small tensor ops |
| 1 job × 4 threads | 13.2 min/model |
| 1 job × 8 threads | 13.4 min — no gain past 4 |
| 6 jobs × 2 threads | 17.6 models/hour |
| **10–12 jobs × 2 threads** | **24 models/hour — use this** |
| 16 jobs × 1 thread | collapses |
| memory, 16 jobs | 6.4 GB of 24 — never the constraint |

**Do not oversubscribe.** Running trainings and evaluations together once starved the
evaluations to 6 minutes of CPU across 100 minutes of wall clock. Run trainings, `wait`,
then evaluations.

---

## 4. The leakage audit — what was found, what was fixed, what was over-corrected

This consumed much of the session. The principle that emerged:

| kind | example | leak? |
|---|---|---|
| **contemporaneous** | the posted price on day *t*; a store's price that week | **No.** Known at the decision. Restricting it deletes the experiment. |
| **past lookup** | days since this household last bought X | **No.** `searchsorted` is strictly-before. |
| **time-aggregate** | the within-item mean that centres price; repurchase gaps; `carried` | **Yes.** Only these are restricted to weeks < 83. |

### Fixed
- price centering used all 712 days → now train weeks only
- backward-fill from the future (affected 1 item)
- `sub_gap` / `cat_gap` repurchase medians → train only
- `carried` availability ("ever sold") → train only. Density 57.2% → **50.9%**

### Over-corrected twice — do not repeat
1. **Restricting the price panel to training weeks.** Held-out price variation went to
   **exactly zero**. Reverted.
2. **Restricting the state event keys.** Deleted legitimate past history for test rows.
   Reverted; only the *aggregate* `sub_gap` is train-only.

### Deliberately not treated as a leak
The `≥100 lines` / `20–300 trips` filters. They are a **sample-definition choice**, not a
prediction-time leak — item existence tells the model nothing about which item a basket
contained. Dropped items have a median of 86 training lines, so including them made the
task *harder*, not easier. **Removing them was tried and reverted** (see §7).

### Inherent to the data, tested
Prices are reconstructed from the very transactions being scored — **41.2%** of held-out
rows are the sole observation of their item-day, and the true item always has a same-day
price where only 43.1% of decoys do. Tested with matched decoys in
`37_price_leak_test.py`: the margin **grew** (+0.342 → +0.449), so it is not driving the
result.

### Net effect on conclusions
One number moved: **generated price response 82% → 77%**. The price coefficient moved 2%,
the decomposition shares by a point, the placebo still collapses to zero. **The EDA did not
move at all** — verified that `29/30/32` read only `baskets.parquet`, `items.parquet` and
`log_price.npy`, none of which the leaks touched.

---

## 5. Results as of the last complete clean set

These are from the run *before* the category-state fix. They will shift slightly.

| quantity | value |
|---|---|
| price coefficient | **+0.7785** (placebo: **0.0000**) |
| own-price elasticity | **−1.1661** = allocation 84% / incidence 4% / quantity 12% |
| κ (median) | 0.638 |
| seed spread | **0.0057** nats |
| ablations | store 0.305, interaction 0.108, state 0.071 |
| vs strongest learned baseline (B-Emb) | **+0.471 nats** |
| high-price decile | model **gains +0.103**; HPF and B-Emb each lose ~0.035 |
| generation | items −0.5%, categories −2.9%, units −0.5% |

### The one real discovery

Generated baskets had roughly the right *amount* of co-purchase structure on the wrong
*pairs*. **Eleven changes to the model moved the per-pair rank correlation by nothing** —
symmetric/asymmetric interaction, tied/untied embeddings, mean/sum pooling, learnable
scale, category context, a free 17,578-parameter category-pair matrix, SHOPPER's
sequential likelihood, SHOPPER's one-vs-each loss.

**One change to the negative sampling moved it.** Drawing a fraction `f` of negatives from
the true item's own category:

| f | lift | Spearman | z |
|---|---|---|---|
| 0 | 48% | −0.002 | −0.2 |
| 0.25 | 50% | +0.021 | +3.0 |
| 0.50 | 54% | +0.059 | +8.8 |
| **0.75** | **54%** | **+0.097** | **+14.9** |
| 1.00 | 56% | +0.077 | +12.8 |

Monotone to 0.75 — a dose-response curve, which is what makes it a mechanism rather than a
lucky setting. Past 0.75 it costs price response (κ collapses).

**Recommended:** `nested_both` for general use; `--neg-in-cat 0.75` if per-pair
co-occurrence matters.

---

## 6. Known defects still open

1. **Item head trained on one choice set, used on another.** Negatives come from the whole
   catalogue (`27:423`); generation draws within a category (`28:436`). This is a plausible
   partial explanation for why in-category negatives help, and it has **not** been
   separated from the "the loss finally has to ask whether an item fits this basket"
   explanation.
2. **Incidence uses a logit where the Poisson–multinomial implies cloglog.** Gap in the
   linear predictor is 0.017 at the 3.25% base rate, 0.367 at p = 0.5. Fine here, not in
   general.
3. **No held-out joint likelihood.** Every likelihood reported is conditional on the rest
   of the basket. Generation is the only test of the joint.
4. **Per-pair co-occurrence tops out at +0.097.** Right amount of clustering, only weakly
   the right pattern.
5. **Deployment: a one-item cart is worse than an empty one.** `‖ᾱ‖` at k=1 is 2.42× the
   training norm. **Fix: rescale `ᾱ` to the training norm** — recovers 0.8 nats, beats
   zeroing at every cart size. Not yet in the model code, only measured in
   `35_context_ablation.py`.
6. **`BASKET_MODEL.md` still carries leaky numbers.** The flat model (`one_*`, `tied_*`)
   was never refitted. Separate model, separate document, not touched.

---

## 7. Things tried and reverted — do not redo without reason

- **Removing the item/household filters.** Catalogue goes to 91,856 items, `carried`
  collapses to 1.5%, surviving candidates drop 15.7 → 9.2 of 21, training costs 1.8×.
  Reverted and the commits removed from history.
- **MPS/GPU.** 2× slower.
- **SHOPPER's untied ρ, sequential likelihood, one-vs-each loss.** All implemented, all
  fitted, none moved the target. Flags still exist: `--untie-rho`, `--prefix-context`,
  `--item-loss ove`.
- **`--ctx-agg sum`.** Cleaner theory, 0.11 nats worse, every co-occurrence measure worse.

---

## 8. Working style this session established

- **Verify, don't assert.** Several conclusions were stated confidently and turned out
  wrong: "56% of baskets violate unit demand" (it is 68.2%), a `[ -f file ]` check that
  read stale artefacts as complete, two over-corrections in the leak audit.
- **Check timestamps, not existence**, when deciding whether an artefact is current:
  compare against `stat -f %m basket_input/meta.json`.
- **Numbers in documents are substituted programmatically** from `out/*.json`. Never type
  one by hand.
- **Report findings first, then interpretation.** No self-congratulatory framing.
- **Explain every symbol where it appears**, and give the formula and the raw values behind
  every number. This was asked for repeatedly.
- **Distinguish mechanical from real changes.** A likelihood that improves because the
  choice set shrank is not a gain.
- **Branch per major change and commit there.**

---

## 9. Key files

| file | what |
|---|---|
| `scripts/22_basket_data.py` | builds `basket_input/`; contains the leak-fix logic |
| `scripts/27_nested_basket.py` | the model and its training loop |
| `scripts/28_nested_counterfactual.py` | placebo, elasticity decomposition, generation |
| `scripts/33_verify_equations.py` | checks every printed equation against the code |
| `scripts/34_generator_eval.py` | the five model-free distributional checks |
| `scripts/35_context_ablation.py` | deployment: recommending without a basket |
| `scripts/36_strong_baselines.py` | HPF, B-Emb, and the price-variation evaluation |
| `scripts/37_price_leak_test.py` | contemporaneous-price control |
| `NESTED_MODEL.md` | the model document; §12 is the changelog |
| `DATA_EXPLORATION.md` | the EDA, standalone, no reference to any model |
| `paper/nested_basket.tex` | 25pp writeup; compiles clean with `pdflatex` ×3 |
