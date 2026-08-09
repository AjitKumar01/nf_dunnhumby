# Handoff

Written after the specification rewrite. Supersedes the previous handoff entirely — that
one describes a model this repository no longer implements.

**Read in this order:** `paper/model_spec.html` (what the model is), then
`paper/experiments.html` (what it does), then this file (how to run it and what is left).
`paper/nested_basket.tex` compiles to the same content in paper form.

---

## 1. What this is

A model of dunnhumby "Complete Journey" supermarket baskets, answering two questions: what
happens if a price changes, and can it generate synthetic baskets worth learning a policy
on.

A basket is `ℬ = (𝒦, S, q)` and its probability factorises in the order those are decided:

```
P(ℬ | household, store, day, week, prices) = P(𝒦) · P(S | 𝒦) · P(q | S)
```

- **𝒦, the composition** — which categories, and how many distinct products from each.
  Complementary log-log incidence per category, plus a truncated shifted-Poisson breadth.
  Uses the solo product value `b` only, because no basket exists yet.
- **S, the contents** — which products fill that shape. A Gibbs measure on the baskets
  matching 𝒦, `P(S|𝒦) ∝ exp E(S)`, whose single-position conditionals are a softmax over
  **the purchased product's category**, not the catalogue.
- **q, the units** — shifted Poisson per product.

The nesting coefficient κ scales how far a category's inclusive value moves its entry
log-hazard. It appears in the incidence linear predictor **and nowhere else** — breadth and
units never see IV, so nothing about category *volume* can be read off κ.

---

## 2. State of the code

Branch `spec-conformance`. Everything below is implemented and verified.

| | |
|---|---|
| item head | exact softmax over the category (median 15 products, max 225) — **no negative sampling anywhere** |
| incidence | cloglog, `1 − exp(−e^η)`, verified against the direct formula to 3.3e−06 |
| breadth | shifted Poisson truncated at the store's stock, verified to sum to 1 |
| interaction | tied `ρ = α`, required for the swap identity and hence for `P(S\|𝒦)` to exist |
| persistence | per-(household, product) share and frequency, train-only, per-product loadings |
| conditioning | on `n ≥ 1`; the empty basket is never observed |
| equations | 30 checks in `scripts/eval/33_verify_equations.py`, all pass |

Key flags: `--l2-incidence 1e-4` (**required** — see §4), `--no-persist`, `--pcd`
(default 0, off), `--placebo-price permute`.

---

## 3. Headline results

Chance on the within-category conditional is `−log 47 = −3.85`, top-1 `2.1%`.

| | item | 95% CI on the gap | clears refit noise? |
|---|---|---|---|
| full model | **−2.4611** | — | — |
| no persistence | −2.5340 | [+0.0648, +0.0816] | yes |
| original incidence penalty | −2.5327 | [+0.0635, +0.0804] | yes |
| basket interaction | −2.6400 | — | yes |
| prices scrambled | −2.5174 | [+0.0489, +0.0626] | yes |
| household state | −2.6305 | — | yes |
| nest | −2.6003 | — | yes, by under 2× |
| quantity head | −2.5813 | — | **no**, and correctly so |

Ablation ordering: store 0.337 ≫ persistence 0.073 > interaction 0.058 ≈ state 0.049 >
nest 0.019 ≫ quantity 0.000.

**Two kinds of uncertainty, measured separately.**

*Refitting* — seven independent fits, identical data and schedule, differing only in seed:
item sd **0.0040**, top-1 sd 0.0013, price coefficient sd 0.0104, κ sd **0.0011**. A gap
between two independent fits must exceed **~0.011** to clear refit noise at 95%. This
replaces the two-seed figure of 0.0095, which was too noisy to support the claims resting
on it.

*Evaluation sampling* — household block bootstrap, 400 draws over 1,664 households, paired
so the interval is on the difference. `scripts/eval/43_bootstrap.py`, no refitting needed.

Neither covers variability from resampling the training households; that needs a refit per
replicate and has not been run.

**Causal.** The placebo retains 0.0% of the price coefficient on both margins; its
allocation-elasticity interval is `[−0.0029, +0.0000]`. Median own-price elasticity
−1.2248: allocation 81%, incidence 10%, quantity 9%.

**Against baselines**, all on identical choice sets (chance −3.85 within category, −8.04
full catalogue):

| | within-category | full catalogue |
|---|---|---|
| HPF | −3.1866 | — |
| B-Emb | −3.0939 | — |
| SHOPPER (reimplemented, `scripts/eval/45_shopper.py`) | −2.8262 | −7.0501 |
| **ours** | **−2.5058** | **−6.7402** |

+0.586 nats over B-Emb [+0.563, +0.609]; +0.320 over SHOPPER within category and **+0.310
on the full catalogue**, which is SHOPPER's own normalisation — that is the point of
scoring both ways. Caveats: one tuned scalar per baseline against 870k parameters, and
SHOPPER is a PyTorch reimplementation fitted by MAP, not the authors' variational CUDA
code. A floor, not a benchmark result.

---

## 3a. Representations, coupon targeting, MDP policies

**Product embeddings — real.** 10-NN sub-commodity purity **0.1126** [0.1041, 0.1260],
against a permutation null at 0.0039 and a random embedding at 0.0029, chance 0.0040 —
**27.8× chance with both nulls at chance**. Same-sub AUC is **0.9384** among pairs co-bought
≥8× but only 0.5919 over all pairs: α is organised where the data is dense and unstructured
where it is sparse. Nearest neighbours are often *basket* relations, not taxonomy
(BANANAS → KIDS COOKIES), which is what the tied interaction asks for.
`scripts/eval/47_embeddings_verified.py`, figure in `figures/embeddings_verified.png`.

**Household embeddings — nothing.** Linear *and* gradient-boosted probes on θ, c_user, γ
fail to beat the majority class on income, household size, age, children or home ownership;
best boosted probe is 0.037 **below** it, and on income the shuffled control outscores the
real labels. Not recoverable linearly or non-linearly at n = 655.

**Coupon targeting — a screen, not a score.** Spearman between predicted `g` and real
held-out slope is **−0.122**; aggregating to products (−0.154) or households (−0.169) does
not help. But the sign split is validated: `g > 0` pairs show a real slope of **−0.188**,
`g ≤ 0` pairs **+0.003**. So the 21% of pairs with the "wrong" sign are the model correctly
flagging non-responders. **Use it as a binary screen; do not rank within it.**

**MDP policies — single-step yes, rollouts drift.** The never-bought state is 18.93%
against a real 13.89% (not the 46% a measurement bug once implied). But the excess
compounds: cumulative new distinct products run **1.156× real after one trip and 1.261×
after twelve** (`scripts/eval/46_horizon.py`), and that is a lower bound because generation reads recency
from the real history rather than its own output. Three structural blockers are untouched:
prices are exogenous so a policy gets no feedback; `Δlog p` is centred on the training
window so a level shift leaves support; and there is no budget constraint. Single-step
counterfactuals are supported; multi-step rollouts are usable over short horizons at most.

---

## 4. Two things that will bite you

**`--l2-incidence 1e-4` is not optional.** With the shared penalty, `c_user`/`c_cat`
collapse to effective **rank 2 and 3** out of 64 — 144,512 parameters delivering one number
per household times one per category, which cannot express "buys nappies, never cat food".
Averaged over all dimensions the data gradient beats the penalty 5.8–17×, but that average
is dominated by the one direction that is learning; on the other 61 the data gradient is
near zero and the penalty is the only force. Separating it gives rank 23 and 41, improves
incidence NLL 0.1249 → 0.1199 against a 0.1132 oracle floor, and cuts the novel-category
share 22.5% → 19.7%.

**`generate_baskets()` returns `.trips`. Use it.** When `trips=` is not passed it selects a
*random* subset of test baskets. Attributing the k-th generated basket to test basket k
compares against a **different household's history** — two random households share ~15% of
their products. That bug produced an apparent 46%-vs-14% novelty catastrophe and cost four
wrong diagnoses and three pointless interventions before it was caught. The function now
carries its trip indices; never index positionally.

---

## 5. Running it

Measured on a MacBook Pro M5 Pro, 15 cores, 24 GB.

**Use one thread per job.** The tensors are small (`[1476, 225]`), so threading buys sync
overhead and no parallelism: 169 ms/iteration at 1 thread against 363 ms at 5, with CPU
efficiency 1.00 versus 0.26. Run 6–8 jobs at `OMP_NUM_THREADS=1`, not 2 jobs at 5.

Never launch a second wave while swap from the first is still occupied — 12 concurrent jobs
once exhausted all 12 GB of swap, collapsed throughput 32×, and crashed the machine.
`/tmp/swapguard.sh` aborts training above 2 GB of swap. Anything that must survive belongs
in the repo; `/private/tmp` does not survive a reboot.

Scripts are grouped by role — `scripts/pipeline`, `scripts/model`, `scripts/eval` — with
everything the two documents do not use in `archive/`. See `scripts/README.md` for the
script-to-section map. **Run each from its own directory**; paths are resolved relative to
it, and `eval/` adds `../model` to `sys.path`.

```bash
export OMP_NUM_THREADS=1

cd scripts/model
python3 27_nested_basket.py --label mymodel --iters 6000 --l2-incidence 1e-4   # ~25 min

cd ../eval
python3 33_verify_equations.py --label mymodel      # must print "all equations verified"
python3 43_bootstrap.py --labels mymodel ...        # CIs, no refitting needed
python3 44_baselines_exact.py --labels mymodel      # HPF and B-Emb
python3 45_shopper.py --labels mymodel              # SHOPPER, both metrics
python3 46_horizon.py --label mymodel               # rollout drift
python3 47_embeddings_verified.py --label mymodel   # embeddings + nulls + figure
python3 42_limitations.py --label mymodel           # NOTE: defaults to a superseded label
python3 34_generator_eval.py --label mymodel --n-trips 24768 --top-items 500
```

The two non-default arguments on the last line are what the published §7 numbers were
produced with; the defaults (6000 / 300) give a smaller sample and different absolute
figures.

Current fitted labels: `ps_nested` (main), `ps_off` (no persistence), `ps_pl` (placebo),
`ps_s2..ps_s7` (seed replicates), `rk_base`/`rk_lowl2`/`rk_ncat`/`rk_both` (the penalty
2×2), `pcd_lo`/`pcd_hi`, `hab_nested`.

---

## 6. What is not established

1. **Refitting uncertainty.** §1 of the experiments page bootstraps the *evaluation*; a
   full bootstrap needs a refit per replicate. Seed replicates are the partial substitute.
2. **The test set has been used repeatedly** across model comparisons. Nothing reserves it
   for a single final evaluation.
3. **Identification is asserted, not tested** — a 2,016-dimensional rotational invariance in
   α and θ, an unconstrained sign on `γ·β` (21% of pairs negative), and category levels
   identified only through the noisy inclusive value.
4. **The assortment threshold is unjustified** and the elasticity is first-order sensitive
   to it, running −0.61 to 0.00 as the rule tightens. Part of that is mechanical.
5. **The IV estimator's "bias is constant per category"** assumption is stated and never
   checked.
6. **Generation is distinguishable from real** at classifier AUC 0.81, and the residual
   novelty excess accumulates over a horizon (1.156 → 1.261 over 12 trips, a lower bound).
   Co-occurrence rank correlation reaches +0.082 against real; within-basket pairs sharing a
   sub-commodity run 0.0375 generated against 0.0646 real.
7. **SHOPPER is a reimplementation** fitted by MAP, not the authors' variational code.

---

## 8. How good is this, against what it was for?

The goal was a retail simulator plus a model that answers counterfactual questions about
price. Scored against that, not against effort.

### Counterfactual price model — 7/10

**Holds up.** Price coefficient +0.695, refit sd 0.0104 over seven fits. The placebo
retains **exactly 0.0%** on both margins with an elasticity interval of
`[−0.0029, +0.0000]` — it contains zero and essentially nothing else. The three-margin
decomposition (allocation 81% / incidence 10% / quantity 9%) is an exact derivative of the
fitted model, autograd-verified, and the quantity margin is unreachable by any
binary-purchase model. Beats SHOPPER by 0.310 nats on SHOPPER's own normalisation.

**Does not hold up.** The causal claim is weaker than the framing suggests. **Permuting a
price series destroys the price–demand association whether it is causal or confounded.** It
rules out a spurious constant; it cannot separate "price moves demand" from "the retailer
times promotions to demand", which is the actual endogeneity concern for within-product
price variation. There is no instrument and no discontinuity anywhere in this work.

Three further cracks: the elasticity runs **−0.61 to 0.00** as the assortment threshold
tightens and nobody has justified the threshold; prices are reconstructed from the very
transactions being scored, with 41% of held-out rows the sole observation of their
item-day; and 21% of `γ·β` pairs carry the economically wrong sign, unconstrained.

*A well-specified demand model whose price parameter is stable and better than the
published alternative. Calling it causal is a stretch the design does not support.*

### Retail simulator — 4/10

Marginals match — basket size to 2%, item marginals +0.891, price response recovered at
76–78%. But a discriminator separates real from generated baskets at **AUC 0.81**, which is
the test that matters. Basket-size TVD is 0.34 despite the means agreeing. Co-occurrence
rank correlation tops out at **+0.082**. Fine substitution is 40% under-produced. Novelty
runs ~20% high and compounds to **1.26× over twelve trips**.

For learning a pricing policy it fails on structure, not just fit: prices are exogenous so
a policy gets no feedback; `Δlog p` is centred on the training window so a level shift
leaves support; there is no budget constraint. Single-step counterfactuals are supported.
Multi-step rollouts are not.

### What is genuinely defensible

The process more than either deliverable. The specification is coherent and the code
conforms to it, checked fifteen ways. Every headline number carries two kinds of
uncertainty, separated. Claims are tested against nulls — the product-embedding result
survives a permutation null, the household negative survives a non-linear probe. Four
negative results are recorded rather than buried (§7). Reversals are on the page: the nest
changed sign once the metric was fixed.

### What not to put in front of a reviewer

- "Is the price response causal?" as a section title.
- The simulator as policy-ready.
- Anything resting on the test set, which has been looked at roughly thirty times.
- The `γ·β` sign and the 2,016-dimensional rotation — both stated, neither tested.

### Overall — 6/10, and the highest-value next step

A solid demand model with an honest audit trail, and a simulator that isn't one yet. The
single highest-value next step is **not more modelling — it is finding real price variation
to identify against.**

dunnhumby ships `causal_data.csv` (664 MB): `PRODUCT_ID × STORE_ID × WEEK_NO` with
**`display`** (in-store placement, codes 0–7) and **`mailer`** (weekly circular placement,
codes 0/A/C/D/F/H/J/L), covering 115 stores over weeks 9–101. Promotional placement is
plausibly excludable from a household's product preference while strongly shifting the
price it faces — the shape of an instrument, or at minimum the basis for a
difference-in-differences on placement rather than on price.

It is read by `archive/normalizing_flow/04_extras.py` but **never reaches the basket model**: `basket_input/` has no
display or mailer field, so stages 22 and 27 have never seen it. Wiring it in and using it
for identification would move the price model from 7 to something defensible far faster
than another architecture change.

---

## 7. Negative results worth not repeating

- **Persistent contrastive divergence** moved generation by nothing at either weight tested
  while costing 0.03–0.04 nats. Implemented and left behind `--pcd 0`. It corrects energy
  *ranking*; the gap is in flatness, which ranking does not touch.
- **An explicit household–category habit feature** contributed +0.006 nats and changed
  generation not at all. The incidence NLL has only ~0.012 nats of headroom to an oracle,
  so the loss cannot reward it.
- **Sharpening the generation softmax** (temperature 1.0 → 0.25) moved novelty 46% → 43%
  while inflating basket size 4.6×.
- **Sampling more categories per trip** (16 → 48) did not fix the rank collapse; the penalty
  did.
- Both PCD and the habit feature were re-measured after the attribution bug was fixed. Both
  nulls stand.
