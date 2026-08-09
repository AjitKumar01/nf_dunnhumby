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
| equations | 15 checks in `33_verify_equations.py`, all pass |

Key flags: `--l2-incidence 1e-4` (**required** — see §4), `--no-persist`, `--pcd`
(default 0, off), `--placebo-price permute`.

---

## 3. Headline results

Chance on the within-category conditional is `−log 47 = −3.85`, top-1 `2.1%`.

| | item | 95% CI on the gap |
|---|---|---|
| full model | **−2.4611** | — |
| no persistence | −2.5340 | [+0.0648, +0.0816] |
| original incidence penalty | −2.5327 | [+0.0635, +0.0804] |
| prices scrambled | −2.5174 | [+0.0489, +0.0626] |

Ablation ordering: store 0.337 ≫ persistence 0.073 > interaction 0.058 ≈ state 0.049 >
nest 0.019 ≫ quantity 0.000.

**Causal.** The placebo retains 0.0% of the price coefficient on both margins; its
allocation-elasticity interval is `[−0.0029, +0.0000]`. Median own-price elasticity
−1.2248: allocation 81%, incidence 10%, quantity 9%.

**Against baselines**, all on identical choice sets:

| | within-category | full catalogue |
|---|---|---|
| HPF | −3.1866 | — |
| B-Emb | −3.0939 | — |
| SHOPPER (reimplemented) | −2.8262 | −7.0501 |
| **ours** | **−2.5058** | **−6.7402** |

The margin survives on SHOPPER's own home ground, which is the point of scoring both ways.

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

```bash
cd scripts
export OMP_NUM_THREADS=1
python3 27_nested_basket.py --label mymodel --iters 6000 --l2-incidence 1e-4   # ~25 min
python3 33_verify_equations.py --label mymodel      # must print "all equations verified"
python3 43_bootstrap.py --labels mymodel ...        # CIs, no refitting needed
python3 44_baselines_exact.py --labels mymodel      # HPF and B-Emb
python3 45_shopper.py --labels mymodel              # SHOPPER, both metrics
```

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
   novelty excess accumulates over a horizon (ratio 1.156 → 1.261 over 12 trips, itself a
   lower bound since generation reads recency from the real history).
7. **SHOPPER is a reimplementation** fitted by MAP, not the authors' variational code.

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
