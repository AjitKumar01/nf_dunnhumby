# The nested basket model

**Status: fitted, results pending validation.** The specification, the data build and
the engineering are settled and described here in full. The results section is
deliberately empty until the model demonstrably meets the acceptance criteria in §7 —
placing numbers there before that would be reporting a work in progress as a finding.

This document is kept current as the model changes; §10 is the changelog.

---

## 1. Why this model exists

`BASKET_MODEL.md` describes a flat basket model that fixed three things about the
paper's port — multi-item baskets, product interactions, and household state — and
scored well. It also, without saying so, threw away three things it should not have.

| what was dropped | why it matters | evidence |
|---|---|---|
| **the nest** | with no incidence layer the model can rank items but cannot say whether a household buys from a category *at all*. It cannot answer "does cutting Tide's price grow total detergent volume, or just move share from Gain" — the question a retailer actually asks | structural: `23_basket_model.py` has no category stage |
| **quantity** | purchase was binary. 22.3% of (basket, item) rows buy more than one unit, and those rows carry **42.6% of all units**. A price cut works through two channels — more buyers *and* more units each — and only the first was modelled | units-per-buyer elasticity **−0.219** |
| **stores** | prices pooled to chain level, assortment ignored. Scoring an item a store never stocked as a *rejected* alternative is a specification error: it was not there to reject | **15.8%** of store-item-weeks differ >1c from the chain price (sd $0.121); median store carries **63%** of the catalogue, p10 carries 39% |

Dropping the within-category softmax was justified — unit demand fails in 56% of
baskets. Dropping the *nest* was not: unit demand failing says nothing about whether
incidence is a separate decision from allocation. Those were conflated.

---

## 2. Data

Built by `scripts/22_basket_data.py` into `basket_input/`.

| | value |
|---|---|
| households | 2,066 |
| items | 5,455 |
| categories (`COMMODITY_DESC`) | 188 |
| sub-commodities | 758 |
| **stores** | **115** |
| (basket, item) rows | 1,566,063 |
| baskets | 199,347 |
| days | 712 (all) |
| units > 1 | 22.3% of rows, 42.6% of units |
| assortment | 57.2% of the item × store grid |
| store-level price cells | 244,880 |

The only item filter is **≥100 purchase lines** — statistical support for an embedding,
not protection of a modelling assumption. Splits are by calendar week with the last 20
held out (train <83, validation 83–90, test 91+).

### Files

| file | contents |
|---|---|
| `baskets.parquet` | one row per (basket, item): `units` as a count, `store_id`, split label |
| `items.parquet` | id maps plus the held-out labels the model never sees |
| `log_price.npy`, `log_price_dev.npy` | item × day log price, raw and centred within item |
| `store_price.npz` | sparse store-level log-price deviations, plus the `carried` availability mask |
| `state.npz` | sorted purchase-day keys for the recency lookup |
| `meta.json` | sizes, split boundaries, filter settings |

### Two data decisions worth defending

**Units are capped at 12.** dunnhumby's `QUANTITY` is unreliable for weighed goods and
the far tail is bulk lines. The cap touches a small share of rows and prevents a
handful of 40-unit lines dominating a Poisson likelihood.

**Availability threshold is 1 sale, not 3.** An item a store ever sold was, by
definition, available there. A threshold of 3 marked genuine purchases as unavailable,
which made the inclusive value `−1e9` for categories a household demonstrably bought
from and blew the incidence loss up to 5.1 million. Assortment coverage went from a
spurious 24.7% to 57.2% when this was corrected.

---

## 3. Specification

For household `i`, on day `t`, at store `s`.

### Item utility (shared by all three heads)

```
u_ijt = λ_j                          item popularity
      + θ_i · α_j                    household taste × item embedding
      + α_j · ᾱ(context)             interaction with the rest of the basket (tied)
      − (γ_i · β_j) · Δlog p_jst     price, chain deviation + store deviation
      + η_j · state_ijt              recency of this sub-commodity for this household
      + μ_j · δ_w                    seasonality, week-of-year
      + ζ_j · ξ_s                    store × item affinity (low rank)
```

`α_j · ᾱ(context)` is the **tied** interaction carried over from `BASKET_MODEL.md` §2:
using `α` itself rather than a free `ρ` is what forces co-purchase structure into the
embedding the sub-commodity test reads. With a free `ρ` the embedding scored 0.058
purity; tied, 0.302.

### The nest, and how it survives multi-item baskets

The paper gets its nest from a within-category softmax plus an outside good — exactly
what unit demand buys it. Drop unit demand and that construction is gone, but the nest
is not. It survives as a **Poisson–multinomial factorisation**:

```
total units from category c     Q_ict ~ Poisson(exp(a_ic + κ_c · IV_ict))
allocation across its items     multinomial(softmax_j u_ijt)
IV_ict = log Σ_{j ∈ c, stocked at s} exp(u_ijt)
```

Poisson–multinomial factorises, so this is equivalent to independent counts

```
q_ijt ~ Poisson(exp(a_ic + (κ_c − 1)·IV_ict + u_ijt))
```

and **κ is a nesting coefficient with the paper's interpretation**:

| κ | meaning |
|---|---|
| **= 1** | IV cancels. Category volume is whatever the items sum to — no expansion (IIA) |
| **= 0** | category volume is fixed. A price cut only moves share |
| **> 1** | the category expands more than proportionally |

κ is estimated per category, parameterised as `softplus(κ_raw)` initialised at exactly
1.0 — the "IV cancels" point — so the data has to move it in either direction.

### Three heads

| head | form | what it identifies |
|---|---|---|
| **item** | softmax over `{chosen} ∪ {20 negatives}` | allocation within category |
| **quantity** | `units − 1 ~ Poisson(exp(z))`, own price coefficient `γ^q · β^q` | the quantity margin |
| **incidence** | Bernoulli per (trip, category), logit includes `κ_c · IV_ict` | category expansion |

The quantity head has its **own** price coefficient rather than sharing the item one.
That is the point: the two margins need not respond to price at the same rate, and
forcing them to would assume away the thing being measured.

---

## 4. Where stores enter

Three distinct channels, which must be kept separate because they are not equally
meaningful:

1. **Store-level price deviation**, added to the chain deviation where the
   store-week was observed (2.3% of the grid; the rest falls back to chain price).
   This is information.
2. **Store × item affinity** `ζ_j · ξ_s`, low rank — format and assortment differ.
   This is information.
3. **The availability mask** — unstocked items leave the choice set entirely.
   This makes the ranking task **mechanically easier**, because there are fewer real
   competitors in the softmax denominator.

Ablations `--no-store`, `--no-store-price` and `--avail-only` exist specifically to
separate (3) from (1) and (2). **Any claim about how much stores are worth must
report that split**, or it is claiming credit for a smaller choice set.

---

## 5. Fitting

`scripts/27_nested_basket.py`. Adam, cosine-decayed learning rate, best-validation
checkpoint on a combined score across the three heads.

### Negative sampling

Unigram^0.75 over training purchases, 20 negatives per positive, drawn only from items
the trip's store actually stocks.

### The inclusive value is sampled, not summed

Every category is padded to the largest one (**225 items**) although the median has
**15**, so a dense IV block spends most of its work on padding and cost **5.4× the item
head**. Instead `iv_cap` (default 32) items are sampled per category and the sum scaled
by `n_c / m` — an unbiased estimator of `Σ_j exp(u_j)`, at fixed cost per category.

Without this the run was on track for **9 hours**; with it, 8.4 minutes.

### Incidence is case-control sampled, and corrected

The true incidence base rate is **6.1 of 188 categories = 3.25%**. Sampling categories
uniformly would give 0.13 positives per trip, so the sampler takes a few bought and a
few not — which over-samples positives about **30×**.

An uncorrected head is therefore calibrated to the *sample*, not to reality. The
symptom was unmissable once baskets were generated: 58 categories per basket against a
real 6.5, a **logit error of 3.39** which is almost exactly `log(30)`.

The fix is the standard case-control correction: an offset `log(π₁/π₀)` is added inside
the logit **during training** and dropped at prediction, so probabilities come out on
the population scale. It is computed per trip, since it depends on how many categories
that trip bought.

---

## 6. Engineering

### Two vectorised lookups

Both the household state and the store price are sparse, high-cardinality lookups that
would be ruinous done naively — materialising (household × day × sub-commodity) is ~32
million rows, and a per-sample Python dict would be ~35 million hits per epoch.

Both use the same trick: values are stored once as a **globally sorted key array**
(`key = group_id · stride + index`), so a whole batch resolves in a single
`np.searchsorted`. The stride exceeds any index, so ordering never crosses a group
boundary.

### Measured performance

| model | iterations | wall time | ms/iter |
|---|---|---|---|
| flat (`one`) | 12,000 | 3.5 min | 17.7 |
| **nested** | 12,000 | **8.4 min** | **42.1** |

2.4× the flat model for three heads instead of one. Six variants ≈ 50 minutes.

**Known bottleneck:** `np.searchsorted` is ~40% of every step (~60,000 queries per
iteration). State features are *data, not parameters* — they depend only on
(household, sub-commodity, day) and never change during training — yet they are
recomputed every iteration for every sampled candidate. Deduplicating repeated
candidates within a batch would recover roughly half of that, ~20% of the step. Not
yet done.

### Scalability

| dimension | cost |
|---|---|
| catalogue size | **independent** — negative sampling fixes candidates at 21 |
| households | **independent** — only the embedding table grows |
| baskets | **independent per step** |
| categories | capped at `n_cat` sampled per trip |
| items per category | capped at `iv_cap` by the sampled IV |

Nothing is quadratic in items or households.

---

## 7. Acceptance criteria

The model is not "done" until each of these is demonstrated. Results go in §8 only
as they are met.

| # | requirement | test | target | status |
|---|---|---|---|---|
| 1 | **multiple items** per category and across categories | likelihood admits multisets; no unit-demand filter | 188 categories retained, none dropped for unit demand | ✅ met by construction |
| 2 | **multiple quantities**, with interaction | quantity head with its own price coefficient | quantity elasticity distinguishable from zero and from the item one | ⏳ pending |
| 3 | **household state as a level** | recency basis per (household, sub-commodity) | ablation cost > seed noise | ⏳ pending |
| 4 | **nested theory retained** | κ estimated per category | κ identified, and its ablation reported | ⏳ pending |
| 5 | **store information used** | prices, affinity, availability | gain reported *split* from the mechanical availability effect | ⏳ pending |
| 6 | **embeddings meaningful** — similar products cluster by sub-commodity | `24_embedding_eval.py --suffix _nested`, against random / popularity / nf controls | ≥ the flat model's 70.6× chance and AUC 0.823 | ⏳ pending |
| 7 | **data generation** | roll incidence → item → units forward, compare basket shape with held-out | items, categories and units within ~10% of real | ⏳ pending |
| 8 | **what-if on price** | structural placebo + elasticity decomposition | placebo retains ~0% of the coefficient; decomposition sums | ⏳ pending |

---

## 8. Results

**Held pending validation.** Numbers exist for a first fit but the model has not yet
cleared §7, and two of its components have known defects (§9). They will be filled in
here once the criteria are met, with the same evidential standard as
`BASKET_MODEL.md`: seed replicates before claiming a gap, controls alongside every
embedding number, and the mechanical part of the store effect separated out.

---

## 9. Known issues

| issue | severity | state |
|---|---|---|
| **The generator draws one item per purchased category**, so generated items and categories coincide exactly. This contradicts the finding that motivated the rebuild — 56% of baskets hold multiple items from one category. The Poisson–multinomial structure supports drawing `Q_ic` and allocating it across items; the sampler does not yet | **high** — blocks criterion 7 | open |
| Generation over-produces by ~25–30% even after the calibration fix | medium | open |
| `np.searchsorted` is 40% of the step and is avoidable | low — performance only | open |
| Store-level prices cover only 2.3% of the item × store × week grid; the rest falls back to chain price | inherent to the data | documented, not fixable |
| Availability is proxied by "this store sold this item"; dunnhumby has no stock-out feed | inherent to the data | documented |
| Placebo battery (`25_basket_placebo.py`) was run on the chain-level panel, not the store-level one | medium | open |

---

## 10. Changelog

| when | change | why |
|---|---|---|
| initial | three-head nested model: incidence with κ, item choice, quantity counts; stores via price, affinity, availability | restores what `BASKET_MODEL.md` dropped |
| fix 1 | availability threshold 3 → 1 sale | a threshold of 3 marked genuine purchases unavailable; incidence loss hit 5.1M |
| fix 2 | inclusive value sampled (`iv_cap`) rather than summed over padded blocks | 5.4× the item head, mostly padding; 9-hour run → 8.4 min |
| fix 3 | case-control offset on the incidence logit | positives over-sampled 30×; generation produced 58 categories per basket against a real 6.5 |
| fix 4 | elasticity decomposition reported from means, not medians | medians do not decompose additively; the parts summed to 82% |
| fix 5 | `28` indexes the chosen item by position, not by mask | padding slots hold item id 0, so `blk == j` also fired on every pad slot when the chosen item was item 0 |
| fix 6 | placebo scrambles store deviations as well as the chain panel | leaving them real would leak genuine prices into the placebo |
| fix 7 | `--avail-only` / `--no-store-price` flags | the store gain conflates real information with a mechanically smaller choice set |
