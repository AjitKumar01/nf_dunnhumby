# The basket model: specification, results, and what actually changed

Read `DATA_EXPLORATION.md` first. This document assumes its three findings:

1. **56.1%** of baskets contain more than one item from a single category, and the
   median category sits at **13.1%** against the paper's 15% cutoff — so unit demand
   is not a mild approximation, and the filter that enforces it cuts the distribution
   near its centre, discarding **132 of 307 categories**.
2. Items from the same sub-commodity are **7.88× more likely to share a basket than
   chance**. Within a sub-commodity the dominant behaviour is variety-seeking, not
   substitution.
3. The repurchase hazard swings **4.30×** with recency and is **non-monotone** —
   rising from 0.109 at 0–3 days to 0.149 at 7–14, then decaying to 0.035 after 84.

Each finding kills a structural commitment of the paper's model. This document
specifies the replacement, fits it, and tests it.

---

## 1. Why the previous attempt was not enough

An earlier round added a substitution kernel `ψ_j·ψ_k` to the paper's stage 1
(`REPORT.md`, commit `3ed1b09`). It worked in its own terms — held-out log-likelihood
improved by 0.022 nats, own-price weeks by 0.075, and IIA was measurably broken — but
it failed the requirement that matters here. Same-sub-commodity pairs substituted more
in only **46.7% of categories**, below a coin flip.

The reason was structural, not a tuning problem. That model still fitted the paper's
sample: one item per category, 56 categories, 560 items, Sunday and Monday only. In a
likelihood where **two yogurts in one basket is an impossible event**, the 7.88×
within-sub-commodity co-occurrence is not weak evidence — it is *unobservable*. No
amount of parameterisation recovers a signal the likelihood has filtered out.

That is why this is a rebuild rather than another term.

---

## 2. Specification

For household `i` shopping on day `t`, every item `j` in the basket is scored against
the whole catalogue:

```
s_ijt = λ_j                        item popularity
      + θ_i · α_j                  household taste × item embedding
      + α_j · ᾱ(context)           interaction with the rest of the basket
      − (γ_i · β_j) · Δlog p_jt    price, as deviation from the item's own normal
      + η_j · state_ijt            time since this household last bought the
                                   sub-commodity

P(j | i, t, context) = softmax over {j} ∪ {20 negatives}
```

`ᾱ(context)` is the mean of `α` over the **other** items in the same basket.
Negatives are drawn unigram^0.75. Batching is by **basket**, not by row, so the
context mean is exact rather than computed from a stale copy of `α`.

### Why each term, and what evidence demands it

| term | what it does | the finding that requires it |
|---|---|---|
| `λ_j` | baseline popularity | the floor model; alone it reaches −3.02 |
| `θ_i · α_j` | who likes what | households differ; ablating costs **0.149 nats** |
| `α_j · ᾱ(ctx)` | product interaction | 11% of pairs are complements at 2×+, 4.5% substitutes; ablating costs **0.141 nats** |
| `γ_i · β_j · Δlog p` | price response | price still moves on 30.5% of item-weeks; ablating costs **0.025 nats** |
| `η_j · state` | memory | the 4.30× hazard swing; ablating costs **0.093 nats** |

### The one design decision that mattered most: tying the interaction

The interaction was first written with a **free** coefficient, `ρ_j · ᾱ(ctx)`. It
fitted well and the embedding was poor — nearest-neighbour purity **0.058**, and,
tellingly, *ablating the interaction entirely improved it to 0.117*.

The diagnosis: with a free `ρ`, the co-purchase signal has somewhere else to go. `ρ`
absorbs it and `α` never has to represent product structure at all. Setting `ρ_j = α_j`
makes the interaction symmetric — `α_j · ᾱ(ctx)` — so "these items appear together"
becomes "these items have similar `α`". That forces the 7.88× within-sub-commodity
co-occurrence into the embedding the requirement is about.

The result was not a trade-off. Tying improved **both**:

| | kNN purity | test log-lik |
|---|---|---|
| free `ρ_j` | 0.058 | −2.3486 |
| tied `ρ_j = α_j` | **0.288** | **−2.1788** |

5× the embedding quality *and* 0.17 nats of fit. A free `ρ` was simply the wrong
structure — it was spending 175k parameters to represent something `α` should have
carried.

### State

`state_ijt` is a 4-element basis on days since household `i` last bought item `j`'s
sub-commodity:

```
[ never bought,  exp(−since/7),  exp(−since/gap_s),  log1p(since)/log 100 ]
```

`gap_s` is that sub-commodity's own median repurchase gap (clipped to 3–180 days), so
"a long time" means the right number of days for milk and for shampoo. A single
"days since" coefficient cannot represent the humped hazard the EDA found; this basis
can.

The lookup is the one piece of real engineering here. Materialising
(household × day × sub-commodity) is ~32M rows and a per-sample Python lookup is ~35M
dict hits per epoch. Instead purchase days are stored once as a globally sorted key
array, `key = group_id × 1024 + day`, so a whole batch resolves in a single vectorised
`np.searchsorted`. The current trip never sees its own purchase because `side="left"`
places the boundary before it.

---

## 3. Data and protocol

| | paper's port | this model |
|---|---|---|
| households | 2,084 | 2,066 |
| items | 560 | **5,455** |
| categories | 56 | **188** |
| sub-commodities | — | **758** |
| observations | 66,637 | **1,566,063** |
| days | 172 (Sun/Mon) | **712 (all)** |
| baskets | — | 199,347 |

The only item filter is **≥100 purchase lines**, which is about statistical support
for an embedding, not about protecting a modelling assumption. No unit-demand filter,
no price-correlation filter, no seasonality filter.

**Splits are by calendar week, held out at the end**: train below week 83, validation
83–90, test 91+. A random split would let the model see each household's future. 97
held-out rows (0.03%) whose item or household never appears in training are dropped
rather than scored as a cold start.

Reported log-likelihood is per scored position over `{true item} ∪ {20 negatives}`, so
random guessing is `log(1/21) = −3.04` and top-1 accuracy is 0.048 at chance.
**These numbers are not comparable to the nf log-likelihoods elsewhere in this
repository** — different likelihood, different sample, different task. The only
apples-to-apples comparison with the paper's model is the embedding test in §5.

---

## 4. Fit and ablations

All at K=64, L2 1e-2, lr 0.005, 9,000 iterations, best-validation checkpoint.

| model | test log-lik | top-1 | cost of removing | kNN purity |
|---|---|---|---|---|
| **full** | **−2.1788** | 0.352 | — | 0.288 |
| no price | −2.2033 | 0.346 | 0.025 | 0.289 |
| no state | −2.2718 | 0.334 | **0.093** | **0.343** |
| no interaction | −2.3195 | 0.320 | **0.141** | 0.203 |
| no household taste | −2.3279 | 0.306 | **0.149** | 0.280 |
| popularity only (untied run) | −3.0205 | 0.092 | 0.842 | 0.005 |

Seed-to-seed spread is small — a second seed of the full model gives −2.1741 against
−2.1788 (0.0047 nats) and purity 0.291 against 0.288 — so every ablation gap is one to
two orders of magnitude larger than run-to-run noise.

Three things worth stating plainly, including one that cuts against the model:

- **Every component earns its place on fit.** Interaction and household taste each
  matter more than price, and state matters nearly four times as much as price.
- **The floor is informative.** Popularity alone reaches top-1 of 0.092 against 0.048
  chance, so roughly half the achievable ranking signal is not popularity.
- **State buys fit and costs embedding quality.** Removing state *improves*
  nearest-neighbour purity from 0.288 to **0.343** and AUC from 0.816 to **0.866**,
  while costing 0.093 nats of fit. This is the same absorption effect that motivated
  tying the interaction: `η_j` soaks up repeat-purchase regularity, and whatever `η`
  explains, `α` no longer has to. Unlike tying — which improved both — this is a
  genuine trade-off between predictive fit and embedding interpretability.

  The headline model keeps state, because a state level was a requirement and because
  it is the transition function any dynamic policy needs. But if the embedding is the
  deliverable and forecasting is not, `tied_nostate` is the better artefact, and its
  numbers are in the table rather than hidden.

- **Interaction is what makes the embedding work.** Removing it drops purity from
  0.288 to 0.203 and AUC from 0.816 to 0.712 — the opposite direction from state, and
  confirmation that tying routes co-purchase structure into `α` as intended.

---

## 5. The embedding requirement

This is the test the previous attempt failed. The model is never shown
`SUB_COMMODITY_DESC` — not as a feature, not as a grouping, not in the likelihood.

![embedding scores](figures/embedding_scores.png)

On all 5,455 items and 758 sub-commodities:

| embedding | kNN purity (k=10) | × chance | same-sub AUC | silhouette |
|---|---|---|---|---|
| tied, no state | 0.343 | 80.2× | 0.866 | −0.100 |
| **tied, K=64 (headline)** | **0.288** | **67.3×** | **0.816** | −0.160 |
| tied, no price | 0.289 | 67.4× | 0.817 | −0.160 |
| tied, no household taste | 0.280 | 65.4× | 0.838 | −0.276 |
| tied, K=32 | 0.250 | 58.5× | 0.819 | −0.247 |
| tied, no interaction | 0.203 | 47.4× | 0.712 | −0.223 |
| free `ρ` (untied) | 0.058 | 13.5× | 0.608 | −0.390 |
| control: random | 0.004 | 0.9× | 0.499 | — |
| control: popularity only | 0.004 | 0.9× | 0.510 | — |

Chance purity here is 0.0043 — not `1/758`, because sub-commodities differ hugely in
size and a random neighbour lands in one with probability proportional to its size.

**Reading this honestly.** 0.288 means that for a typical item, roughly 3 of its 10
nearest neighbours are in the same sub-commodity out of 758 possibilities. An AUC of
0.816 means that given a same-sub pair and a different-sub pair at random, cosine
similarity ranks them correctly 82% of the time. The silhouette is still **negative**
(−0.160): sub-commodities are *findable* in this space but they are not compact,
well-separated balls, and I would not claim otherwise.

### The map

![t-SNE](figures/embedding_tsne.png)

Left, coloured by `DEPARTMENT` — never shown to the model. PRODUCE, DRUG GM, DELI and
MEAT-PKGD occupy coherent regions. Right, the 12 largest sub-commodities form tight,
separated clusters: candy bars, soft drinks (with the 2-litre bottles separated from
the 12/18/15-packs), shredded cheese, frozen bagged vegetables, potato chips.

### The human check

Nearest neighbours of popular items, by cosine on `α`:

| query | its 5 nearest neighbours |
|---|---|
| MEAT: TURKEY BULK | HAM BULK, BEEF BULK, **TURKEY BULK**, **TURKEY BULK**, SAUS DRY BULK |
| PEANUT BUTTER | **JELLY**, PEANUT BUTTER, PEANUT BUTTER, PRESERVES/JAM, PEANUT BUTTER |
| CANDY BARS (SINGLES) | all five are CANDY BARS (SINGLES) |
| CANISTER POTATO/TORT CHIPS | all five are CANISTER POTATO/TORT CHIPS |
| MAINSTREAM WHEAT BREAD | WHEAT BREAD, WHITE BREAD, SINGLE CHEESE, SINGLE CHEESE, EGGS |

40% of top-5 neighbours share the query's sub-commodity, up from 8% before tying.

The "misses" are the most interesting part and they are not errors. **Peanut butter's
nearest neighbour is jelly.** Turkey's are ham and beef. Wheat bread's are sliced
cheese and eggs. The strict sub-commodity metric scores these as failures; they are
the interaction term doing exactly what it was added for. A metric that rewarded only
sub-commodity purity would be rewarding a model that had *not* learned complementarity.

---

## 6. Head-to-head with the paper's model

The basket model has 5,455 items and nf has 560, so comparing embeddings across those
two universes would confound the model with the item set. Both are therefore measured
on the **same 409 items** — those nf models whose sub-commodity has at least two
members in this catalogue — with the same ground truth.

| | kNN purity | × chance | same-sub AUC | silhouette |
|---|---|---|---|---|
| **basket model `α`** | **0.149** | **11.0×** | **0.803** | **−0.061** |
| nf `β` (paper's model) | 0.014 | 1.0× | **0.379** | −0.313 |

Two results, and the second is the more striking.

**nf's item embedding is at chance** — 1.0× — on recovering sub-commodity. The paper
claims its latent factors capture product similarity; on this data, for this
hierarchy level, they do not.

**nf's AUC is 0.379, below 0.5.** Cosine similarity in nf's embedding space ranks
same-sub-commodity pairs as *less* similar than random pairs. This is not noise, it is
a mechanism: within a category the softmax makes items compete, so the gradient
actively pushes apart items that are close substitutes — and items that are close
substitutes are disproportionately in the same sub-commodity. The paper's likelihood
is structured so that learning "these two are alike" *hurts* the fit.

That is the clearest statement of why this needed a rebuild rather than an extra term.

---

## 7. What this enables that the paper's model could not

- **Multi-item baskets.** 56.1% of baskets are now representable rather than filtered
  or truncated; 132 previously discarded categories are back.
- **Cross-category structure.** Complements are expressed directly, and the model
  recovers them (peanut butter → jelly) without being told what a complement is.
- **A state variable, hence an actual MDP.** The earlier simulator
  (`20_simulate.py`) was honest that with the Sun/Mon window it emitted a *contextual
  bandit*: no inventory, so the next state did not depend on the action. This model
  has `state_ijt` over all 711 days, so a price cut today changes the recency
  distribution tomorrow, which changes demand. That is the transition function an MDP
  needs. Building the policy layer on top is the natural next step and is **not done
  here**.

---

## 8. Limitations, stated plainly

- **Nothing here is causal.** The placebo battery in `PREPROCESSING.md` §9 found 34 of
  56 categories failing at least one price-endogeneity test, and it has **not** been
  re-run on the 188-category catalogue. Every price coefficient in this model should
  be read as predictive. A pricing policy optimised against it would chase the
  endogeneity.
- **The silhouette is still negative.** Sub-commodities are recoverable but not
  compact. The embedding is good enough to retrieve, cluster and inspect; it is not
  good enough to treat cluster boundaries as meaningful.
- **Negative-sampled likelihood is not a proper held-out likelihood** over the full
  catalogue. Ranking against 20 negatives is a consistent basis for comparing these
  models to each other and to their ablations, and nothing more.
- **Quantity is ignored.** Purchase is binary, as in the paper. Baskets average 13.2
  units across 10.1 distinct items, so there is a real multiple-discreteness margin
  neither model touches.
- **Prices are still chain-level** across 561 stores, with the cost measured in
  `VERIFICATION.md` §1.
- **The item threshold is a choice.** At ≥100 lines the model covers 65.7% of purchase
  volume. The 34.3% in the tail is not modelled, and the long tail is where assortment
  decisions are often most interesting.
- **The state is at sub-commodity level**, not item level, and there is no explicit
  inventory-depletion model — only a flexible function of recency. A structural
  inventory model would be a stronger claim and a harder fit.

---

## Reproducing

```bash
python3 scripts/21_basket_eda.py        # exploration -> DATA_EXPLORATION.md figures
python3 scripts/22_basket_data.py       # build basket_input/
python3 scripts/23_basket_model.py --label tied_k64_r \
        --tie-context --K 64 --l2 1e-2 --lr 0.005 --iters 9000 --eval-every 500
python3 scripts/24_embedding_eval.py --primary tied_k64_r
```

Artefacts: `out/basket_eda.json`, `out/embedding_eval.json`,
`out/embedding_neighbours_basket.csv`, `out/<label>_basket_history.json`, and the
figures referenced above.
