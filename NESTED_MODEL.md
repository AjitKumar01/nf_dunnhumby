# The nested basket model

`scripts/27_nested_basket.py` fits it, `scripts/28_nested_counterfactual.py` uses it,
`scripts/22_basket_data.py` builds its input. Every number below is read from an
artefact in `out/` or `basket_input/meta.json`; §8 names the file behind each result.

---

## 1. Why this model exists

`BASKET_MODEL.md` describes a flat basket model that fixed three things about the
paper's port — multi-item baskets, product interactions, household state — and scored
well. It also dropped three things it should not have.

| dropped | why it matters | evidence |
|---|---|---|
| **the nest** | with no category stage the model ranks items but cannot say whether a household buys from a category *at all*. It cannot answer "does cutting Tide's price grow total detergent volume, or move share from Gain" | structural: `23_basket_model.py` has no category stage |
| **quantity** | purchase was binary. **22.3%** of (basket, item) rows buy more than one unit, and those rows carry **42.6%** of all units | `DATA_EXPLORATION.md` §6.1: units-per-buyer elasticity −0.235 |
| **stores** | prices pooled to chain level, assortment ignored. Scoring an item a store never stocked as a *rejected* alternative is a specification error | 15.8% of store-item-weeks differ >1¢ from the chain price; the `carried` mask covers **57.2%** of the item × store grid |

Dropping the within-category softmax was justified. Unit demand — at most one item per
category, at most one unit of it — is violated by **68.2%** of the 199,345 baskets in
`basket_input`: 48.6% hold more than one item from some category, 57.6% have a line
buying more than one unit. Dropping the *nest* was not justified: unit demand failing
says nothing about whether incidence is a separate decision from allocation.

---

## 2. Data

`scripts/22_basket_data.py` → `basket_input/`. Sizes from `meta.json`:

| | value |
|---|---|
| households `N` | 2,066 |
| items `J` | 5,455 |
| categories `C` (`COMMODITY_DESC`) | 188 |
| sub-commodities `S` | 758 |
| stores | 115 |
| days `D` | 712 |
| (basket, item) rows | 1,566,063 |
| baskets | 199,345 |
| train / validation / test rows | 1,228,695 / 134,665 / 202,703 |
| units > 1 | 22.3% of rows, 42.6% of units |
| `carried` mask density | 57.2% |
| store-level price cells | 244,880 |
| item-days with a directly observed price | 24.7% |
| median repurchase gap | 28 days |

**Filters.** An item needs ≥100 purchase lines, which is statistical support for its own
embedding. A household needs 20–300 trips (the paper's own rule). Nothing is dropped
for violating unit demand, for co-moving prices, or for seasonality.

**Splits are by calendar week**, not a random slice: train < 83, validation
83–90, test ≥ 91. Held-out rows whose item or household never appears in
training are dropped rather than scored as a cold start the model was never given a
chance at.

### Files

| file | contents |
|---|---|
| `baskets.parquet` | one row per (basket, item): `units` as a count, `store_id`, `split` |
| `items.parquet` | id maps plus the held-out labels (sub-commodity, brand, manufacturer, department) the model never sees |
| `log_price.npy`, `log_price_dev.npy` | `[J, D]` log price by item and day, raw and centred within item |
| `store_price.npz` | sparse store-level log-price deviations, plus the `carried` availability mask |
| `state.npz` | sorted purchase-day keys for the recency lookup, and `sub_gap` |
| `meta.json` | sizes, split boundaries, filter settings |

### Two data decisions worth defending

**Units capped at 12.** dunnhumby's `QUANTITY` is unreliable for weighed goods and the
far tail is bulk lines. The cap prevents a handful of 40-unit lines dominating a
Poisson likelihood.

**Availability threshold is 1 sale, not 3** (`--min-store-lines`). An item a store ever
sold was by definition available there. A threshold of 3 marked genuine purchases
unavailable, which set `IV = −1e9` for categories a household demonstrably bought from
and blew the incidence loss up to 5.1 million.

---

## 3. Specification

Household `i`, day `t`, week-of-year `w`, store `s`, item `j`, category `c = cat(j)`.

### 3.1 Item utility

Every head is built on one quantity, `NestedModel.item_utility` (`27:260-272`):

$$
u_{ijt} \;=\; \lambda_j
\;+\; \theta_i^{\top}\alpha_j
\;+\; \alpha_j^{\top}\bar\alpha_j(S)
\;-\; \bigl(\gamma_i^{\top}\beta_j\bigr)\,\Delta\log p_{jst}
\;+\; \eta_j^{\top}x_{ijt}
\;+\; \mu_j^{\top}\delta_w
\;+\; \zeta_j^{\top}\xi_s
$$

| term | code | what it is |
|---|---|---|
| $\lambda_j$ | `lam[j]` | item popularity |
| $\theta_i^{\top}\alpha_j$ | `theta[i]·alpha[j]` | household taste × item embedding |
| $\alpha_j^{\top}\bar\alpha_j(S)$ | `alpha[j]·ctx` | basket interaction, tied (§3.3) |
| $-(\gamma_i^{\top}\beta_j)\,\Delta\log p_{jst}$ | `gamma[i]·beta[j]` | price |
| $\eta_j^{\top}x_{ijt}$ | `eta[j]·state` | recency basis (§3.2) |
| $\mu_j^{\top}\delta_w$ | `mu[j]·delta[w]` | seasonality, week-of-year |
| $\zeta_j^{\top}\xi_s$ | `item_store[j]·store_vec[s]` | store × item affinity |

Dimensions: $\alpha_j,\theta_i\in\mathbb{R}^{64}$; $\gamma_i,\beta_j\in\mathbb{R}^{8}$;
$\mu_j,\delta_w\in\mathbb{R}^{8}$; $\zeta_j,\xi_s\in\mathbb{R}^{4}$;
$\eta_j\in\mathbb{R}^{4}$. **870,392 parameters** in total.

**The price term is a deviation, not a level.** `22_basket_data.py:187` centres the log
price within item:

$$
\Delta\log p_{jt} \;=\; \log p_{jt} \;-\; \frac{1}{D}\sum_{t'=1}^{D}\log p_{jt'}
\qquad\qquad
\Delta\log p_{jst} \;=\; \Delta\log p_{jt} \;+\; d_{jsw}
$$

where $d_{jsw}$ is the sparse store-level deviation, **0** wherever that store-week was
never observed.

so the item's price level is absorbed by `λ_j`, and what identifies the response is
that item moving against its own normal price. With stores on, the store-level
deviation is added on top (`27:322`), falling back to 0 wherever that store-week was
never observed.

**The price coefficient is bilinear, not scalar.** `γ_i · β_j` gives each household its
own sensitivity and each item its own loading at `(N + J)·Kp` parameters instead of
`N·J`. It enters with a minus sign, so a **positive** `γ_i · β_j` means demand falls
when price rises.

### 3.2 The household state basis

`η_j · x_ijt` is the only term carrying history. `NestedData.state` (`27:174-188`)
finds the last day this household bought this item's **sub-commodity** strictly before
day `t`, and returns four features of `τ`, the days since:

$$
\tau_{ijt} \;=\; t \;-\; \max\bigl\{\,t' < t \;:\; i \text{ bought sub}(j) \text{ on day } t'\,\bigr\}
$$

$$
x_{ijt} \;=\;
\begin{bmatrix}
\mathbb{1}\{\text{no such } t'\} \\[2pt]
e^{-\tau_{ijt}/7} \\[2pt]
e^{-\tau_{ijt}/g_{\mathrm{sub}(j)}} \\[2pt]
\log(1+\tau_{ijt}) \,/\, \log 100
\end{bmatrix}
\qquad
\begin{aligned}
&\text{weekly decay} \\
&\text{that sub-commodity's own timescale} \\
&\text{slow, unbounded}
\end{aligned}
$$

with the last three set to 0 when there is no previous purchase. `g_sub` is that
sub-commodity's median repurchase gap, clipped to [3, 180] days
(`22_basket_data.py:257-259`), so "a long time" means the right number of days for milk
and for shampoo.

The lookup is a single `np.searchsorted` over a globally sorted key array,
`key = (i·S + sub) · 1024 + day`. The stride exceeds any day index, so the ordering
never crosses a (household, sub-commodity) boundary and a whole batch resolves in one
call.

### 3.3 The basket interaction, and the joint it implies

For item `j` in a basket `S` of `n` items, `ᾱ_j(S)` is the mean of `α` over the other
`n − 1` items (`27:302-308`):

$$
\bar\alpha_j(S) \;=\; \frac{1}{n-1}\Bigl(\sum_{k\in S}\alpha_k \;-\; \alpha_j\Bigr),
\qquad n = |S|,
\qquad \bar\alpha_j(S) := \mathbf{0} \ \text{ when } n = 1
$$

so the term entering `u_j` is

$$
\alpha_j^{\top}\bar\alpha_j(S)
\;=\; \frac{1}{n-1}\sum_{k\in S,\; k\neq j} \alpha_j^{\top}\alpha_k
$$

the average dot product between `j` and each other item in the basket. Using `α` itself
rather than a free `ρ` is what forces co-purchase structure into the embedding that the
sub-commodity test reads (`BASKET_MODEL.md` §2: a free `ρ` scored 0.058 purity, tied
scored 0.302).

**It is the whole basket, not a prefix.** The data carries `BASKET_ID` and `DAY` and no
within-receipt order, so a basket is a *set*. There is no "what was in the cart at the
time" to condition on; the model is not sequential and could not be.

**What that implies.** Write `b_j` for everything in `u_j` that does not involve other
basket items. For baskets of size `n`, define

$$
E(S) \;=\; \sum_{j\in S} b_j \;+\; \frac{1}{n-1}\!\!\sum_{\{j,k\}\subset S}\!\! \alpha_j^{\top}\alpha_k
$$

Swapping item `j` for item `m` holds `|S| = n`, so `1/(n−1)` is identical on both sides:

$$
\begin{aligned}
E(S') - E(S)
&= (b_m - b_j) + \frac{1}{n-1}\Bigl[\sum_{k\neq j}\alpha_m^{\top}\alpha_k - \sum_{k\neq j}\alpha_j^{\top}\alpha_k\Bigr] \\
&= (b_m - b_j) + \alpha_m^{\top}\bar\alpha_j(S) - \alpha_j^{\top}\bar\alpha_j(S) \\
&= u_m - u_j
\end{aligned}
$$

which is exactly the difference the candidate softmax scores at `27:492`. **The item
head's conditionals are therefore the single-slot conditionals of `P(S) ∝ exp(E(S))` at
fixed basket size `n`.** Training maximises `sum_j log P(j | S minus j)` rather than
`log P(S)`, but those conditionals are consistent with that joint, so Gibbs sweeps over
a draft basket converge to it.

The `1/(n−1)` is what makes this work rather than what breaks it: it is constant inside
a fixed-`n` conditional and cancels in `u_m − u_j`. Across different `n` the coupling
strength differs, so `E` is a family indexed by basket size rather than one energy over
all baskets — which does not obstruct generation, because `n` is drawn from the
incidence and breadth heads before any item is.

### 3.4 The nest

The paper gets its nest from a within-category softmax plus an outside good, which is
exactly what unit demand buys it. Drop unit demand — 68.2% of baskets violate it — and
that construction is gone; the nest is not. It survives as a **Poisson–multinomial factorisation**:

$$
\begin{aligned}
\text{units from category } c:\quad & Q_{ict} \sim \mathrm{Poisson}\!\bigl(\exp(a_{ic} + \kappa_c\, IV_{ict})\bigr) \\
\text{allocation across its items:}\quad & (q_{ijt})_{j\in c} \mid Q_{ict} \sim \mathrm{Multinomial}\bigl(Q_{ict},\; \mathrm{softmax}_j\, u_{ijt}\bigr) \\
\text{inclusive value:}\quad & IV_{ict} \;=\; \log \!\!\sum_{j\in c,\; \texttt{carried}[j,s]} \!\! e^{\,u_{ijt}}
\end{aligned}
$$

Poisson–multinomial factorises, so this is equivalent to independent per-item counts

$$
q_{ijt} \;\sim\; \mathrm{Poisson}\!\Bigl(\exp\bigl(a_{ic} + (\kappa_c - 1)\,IV_{ict} + u_{ijt}\bigr)\Bigr)
$$

and `κ_c` is a nesting coefficient with the paper's interpretation:

| κ | meaning |
|---|---|
| **= 1** | `IV` cancels in the factorised form. Category volume is whatever the items sum to — no expansion (IIA) |
| **= 0** | category volume is fixed; a price cut only moves share |
| **> 1** | the category expands more than proportionally |

$\kappa_c = \mathrm{softplus}(\kappa^{\mathrm{raw}}_c) = \log(1+e^{\kappa^{\mathrm{raw}}_c})$, with $\kappa^{\mathrm{raw}}$ initialised at 0.5413 so $\kappa = 1.0$ exactly — the
"IV cancels" point — so the data has to move it in either direction (`27:254`).

### 3.5 The four heads

Four likelihood terms sharing `u_ijt` (`27:487-521`).

**Item head** — allocation. Per purchased row, a softmax over the true item plus
`n_neg = 20` sampled negatives, unavailable candidates masked to `−1e9`:

$$
\mathcal{L}_{\text{item}}
\;=\; -\frac{1}{|R|}\sum_{r\in R}
\log \frac{e^{\,u_{j_r}}}{\displaystyle\sum_{m\in C_r} \mathbb{1}\{\texttt{avail}_m\}\, e^{\,u_m}}
$$

$R$ is the set of purchase rows in the batch, $C_r = \{j_r\}\cup\{20 \text{ negatives}\}$.

**Quantity head** — units on a purchased line, with its **own** price coefficient so
the two margins are not forced to share one elasticity:

$$
z_{ijt} \;=\; \mathrm{clip}_{[-6,\,4]}\!\Bigl(q^{0}_j - \bigl(\gamma^{q\top}_i\beta^{q}_j\bigr)\Delta\log p_{jst} + \eta^{q\top}_j x_{ijt}\Bigr),
\qquad
\text{units}_{ijt} - 1 \;\sim\; \mathrm{Poisson}\bigl(e^{z_{ijt}}\bigr)
$$

$$
\mathcal{L}_{\text{qty}} \;=\; \operatorname{mean}\Bigl(e^{z} - k\,z + \log\Gamma(k+1)\Bigr),
\qquad k = \max(\text{units}-1,\,0)
$$

**Incidence head** — was this category bought on this trip. Bernoulli on

$$
\operatorname{logit}_{ict}
\;=\; c^{0}_c \;+\; c^{u\top}_i c^{c}_c \;+\; c^{s\top}_c x_{ict}
\;+\; \kappa_c\bigl(IV_{ict} - \overline{IV}_c\bigr)
$$

`ivref_c` is a frozen per-category constant (§5.3). `x_ict` is the state basis
evaluated at the category's first block item.

**Breadth head** — given the category was bought, how many *distinct* items:

$$
z^{b}_{ic} \;=\; \mathrm{clip}_{[-6,\,3]}\!\Bigl(b^{0}_c + b^{u}_i - b^{p}_c\,\overline{\Delta\log p}_{ct}\Bigr),
\qquad
\bigl(\text{distinct items in } c\bigr) - 1 \;\sim\; \mathrm{Poisson}\bigl(e^{z^{b}}\bigr)
$$

fitted on bought categories only. `Δlog p̄_ct` is the mean price deviation across that
category's stocked items (`27:427`), so a promotion widens the basket as well as
deepening it.

**Why breadth needs its own head.** The incidence head only ever sees 0/1, so nothing
else in the model knows how many distinct items a category purchase contains.
Recovering a count from $P(\text{buy})$ assumes $Q\sim\mathrm{Poisson}(\lambda)$ with
$P(Q>0) = 1-e^{-\lambda} = p$, so $\lambda = -\log(1-p)$ and

$$
\mathbb{E}[\,Q \mid Q>0\,] \;=\; \frac{\lambda}{1-e^{-\lambda}} \;=\; \frac{\lambda}{p}
$$

which reproduces the probability but implies

| P(buy) | implied E[items given bought] |
|---|---|
| 0.02 | 1.010 |
| 0.05 | 1.026 |
| 0.10 | 1.054 |
| 0.20 | 1.116 |

against a real **1.284**. It can only ever produce about 1.0–1.1. This was the one gap
among the fixes in §10 that was a *missing component* rather than a sampling bug.

### 3.6 Objective

One scalar per step (`27:612-618`):

$$
\mathcal{L}
\;=\; \mathcal{L}_{\text{item}}
\;+\; w_q\mathcal{L}_{\text{qty}}
\;+\; w_c\mathcal{L}_{\text{inc}}
\;+\; w_b\mathcal{L}_{\text{brd}}
\;+\; \frac{1}{B}\Bigl(\ell_2\lVert\Theta_{\text{repr}}\rVert^2 + \ell_2^{p}\lVert\Theta_{\text{price}}\rVert^2\Bigr)
$$

with `w_q = w_c = w_b = 1`, `l2 = 0.01`, `l2price = 0.0001`, and `B` the number of item
rows in the batch.

**The two L2 groups are separate on purpose.** `l2_repr` covers `α, θ, η, μ, δ, ζ, ξ`
and the incidence embeddings; `l2_price` covers only `γ, β, γq, βq`. Shrinking the
price block at the representation block's rate biases the elasticity toward zero, and
the elasticity is the quantity being measured.

Adam, `lr = 0.005` cosine-annealed to `0.05·lr` over 12,000 iterations. A batch is
192 baskets, expanded to all of their purchase rows.

---

## 4. Where stores enter

Three channels, which must be kept apart because they are not equally meaningful:

1. **Store-level price deviation**, added to the chain deviation where that store-week
   was observed — 244,880 cells, **0.53%** of the (item × store × week) grid. Everything
   else falls back to the chain price. This is information.
2. **Store × item affinity** `ζ_j · ξ_s`, low rank — format and assortment differ.
   This is information.
3. **The availability mask.** Items a store never sold are masked out of the softmax
   denominator. This makes the ranking task **mechanically easier**, because there are
   fewer real competitors.

`--no-store`, `--no-store-price` and `--avail-only` exist to separate (3) from (1) and
(2). **Any claim about what stores are worth must report that split** (§8.2), or it is
claiming credit for a smaller choice set.

---

## 5. Fitting

### 5.1 Negative sampling

Negatives are drawn from a unigram^0.75 distribution over **training** purchase counts
(`27:151-153`):

$$
p_{\text{neg}}(j) \;\propto\; \max\bigl(\mathrm{count}_{\text{train}}(j),\,1\bigr)^{0.75}
$$

`n_neg = 20` per positive, so each softmax has 21 candidates.

**Negatives are drawn from the whole catalogue, then masked** (`27:310`, `27:325-328`).
They are *not* drawn only from what the store stocks, which earlier versions of this
document and the module docstring both claimed. The sequence is: sample 20 items from
all `J`, look up `carried[j, s]`, force the true item's slot to available (it was
bought there, so it was stocked), and mask the rest to `−1e9`. A consequence worth
knowing: the **effective** number of competitors is below 20 and varies by store, since
an unstocked draw contributes nothing to the denominator.

### 5.2 Category sampling for the incidence head is UNIFORM

`incidence_batch` (`27:337-433`) draws `n_cat = 16` categories per trip **uniformly**
from all 188, independent of what the trip bought:

$$
\text{for each trip: } c_1,\dots,c_{n_{\text{cat}}} \overset{\text{iid}}{\sim} \mathrm{Unif}\{0,\dots,C-1\},
\qquad
y = \mathbb{1}\{\text{the trip bought } c\}
$$

The base rate is low — 6.075 categories per training basket out of 188, so **3.23%** — and
uniform sampling therefore sees about 0.52 positives per trip. That is the price paid,
and it is paid deliberately.

**Case-control sampling was tried and abandoned.** Taking a few bought and a few not
over-samples positives about 30×, and it biases the fit in two separate ways:

1. **the intercept** — which the standard offset `log(π₁/π₀)` does correct;
2. **the term `κ_c · (IV − ivref)`** — whose *mean differs between the sample and the
   population*, because bought categories have higher inclusive values. The offset does
   not touch this. `c0` absorbs the training-sample average, and at generation time a
   uniform draw sits about 0.9 lower in `(IV − ivref)`; with `κ ≈ 0.8` that is a 0.73
   error in the logit, which emptied the generated baskets — 4.0 categories against a
   real 6.5, median basket 0 items.

Uniform sampling removes both at source and needs **no correction at all**. `off` is
still built in `incidence_batch` and returned in the batch dict, but it is always 0.0
and `losses` never reads it — dead code left from the case-control version.

### 5.3 The inclusive value is sampled, and the reference is frozen

**Sampled.** Categories are padded to the largest (225 items) although the median has
15, so a dense `IV` block spends most of its work on padding. Instead `iv_cap = 32`
items are drawn per category (with replacement, `27:386-390`) and the sum scaled:

$$
\widehat{IV}_{ict} \;=\; \log\Bigl(\sum_{s=1}^{m} e^{\,u_{i j_s t}}\Bigr) \;+\; \log\frac{n_c}{m},
\qquad j_s \overset{\text{iid}}{\sim} \mathrm{Unif}\{\text{stocked items of } c\}
$$

The **sum** inside is unbiased. Each draw is uniform over the $n_c$ stocked items, so

$$
\mathbb{E}\Bigl[\frac{n_c}{m}\sum_{s=1}^{m} e^{\,u_{j_s}}\Bigr]
\;=\; \frac{n_c}{m}\cdot m\cdot \frac{1}{n_c}\sum_{j\in c} e^{\,u_j}
\;=\; \sum_{j\in c} e^{\,u_j}
$$

The **log** of it is not: by Jensen $\mathbb{E}[\log X] \le \log\mathbb{E}[X]$, so
$\widehat{IV}$ is biased slightly low. The bias is
absorbed into `c0_c` for training, which is why it does no harm there, and generation
uses the same estimator so the two stay consistent.

**Frozen reference.** `ivref_c` is fixed once, at iteration 500, to the mean `IV` over a
**uniform** category sample (`uniform_iv`, `27:436-484`, 24 categories per trip × 8
batches). Any fixed constant is absorbed by `c0`, so freezing does not change the fit —
it guarantees training and generation subtract the *same* thing. Taking the reference
from `incidence_batch` instead used the case-control sample, which over-weights bought
categories; the reference then sat ≈0.9 too high and most generated baskets came out
empty.

Categories with nothing stocked at that store get `IV = 0` rather than `−1e9`, so the
category intercept explains them instead of the inclusive value dominating the logit
(`27:424`).

### 5.4 Checkpointing

Validation is run every 6,000 iterations on up to 3,000 baskets. The checkpoint is kept
on

$$
\text{score} \;=\; \ell^{\text{val}}_{\text{item}}
\;-\; w_q\,\mathrm{NLL}^{\text{val}}_{\text{qty}}
\;-\; w_c\,\mathrm{NLL}^{\text{val}}_{\text{inc}}
$$

(`27:633`). **Breadth is not in the score** — it has no held-out scalar in `evaluate`,
so it is trained but never used for model selection. The final test pass reloads the
best checkpoint and scores 6,000 baskets with a fixed seed.

Note `--iv-center` is accepted, threaded through `evaluate` and `losses`, and **never
read** — `losses` centres on `model.iv_ref` alone. Dead argument.

### 5.5 What validation actually scores

This is the part most easily misread, so it is written out in full.

**Validation does not simulate a shopper filling a cart.** `evaluate` (`27:524-550`)
takes **real held-out baskets** and scores each purchased row in place. For a basket
$S$ with $n$ items, every one of the $n$ rows is scored independently:

$$
\ell_r \;=\; \log \frac{e^{\,u_{j_r}}}{\displaystyle\sum_{m \in C_r} \mathbb{1}\{\texttt{avail}_m\}\,e^{\,u_m}},
\qquad C_r = \{j_r\} \cup \{20\ \text{negatives}\}
$$

and the reported number is the mean of $\ell_r$ over rows. The context used for row $r$
is the leave-one-out mean over **the other items of that same real basket**:

$$
\bar\alpha_{j_r}(S) \;=\; \frac{1}{n-1}\Bigl(\sum_{k\in S}\alpha_k - \alpha_{j_r}\Bigr)
$$

So the question being answered is **"which item fills this slot, given the rest of the
basket"** — a fill-in-the-blank score. It is not "predict the basket", and it is not
"predict the next item". Nothing is ever asked sequentially.

**A one-item basket.** When $n = 1$ there is no other item, and `27:308` sets the
context to the zero vector rather than dividing by zero:

$$
n = 1 \;\Longrightarrow\; \bar\alpha_j(S) = \mathbf{0}
\;\Longrightarrow\;
u_{ijt} = \lambda_j + \theta_i^{\top}\alpha_j - (\gamma_i^{\top}\beta_j)\Delta\log p_{jst} + \eta_j^{\top}x_{ijt} + \mu_j^{\top}\delta_w + \zeta_j^{\top}\xi_s
$$

The interaction term contributes exactly nothing, and the choice rests on popularity,
household taste, price, recency, season and store affinity. This is not a rare corner:

| split | baskets | single-item baskets | share of scored rows they carry |
|---|---|---|---|
| train | 157,464 | 19.6% | 2.5% |
| validation | 17,113 | 18.0% | 2.3% |
| test | 24,768 | 17.8% | 2.2% |

Nearly a fifth of baskets are scored with the interaction term switched off, but they
are small baskets, so they account for only about 2% of the rows the log-likelihood
averages over.

**"I have bought $n$ items, what is item $n+1$?"** The model can answer this, but
**nothing in the repository asks it.** Written out, adding an item to an existing
basket $S_n = \{j_1,\dots,j_n\}$ makes the basket size $n+1$, so for a candidate $m$ the
leave-one-out formula gives

$$
\bar\alpha_m(S_n \cup \{m\})
\;=\; \frac{1}{(n+1)-1}\Bigl(\sum_{k\in S_n}\alpha_k + \alpha_m - \alpha_m\Bigr)
\;=\; \frac{1}{n}\sum_{k\in S_n}\alpha_k
$$

— the plain mean of what is already in the cart, with no dependence on the candidate.
Ranking candidates by

$$
u_{imt} \;=\; b_m \;+\; \alpha_m^{\top}\Bigl(\tfrac{1}{n}\textstyle\sum_{k\in S_n}\alpha_k\Bigr)
$$

is therefore well defined: items whose embedding points the same way as the cart's
average get a boost, which is exactly the co-purchase structure the tied interaction
was built to carry.

**Two caveats on doing that.**

*It ranks, it does not decide whether to stop.* §3.3 shows $E(S)$ is a family indexed by
basket size: going from $n$ to $n+1$ changes the pairwise coupling from $\tfrac{1}{n-1}$
to $\tfrac{1}{n}$ for **every** pair, so the item head gives no comparable probability
for "add this item" against "add nothing". How many items a basket holds comes from the
incidence and breadth heads, never from the item head.

*The generator does not do it.* `28:198` passes a zero context, so generated baskets
carry no co-purchase structure (§8.1b, §9). Adding the running mean above is the
one-line version of the fix; Gibbs sweeps over a completed draft are the principled
version, since the model conditions on the full set rather than a prefix.

**One implementation detail with a real consequence.** The context is built from
`model.alpha.detach()` (`27:303`). Gradient flows into $\alpha_m$ for the candidate
being scored, but **not** into the $\alpha_k$ of the items forming the context — a
stop-gradient on the context side. Each item's embedding is therefore pulled by the
baskets it appears in *as the scored item*, and never by appearing as a neighbour.

---

## 6. Engineering

### Two vectorised lookups

Household state and store price are both sparse, high-cardinality lookups that are
ruinous done naively: materialising (household × day × sub-commodity) is ≈32 million
rows, and a per-sample Python dict is ≈35 million hits per epoch.

Both use one trick — values stored once as a globally sorted key array,
`key = group_id · stride + index`, so a whole batch resolves in a single
`np.searchsorted`. The stride exceeds any index, so ordering never crosses a group
boundary.

### Measured cost

From `out/*_nested_history.json`, the `secs` field at the last evaluation (wall clock,
including validation passes), CPU:

| run | iterations | wall time | ms/iter |
|---|---|---|---|
| `nested` | 12,000 | 10.8 min | 53.8 |
| `nested_nostore` | 12,000 | 7.9 min | 39.5 |
| `nested_nonest` | 12,000 | 3.7 min | 18.3 |

**Known bottleneck:** `np.searchsorted` is roughly 40% of a step (≈60,000 queries per
iteration). State features are *data, not parameters* — they depend only on
(household, sub-commodity, day) and never change during training — yet they are
recomputed every iteration for every sampled candidate. Deduplicating repeated
candidates within a batch would recover about half of that. Not done.

### Scalability

| dimension | cost |
|---|---|
| catalogue size | **independent** — negative sampling fixes candidates at 21 |
| households | **independent** — only the embedding table grows |
| baskets | **independent per step** |
| categories | capped at `n_cat = 16` sampled per trip |
| items per category | capped at `iv_cap = 32` by the sampled IV |

Nothing is quadratic in items or households.

---

## 7. Acceptance criteria

| # | requirement | test | status |
|---|---|---|---|
| 1 | **multiple items** per category and across categories | likelihood admits multisets; no unit-demand filter | ✅ all 188 categories retained, none dropped for unit demand |
| 2 | **multiple quantities**, with price interaction | quantity head with its own coefficient | ✅ `γq·βq` = +0.134 against the item head's +0.794; 12% of total elasticity (§8.4) |
| 2b | **multiple items per category** | breadth head | ✅ generated 1.324 items per purchased category against a real 1.287 (§8.6) |
| 2c | **product interaction** | tied `α_j · ᾱ(basket)` | ✅ 0.115 nats, and purity 0.2976 → 0.2119 without it (§8.1b) — but inert outside the item head |
| 3 | **household state as a level** | recency basis per (household, sub-commodity) | ✅ 0.076 nats against a 0.0032 seed spread (§8.7) |
| 4 | **nested theory retained** | `κ` estimated per category | ⚠️ `κ` = 0.663, stable across seeds, but weakly identified (§8.4) |
| 5 | **store information used** | prices, affinity, availability | ⚠️ used, but 99.7% of the gain is the availability mask, not information (§8.2) |
| 6 | **embeddings recover sub-commodity structure** | `24_embedding_eval.py`, against random / popularity / nf | ⚠️ 69.6× chance, AUC 0.8222 — beats every control decisively, but marginally **below** the flat model's 70.6× and 0.8233 (§8.5) |
| 7 | **data generation** | roll incidence → breadth → items → units forward; item ids via `generate_baskets` | ✅ items -1.0%, units -1.4%, categories -3.8% against held-out (§8.6) |
| 8 | **what-if on price** | structural placebo + elasticity decomposition | ✅ placebo retains 0.0% of the coefficient; decomposition sums exactly (§8.3) |
| 9 | **beats a simpler alternative** | one scorer, identical candidate sets, tuned baselines | ✅ +0.342 nats and 8.5 points of top-1 over household repeat-purchase (§8.1c) |

Criteria 4, 5 and 6 are marked ⚠️ rather than ✅. Each is *met* in the sense that the
component exists and is used, but each carries a qualification that the ✅ would hide,
and those qualifications are the honest content of §8.

---

## 8. Results

All fits use the recipe in §5. Sources: `out/*_nested_history.json` (fit and
ablations), `out/nested_counterfactual.json` (placebo, decomposition, generation),
`out/benchmark.json` (§8.1c), `out/embedding_eval.json` (§8.5).

**Seed spread is 0.0032 nats** on item log-likelihood (`nested` -1.9927 against
`nested_s1` -1.9895), so any gap below larger than ≈0.01 is real.

### 8.1 Fit and ablations

Test-set item log-likelihood, top-1, and the two auxiliary NLLs:

| model | item log-lik | top-1 | quantity NLL | incidence NLL | cost of removing |
|---|---|---|---|---|---|
| **`nested`** | -1.9927 | 0.378 | 0.6666 | 0.1104 | — |
| seed 1 | -1.9895 | 0.379 | 0.6693 | 0.1103 | — |
| no store | -2.1820 | 0.355 | 0.6651 | 0.1102 | **0.189** |
| no interaction | -2.1074 | 0.347 | 0.6666 | 0.1101 | **0.115** |
| no state | -2.0684 | 0.360 | 0.6665 | 0.1106 | **0.076** |
| prices scrambled | -2.0416 | 0.362 | 0.6887 | 0.1103 | **0.049** |
| availability only | -1.9932 | 0.375 | 0.6651 | 0.1104 | **0.001** |
| no breadth | -1.9927 | 0.378 | 0.6666 | 0.1104 | **0.000** |
| no quantity | -1.9925 | 0.378 | — | 0.1104 | **-0.000** |
| no nest | -1.9877 | 0.380 | 0.6677 | — | **-0.005** |

**Three of the four heads cost nothing on item ranking, and the nest is marginally
better without it.** That is the design working, not failing. The item head ranks
items; the nest, quantity and breadth heads answer questions the item head does not
ask, and they share only `u_ijt`. A head that improved item ranking *because* it also
modelled category incidence would mean the two were entangled — which is exactly what
`BASKET_MODEL.md` §2 had to fix in the flat model, where a free `ρ` absorbed structure
that belonged in `α`.

So each head must be scored on its own quantity:

| head | scored on | result |
|---|---|---|
| item | item log-lik, top-1 | -1.9927, 0.378 |
| interaction (tied `α·ᾱ`) | item log-lik, and embedding purity | 0.115 nats; purity 0.2976 → 0.2119 |
| incidence | incidence NLL, and `κ` | 0.1104; `κ` = 0.663 |
| quantity | units per item in generation | 1.343 against a real 1.348 (§8.6) |
| breadth | distinct items per purchased category | 1.324 against a real 1.287 (§8.6) |

The breadth head is the case in point. On item log-likelihood it is worth **0.000** —
the two runs agree to four decimals — yet without it generated baskets hold 6.94 items
against a real 8.36, and with it 8.27. Judging it by the ablation column alone would
have deleted the fix for the only criterion that was failing.

### 8.1b The interaction term, and where it is inert

| | with interaction | without |
|---|---|---|
| item log-lik | **-1.9927** | -2.1074 |
| embedding purity (5,455 items) | **0.2976** (69.6× chance) | 0.2119 (49.5×) |
| embedding AUC | **0.8222** | 0.7267 |

**0.115 nats**, the second-largest component after the store availability mask, and the
only thing besides that mask which materially moves the embedding.

**But it is inert in three of the four places the model is used:**

| where | context | why |
|---|---|---|
| item head | **active** | the basket is known; this is what 0.115 measures |
| incidence head (`27:413`) | zeroed | incidence is decided *before* the basket exists — conditioning on it would be circular |
| elasticity decomposition (`28:111`) | zeroed | evaluated at the point of category choice, so pre-basket |
| generator (`28:198`) | zeroed | one forward pass, category by category; no draft basket to condition on |

The first two are correct by construction. **The third is a real limitation**: the
reported allocation channel excludes any interaction effect, so a price cut that
changes *what else lands in the basket* is not counted in the -1.188.

**The fourth is a gap in the sampler, and it is fixable.** §3.3 shows the item head's
conditionals are those of `P(S) ∝ exp(E(S))` at fixed basket size, so there *is* a joint
to sample from, and `n` is drawn from the incidence and breadth heads before any item
is. Gibbs sweeps over a draft basket would converge to it. The current generator never
revisits a slot, so generated baskets carry no co-purchase structure at all: the term
worth 0.115 nats contributes nothing to the data the generator emits.

### 8.1c Benchmarks against simpler alternatives

![benchmark](figures/benchmark.png)

Everything in §8.1 is an **ablation** — the model against itself with a piece removed.
That shows each piece is used; it does not show the model beats what a practitioner
would reach for. Every model here is scored through **one scorer on identical candidate
sets**: same positives, same negatives, same availability mask, 51,167 held-out
purchases with 20 negatives each.

That protocol detail matters. The nested model masks unstocked items out of its choice
set, which shrinks the denominator and mechanically raises its log-likelihood (§8.2
measures this at 99.7% of the apparent store gain). Comparing its own reported number
against a differently-evaluated baseline would credit it for an easier question.

| model | log-lik | top-1 | vs popularity |
|---|---|---|---|
| **nested** | -2.0011 | 0.371 | +0.736 |
| nested, no interaction | -2.1195 | 0.342 | +0.618 |
| household repeat-purchase | -2.3428 | 0.285 | +0.395 |
| popularity | -2.7375 | 0.091 | +0.000 |
| random | -2.7378 | 0.067 | -0.000 |
| household + co-occurrence | -3.0526 | 0.182 | -0.315 |
| item–item co-occurrence | -3.0655 | 0.078 | -0.328 |

Baseline weights and temperatures are **tuned on validation**, because the fitted model
had its hyperparameters selected and an untuned baseline is not a fair reference.

**The honest reading.** The strongest baseline is not popularity, it is **household
repeat-purchase** — "this household bought it before" — at top-1 0.285 against the
model's 0.371. Grocery is repetitive, and most of what any model can predict is that
people rebuy what they always rebuy. The model's real margin over a serious baseline is
**+0.342 nats and 8.5 points of top-1**, not the +0.736 against popularity that a
friendlier framing would quote.

Two results that cut against the model:

- **Popularity ≈ random** (-2.7375 vs -2.7378). Negatives are drawn unigram^0.75, i.e.
  popularity-weighted, so popularity is being asked to separate a popular true item
  from popular decoys. "We beat popularity by +0.736 nats" is therefore a much weaker
  claim than it sounds.
- **Item–item co-occurrence scores below random** (-3.0655). Counting co-occurrence is
  a model-free stand-in for the interaction term, and on its own it is worse than
  nothing here — the signal exists (`DATA_EXPLORATION.md` §2) but raw counts cannot
  exploit it against popularity-matched decoys. The learned tied embedding can:
  removing it costs 0.118 nats on this same scorer.

**What this section does not measure** (see §5.5). The item log-likelihood is a
*conditional* number: `P(item j | the other items in the same basket, household, week, store,
prices)`. Test baskets are from held-out weeks, so nothing leaks across time, but every
model here — the nested model and the co-occurrence baselines alike — is told the rest
of the basket. It is a fill-in-the-blank score, not "predict the basket". The
from-scratch test is §8.6, where the context is zeroed everywhere.

### 8.2 Criterion 5 — stores, split honestly

| | item log-lik | gain |
|---|---|---|
| no store at all | -2.1820 | — |
| **+ availability mask only** | -1.9932 | **+0.1888** |
| + store prices and affinity | -1.9927 | **+0.0005** |

**99.7% of the store gain is the mechanically smaller choice set**, not information.

Modelling stores was still correct — treating an item a store never stocked as a
*rejected* alternative is a specification error — but it is a correctness fix, not an
information gain, and it makes item log-likelihood **incomparable to the flat model**
in `BASKET_MODEL.md`.

This sits awkwardly against `DATA_EXPLORATION.md` §7.3, which finds households use a
median of 4 stores with 30% of consecutive trips switching. Store-level prices ought to
matter. That they contribute +0.0005 nats suggests either that 0.53% grid coverage is
too sparse to help, or that the chain price is already a good proxy. Open question,
recorded in §9 rather than resolved.

### 8.3 Criterion 8 — is the price response causal?

Coefficients are the median of `γ @ βᵀ` and `γq @ βqᵀ` over all (household, item) pairs,
read directly from the checkpoints:

| model | price coefficient | quantity price coefficient | κ |
|---|---|---|---|
| `nested` | **+0.7945** | +0.1339 | 0.6626 |
| `nested_s1` (seed 1) | +0.7942 | +0.1085 | 0.6740 |
| **`nested_pl`** (prices scrambled) | **−0.0000** | **−0.0000** | 0.6752 |

The **structural placebo** refits the whole model on a price panel scrambled before
fitting (`--placebo-price permute`, `27:96-103`), including the store-level deviations
(`27:118-120`) — leaving those real would leak genuine prices back in. It retains
0.0% of the price coefficient and costs 0.049 nats of item ranking, while leaving `κ`
and the incidence NLL untouched. Prices scramble; category structure does not.

For scale, `29_demand_eda.py` measures a model-free within-item elasticity of **−0.945
on units**; the model's allocation channel alone is -0.991.

### 8.4 Criteria 2 and 4 — where a price cut actually goes

A 1% cut in item `j`'s price moves demand through three channels. From the softmax and
the Poisson–multinomial (`28:125-135`):

$$
\begin{aligned}
\text{allocation}\quad & \frac{\partial \log \pi_j}{\partial \log p_j} = -\bigl(\gamma_i^{\top}\beta_j\bigr)\bigl(1-\pi_j\bigr) \\[4pt]
\text{incidence}\quad  & \kappa_c\frac{\partial\, IV}{\partial \log p_j} = -\kappa_c\bigl(\gamma_i^{\top}\beta_j\bigr)\pi_j \\[4pt]
\text{quantity}\quad   & \frac{\partial \log \mathbb{E}[\text{units}]}{\partial \log p_j} = -\bigl(\gamma^{q\top}_i\beta^{q}_j\bigr)\frac{\lambda}{1+\lambda},\qquad \lambda = e^{z}
\end{aligned}
$$

The first two are exact derivatives of the same softmax, which is why they carry
$(1-\pi_j)$ and $\pi_j$; the third follows from $\mathbb{E}[\text{units}] = 1+\lambda$.
Allocation and incidence therefore combine to

$$
-\bigl(\gamma_i^{\top}\beta_j\bigr)\bigl(1-\pi_j\bigr) - \kappa_c\bigl(\gamma_i^{\top}\beta_j\bigr)\pi_j
\;=\; -\bigl(\gamma_i^{\top}\beta_j\bigr)\bigl[\,1 - \pi_j(1-\kappa_c)\,\bigr]
$$

Evaluated on 12,724 held-out purchase rows (`28:72-154`):

```
total own-price elasticity (mean)   -1.188
  allocation  (share within category)  -0.991   (83%)
  incidence   (the category expands)   -0.058   (5%)
  quantity    (units per buyer)        -0.139   (12%)
```

The median total is -0.943; **the shares are computed from means because medians do not
decompose additively** — reporting median components against a median total does not
sum to 100% (`28:139-141`).

**Mostly it moves share.** 83% of the response is reallocation inside the category;
only 5% is the category expanding. The quantity margin is **12%** — a share that a
binary-purchase model cannot reach at all, independently corroborated by the model-free
units-per-buyer elasticity of −0.235 in `DATA_EXPLORATION.md` §6.1.

**κ = 0.663, with 1.1% of categories above 1.** Below 1 means a price cut grows the
category *less* than proportionally to what its items gain.

Read κ with care. Across three incidence samplers it moved 1.411 → 0.790 → 0.663, and
only the last is unbiased (§5.2). It moved with the *sampler*, not the data. It is
stable across seeds (0.663 vs 0.674), but that history means **κ is weakly identified**
and no argument here rests on its exact value.

### 8.5 Criterion 6 — do the embeddings recover sub-commodity structure?

Sub-commodity labels are held out of the model entirely. On the full 5,455-item catalogue
(`out/embedding_eval.json`):

| | kNN purity | × chance | AUC | silhouette |
|---|---|---|---|---|
| **`nested` α** | 0.2976 | 69.6× | 0.8222 | -0.1610 |
| no interaction | 0.2119 | 49.5× | 0.7267 | -0.2250 |
| control: popularity | 0.0038 | 0.9× | 0.5100 | -0.4468 |
| control: random | 0.0037 | 0.9× | 0.4986 | -0.3143 |

Head-to-head on the 409 items the paper's model also covers and whose sub-commodity has
2+ members (`out/log_embed_nested.txt`):

| | kNN purity | × chance | AUC | silhouette |
|---|---|---|---|---|
| **`nested` α** | **0.1648** | **12.1×** | **0.8045** | **−0.0535** |
| nf β (the paper) | 0.0142 | 1.0× | 0.3789 | −0.3127 |

nf's AUC of **0.3789 — below 0.5** — is a mechanism, not noise: its within-category
softmax makes items compete, so the gradient pushes apart exactly the items that are
close substitutes.

**Where this criterion falls short.** The target was to match the flat model's 70.6×
chance and AUC 0.8233. `nested` reaches 69.6× and 0.8222 — marginally *below* both. The
gap is small and within the range the seed replicate spans (`nested_s1` is 68.9× and
0.8240), but the criterion as written is not cleanly met and earlier versions of this
document reported 70.1× and 0.828, which do not appear in any artefact.

The same trade-off as the flat model reappears: **`nested_nostate` has the best
embedding of any variant** (0.3458, 80.8× chance, AUC 0.8696) while costing 0.076
nats of item log-likelihood. `η_j` absorbs repeat-purchase regularity that `α` would
otherwise carry. The headline model keeps state because a state level was required and
it is the transition function any dynamic policy needs — not because it is free.

### 8.6 Criterion 7 — generation

`28:158-267`. For each held-out trip, roll the layers forward with the household, day,
week and store fixed to the real ones: draw incidence per category from the Bernoulli
head, then distinct items from the breadth head, then expected units from the quantity
head. Categories are subsampled (24 of 188) and the counts scaled by `C / n_cat_eval`.

| | real | generated | error |
|---|---|---|---|
| items per basket | 8.36 | **8.27** | **-1.0%** |
| units per basket | 11.27 | **11.11** | **-1.4%** |
| categories per basket | 6.49 | **6.25** | **-3.8%** |

The internal ratios hold too: **1.324** items per purchased category against a real
1.287, and **1.343** units per item against 1.348.

Four of the five generator versions were wrong, each for a different reason:

| generator | categories | items | what was wrong |
|---|---|---|---|
| 1. case-control, batch-centred IV | 58.0 | 58.0 | positives over-sampled 30×; a `log(30)` logit error |
| 2. case-control, frozen IV | 10.4 | 10.4 | reference taken from the case-control sample itself |
| 3. uniform, frozen IV | 4.0 | 4.0 | `κ·(IV − ref)` still uncorrected for the sample |
| 4. + units from the quantity head | 6.39 | 6.94 | the generator never called the quantity head |
| **5. + breadth head** | **6.25** | **8.27** | — |
| **real** | **6.49** | **8.36** | |

Only the last was a model gap; the rest were sampling.

### 8.6b Generation with actual item ids, and the context problem

**`generate` never samples an item.** It accumulates expected counts — `n_items`,
`n_c`, `n_u` at `28:253-255` — and uses the within-category choice probabilities only to
weight expected units. So §8.6 measures basket *shape*, and the question "where does the
basket context come from at generation time" never arose there, because no basket
exists. Nothing downstream that needs item ids, an MDP included, can consume that output.

`generate_baskets` (`28:158`) emits actual items, in two passes.

**Pass 1 — build a draft.** For every one of the 188 categories: incidence from the
Bernoulli head, breadth from the breadth head, then that many **distinct** items drawn
without replacement from `softmax(u)` over the category's stocked items. The context is
zero here, and correctly so — no basket exists yet.

**Pass 2 — Gibbs sweeps.** §3.3 shows the item head's conditionals are those of
$P(S)\propto e^{E(S)}$ at fixed basket size, so resampling one slot at a time from its
own category, with

$$
\bar\alpha_{j}(S) = \frac{1}{n-1}\Bigl(\sum_{k\in S}\alpha_k - \alpha_{j}\Bigr)
$$

recomputed from the current draft, targets exactly that joint. Basket size and category
composition are held fixed, which is the move $E(S)$ is defined over. The running sum
$\sum_k \alpha_k$ is updated incrementally, and only the resampled category's utilities
are recomputed — touching all 188 per slot is 188× the work for no extra information.

**Does it work?** Two measures, both averaged over the item pairs inside a basket:
the mean $\alpha_j^{\top}\alpha_k$, which is the quantity the interaction term acts on;
and the share of pairs sharing a `SUB_COMMODITY`, a label the model never sees. Five
seeds of 300 trips each:

| | items | mean $\alpha_j^{\top}\alpha_k$ | same-sub pair share |
|---|---|---|---|
| **real held-out** | 8.37 | **+0.5580** ± 0.0443 | **0.0656** ± 0.0121 |
| generated, context zeroed | 8.17 | +0.1334 ± 0.0087 | 0.0270 ± 0.0045 |
| generated, 4 Gibbs sweeps | 8.25 | **+0.2820** ± 0.0294 | **0.0358** ± 0.0036 |

Gibbs **2.1×** the within-basket embedding alignment, from 0.1334 to 0.2820 —
far outside the ±0.0294 seed spread — and lifts the same-sub pair share from
0.0270 to 0.0358. Mean basket size is unchanged (8.17 → 8.25 against a real
8.37), so the structure is added without distorting the marginals §8.6 reports. The
chain settles after roughly one sweep; more sweeps do not climb further.

**It closes about half the gap, not all of it.** Generated baskets reach
**51%** of the real $\alpha^{\top}\alpha$ level and **55%** of the real same-sub share.
Real baskets are more internally coherent than anything this model generates. The
interaction term is worth 0.115 nats when the basket is *given* (§8.1b); asked to
produce that structure from nothing, it recovers half of it.

**What generation carries, stated plainly.** With the context zeroed it reproduces
marginals only. With Gibbs sweeps it also carries roughly half the real co-purchase
structure. Neither is the full joint: §9 records that there is no held-out joint
likelihood, because the partition function over size-$n$ baskets is intractable.

### 8.7 Criterion 3 — household state

Removing the state basis costs **0.076 nats**, the largest genuine gain of any
component after the store availability mask, and 23× the seed spread.

Independently corroborated model-free: `DATA_EXPLORATION.md` §7.1 finds taste 1.66×
more self-similar within household than across (0.7838 against 0.4710 over all
4,266,290 cross-household pairs), and §7.2 finds a split-half correlation of +0.236 in
per-household price sensitivity — modest, and §7.6 shows why: most households never see
the price move on what they repeatedly buy.

---

## 9. Known issues

| issue | severity | state |
|---|---|---|
| **Criterion 6 is marginally missed.** Embedding purity 69.6× and AUC 0.8222 against the flat model's 70.6× and 0.8233 | low — beats every control by 70×, and within seed range | open |
| **The elasticity decomposition excludes the interaction.** Context is zeroed when it is computed, so a price cut that changes what *else* enters the basket is not counted in the -1.188 | medium — the number is a lower bound on the true own-price response | open |
| **`generate` emits counts, not baskets.** It accumulates expected `n_items`/`n_c`/`n_u` (`28:253-255`) and never samples an item, so its output cannot feed anything needing item ids | medium — §8.6 measures shape only | **fixed**: `generate_baskets` emits items |
| **Generated baskets recover about half the real co-purchase structure.** Gibbs sweeps lift within-basket $\alpha^\top\alpha$ from 0.1334 to 0.2820 against a real 0.5580 (51%) | medium — marginals are right, internal coherence is half | open |
| **There is no held-out joint likelihood.** (§5.5) `P(S)` needs the partition function over all size-`n` baskets, which is intractable to evaluate exactly. The reported item log-likelihood is conditional on the rest of the basket and is not a substitute | medium — the from-scratch evidence is §8.6 only | open |
| Store-level prices and affinity contribute +0.0005 nats, although `DATA_EXPLORATION.md` §7.3 shows households use 4 stores and switch on 30% of trips | medium | open |
| `κ` moved 1.411 → 0.790 → 0.663 across three incidence samplers — with the *sampler*, not the data. Stable across seeds now, but weakly identified | medium | documented, not resolved |
| Breadth is trained but is **not in the checkpoint-selection score** (`27:633`), which covers item, quantity and incidence only | low | open |
| `--iv-center` is accepted and threaded through `evaluate`/`losses` but never read; `offset` is built in `incidence_batch` and never consumed | low — dead code, no effect on results | open |
| `gstart_keys` / `gstart_vals` are written by `22_basket_data.py` and never loaded by `NestedData` | low — dead artefact | open |
| `28:310` logs `dec['total']` as "median own-price elasticity"; it is the **mean** | low — mislabelled log line only, the JSON field is correct | open |
| `np.searchsorted` is ≈40% of a step and is avoidable by deduplicating candidates | low — performance only | open |
| Store-level prices cover 0.53% of the item × store × week grid; the rest falls back to chain price | inherent to the data | documented |
| Availability is proxied by "this store sold this item"; dunnhumby has no stock-out feed | inherent to the data | documented |
| `IV_hat` is unbiased for the *sum* but biased low for its *log* (Jensen). Absorbed by `c0` in training, and generation uses the same estimator | low — consistent between fit and use | documented |
| Placebo battery (`25_basket_placebo.py`) was run on the chain-level panel, not the store-level one | low — stores contribute +0.0005 nats, so the panels are near-identical in practice | open |

---

## 10. Changelog

| # | change | why |
|---|---|---|
| initial | four-head nested model: incidence with `κ`, item choice, quantity counts, breadth; stores via price, affinity, availability | restores what `BASKET_MODEL.md` dropped |
| 1 | availability threshold 3 → 1 sale | a threshold of 3 marked genuine purchases unavailable; incidence loss hit 5.1M |
| 2 | inclusive value sampled (`iv_cap`) rather than summed over padded blocks | dense blocks are mostly padding and cost 5.4× the item head |
| 3 | case-control offset on the incidence logit | positives over-sampled 30×; generation produced 58 categories per basket against a real 6.5 |
| 4 | elasticity decomposition reported from means, not medians | medians do not decompose additively; the parts summed to 82% |
| 5 | `28` indexes the chosen item by position, not by mask | padding slots hold item id 0, so `blk == j` also fired on every pad slot when the chosen item was item 0 |
| 6 | placebo scrambles store deviations as well as the chain panel | leaving them real would leak genuine prices into the placebo |
| 7 | `--avail-only` / `--no-store-price` flags | the store gain conflates real information with a mechanically smaller choice set |
| 8 | frozen per-category IV reference | training centred the inclusive value on the batch, generation on a different set; `κ` absorbed the difference and generation ran 60% high |
| 9 | IV reference taken from a *uniform* category sample | taking it from `incidence_batch` used the case-control sample, which over-weights bought categories; the reference sat ≈0.9 too high and generation ran 38% low |
| 10 | **incidence sampling switched from case-control to uniform** | the `log(π₁/π₀)` offset corrects the intercept but not `κ·(IV − ref)`, whose mean differs between sample and population. Uniform removes both biases at source |
| 11 | generator draws units from the quantity head | it accumulated the category pick-count directly, giving every item exactly 1 unit — 1.007 against a real 1.348, while the head itself predicted 1.389 |
| 12 | `--no-state` flag added | criterion 3 was written but could not be tested; the flag did not exist |
| 13 | **breadth head** — `distinct items − 1 ~ Poisson`, per category | the only *model* gap among these fixes. The incidence head sees 0/1, so deriving a count from `P(buy)` can only ever yield ≈1.0–1.1 items per purchased category against a real 1.284 |
| 14 | `--no-context` flag added | the tied interaction was hard-wired, so the claim that it is load-bearing rested on the flat model's evidence and had never been tested here |
| 15 | benchmark scorer gives each model its own basket context | the shared batch builder computed context from a stub with zero `α`, so the nested model was scored with its interaction disabled and lost to its own no-interaction ablation |
| 17 | **`generate_baskets`: real item ids, plus Gibbs sweeps for the context** | `generate` only ever accumulated counts, so nothing downstream could consume it and the interaction term never reached the output. Gibbs lifts within-basket $\alpha^\top\alpha$ 0.1334 → 0.2820 (2.1×), reaching 51% of the real level, with basket size unchanged |
| 16 | **this document rewritten against the code** | §5 still described case-control sampling with the `log(π₁/π₀)` offset, three releases after fix 10 replaced it with uniform sampling. The audit that found it is §12 |

---

## 11. How the exploration was sequenced, and what that cost

`DATA_EXPLORATION.md` describes the data on its own terms and makes no reference to
this model. That separation was arrived at late; this section records why, because the
sequencing error was expensive and is easy to repeat.

The first version of the exploration examined only the three assumptions this model set
out to overturn: unit demand, category independence, no state. It was silent on price
response, quantity, stores, household heterogeneity and base rates. Everything it
omitted was later discovered by a **bug**, not by looking at the data:

| what was missing | how it surfaced | cost |
|---|---|---|
| does demand respond to price, and by how much? | the elasticity first appeared inside `25_basket_placebo.py`, long after the model was built | a fitted coefficient of +0.081 against a true ≈0.95 went unnoticed until an unrelated test caught it |
| units per line, and the quantity margin | only measured when challenged | a quarter of the price response was assumed away |
| store price dispersion and assortment | only measured when challenged | prices pooled across 115 stores; unstocked items scored as "rejected" |
| **category incidence base rate** | only after the generator produced **58 categories per basket** against a real 6.5 | a `log(30)` calibration error and five full retrains |
| **household taste and price sensitivity** | only when asked directly whether the exploration was exhaustive | the premise of the whole model went unmeasured through every version of it |
| breadth — distinct items per category purchase | caught by the coverage audit, before a bug | none |

### The two lessons

**Explore what the model will have to reproduce, not what you intend to change.** The
base rate of 3.23% is one line of pandas; the whole demand exploration runs in about two
seconds. Both were sitting in the same parquet file the entire time.

**Audit every model term against the exploration, mechanically.** Household taste and
price sensitivity never registered as things to check *because the model already had
parameters for them* — having `θ_i` in the specification made it feel established. A
model fits per-household parameters whether or not the heterogeneity is real. Only the
split-half correlation distinguishes signal from noise, and computing it took a direct
challenge.

---

## 12. What this rewrite corrected

This document was audited line by line against `27_nested_basket.py`,
`28_nested_counterfactual.py`, `22_basket_data.py` and `31_benchmark.py`, with every
number re-read from `out/`. What was wrong:

| claim as written | what the code and artefacts say |
|---|---|
| §5 "Incidence is case-control sampled, and corrected", with a `log(π₁/π₀)` offset applied during training | **Sampling is uniform** (`27:364`) and there is no correction. Fix 10 replaced case-control three releases earlier and §5 was never updated. `off` is still built and is always 0.0, and `losses` never reads it |
| "negatives drawn only from items the trip's store actually stocks" (§5, and the module docstring at `27:51`) | Negatives are drawn from the **whole catalogue** (`27:310`) and unstocked ones are then masked to `−1e9`. The effective competitor count is therefore below 20 and varies by store |
| "+0.394 nats" margin over household repeat-purchase (§8.1c) | +0.342 nats. `−2.0011 − (−2.3428)` |
| "0.300 purity (70.1× chance), AUC 0.828" on the full catalogue (§8.5) | 0.2976 purity, 69.6× chance, AUC 0.8222. The old figures appear in no artefact; the flat model's are 0.3023 / 70.6× / 0.8233, so criterion 6 is marginally **missed**, not met |
| "8.4 min, 42.1 ms/iter" (§6) | 646 s for 12,000 iterations = 53.8 ms/iter, from `nested_nested_history.json` |
| "2.3% of the grid" for store-level prices (§4) | 0.53% — 244,880 cells over 399,579 × 115 (item-week × store) |
| "199,347 baskets" (§2) | 199,345, from `meta.json` |
| generation errors "−1.1% / −1.4% / −3.7%" (§8.6) | -1.0% / -1.4% / -3.8% |
| "1.323 items per purchased category against a real 1.288" (§8.6) | 1.324 against 1.287 |
| §7 marked criteria 4, 5 and 6 as ✅ | 4 is weakly identified, 5 is 99.7% mechanical, 6 is marginally below target. All three are now ⚠️ with the qualification stated |
| §3 gave the item utility but never the state basis, the head likelihoods, the objective, or the L2 split | all now written out (§3.2, §3.5, §3.6) |
| nothing recorded that the item log-likelihood is **conditional on the rest of the basket** | stated in §8.1c, with the joint-likelihood gap added to §9 |
| "unit demand fails in 56% of baskets" (§1) | **68.2%** on `basket_input`: 48.6% hold >1 item from some category, 57.6% have a line buying >1 unit. The 56% figure is not reproducible from any current artefact |

Four pieces of dead code were found in the process and are recorded in §9: `--iv-center`,
the `offset` field, `gstart_keys`/`gstart_vals`, and a mislabelled log line at `28:310`.
None affects a reported result.
