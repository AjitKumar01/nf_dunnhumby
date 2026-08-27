# Theory

How the model is defined, why its normaliser is computable, and what each design choice is
paying for. Every number quoted here was measured on this dataset; where a claim is a
limitation rather than a result, it says so.

---

## 1. The object being modelled

A shopping trip produces a **set**, not a sequence of independent choices. A household walks
into store `s` on day `t`, faces an assortment `A_s` of a few thousand products at posted
prices, and leaves with a subset `S ⊆ A_s`.

Modelling this as `|A_s|` independent binary choices throws away the two things that make a
basket a basket: items are bought *together* (hot dogs and buns), and the basket has a
*size* that is itself a decision. Modelling it as a sequence (as SHOPPER does) imposes an
order that the data does not contain.

So the model puts a distribution directly on subsets:

```
P(S | x) = exp( E(S, x) ) / Z(x),        Z(x) = sum over ALL subsets of A_s
```

with the energy

```
E(S) =   sum_{j in S}  b_j(x)                    (1) utility of each item
       + sum_{j<k, both in S} phi_j' phi_k       (2) pairwise interaction, low rank
       - sum_c rho_c * C(n_c, 2)                 (3) within-category effect, n_c items from category c
       - rho_0(|S|)                              (4) basket-size potential
```

**(1) utility.** `b_j` carries price, promotion (display, mailer), seasonality, store, the
household's taste, and how long since that household last bought the item's sub-commodity.
Section 6 covers the price part, which is where most of the modelling difficulty is.

**(2) interaction.** `phi_j ∈ R^Kz` with `Kz = 4`. The pair effect `phi_j' phi_k` is
positive for complements and negative for items that repel. It is deliberately low rank and
sparse — section 4 explains why that is forced, not chosen for convenience.

**(3) within-category.** `n_c` is how many items the basket takes from category `c`.
`rho_c < 0` means "having taken one from this category, take more" — complementarity.
Fitted on this data, `rho_c` is negative, and section 7 shows the co-purchase evidence.

**(4) size.** `rho_0(n)` is a free function of basket size, so the model can reproduce the
observed size law (median 4, mean 7.8, long right tail) instead of inheriting whatever size
distribution the item terms happen to imply.

---

## 2. The normaliser, and why it is not hopeless

`Z` sums over `2^|A_s|` subsets. With `|A_s| ≈ 5,000` that is not an approximation problem,
it is an impossibility — unless the energy has structure. It does.

The only term coupling items pairwise is (2). Complete the square:

```
sum_{j<k in S} phi_j' phi_k  =  ½( ||v_S||²  -  sum_{j in S} ||phi_j||² ),
                                                    v_S = sum_{j in S} phi_j
```

and apply the **Hubbard–Stratonovich identity**, which linearises a squared norm by
integrating against a Gaussian:

```
exp( ½||v||² )  =  E_{z ~ N(0, I_Kz)} [ exp( v'z ) ]
```

This is exact for any `v` — it is just the Gaussian moment generating function. Substituting
and exchanging the sum over subsets with the expectation over `z`:

```
Z  =  E_z [ f(z) ],        f(z) = sum over subsets of  prod_{j in S} w_j(z) * (size & category terms)
w_j(z) = exp( b_j - ½||phi_j||² + phi_j' z )
```

**Conditional on `z`, the items are independent.** The pairwise coupling has been traded for
one `Kz`-dimensional integral. That is the whole trick: a `2^5000` sum becomes a 4-dimensional
quadrature over an integrand that itself factorises.

### What `f(z)` is

`log f(z)` is the **cumulant generating function of `v_S`**. That is not a curiosity, it is
the diagnostic tool that makes the model debuggable:

```
grad log f(0)   = E[v_S]
grad² log f(0)  = Cov(v_S)
```

so quantities that would otherwise need sampling are available by differentiating the same
object the normaliser already computes. `pi_j = P(j in S)` is likewise exact:

```
pi_j = d log(Z - 1) / d b_j
```

(the `-1` drops the empty basket, which is not an observed trip). This is `pi_quad` in the
code, computed by autograd through the quadrature — and it guarantees `sum_j pi_j = E[n]`
by construction rather than by hope, which is the sanity check used throughout.

---

## 3. Evaluating `f(z)`: elementary symmetric polynomials on a ragged index

Given `z`, items are independent, but the size potential (4) and the category term (3)
still couple them *through counts*. Both are functions of counts only, which is exactly
what elementary symmetric polynomials handle.

For a set of weights `w_1..w_m`, the elementary symmetric polynomial `e_r` is the sum over
all `r`-subsets of the product of their weights:

```
e_r = sum_{|T| = r} prod_{j in T} w_j
```

computed by the recursion `e_r^{(k)} = e_r^{(k-1)} + w_k * e_{r-1}^{(k-1)}` in `O(m·R)`.
So `e_r` gives "the total weight of all ways to take exactly `r` items", and the category
term `exp(-rho_c C(r,2))` — a function of `r` alone — multiplies straight into it.

The assortment is organised as **(store, category) rows**:

* rows are ragged: the median row holds 3 products and the largest 1,773, but weighted
  by where purchases actually fall the median is 128 -- shoppers buy from the big rows;
* padding every row to the maximum would waste ~20× the work, so items live in one flat
  array with a row index, and only the short category axis is padded;
* a purchased product is stored as its **position within its row**, because that is what
  the polynomial indexes.

`A_n(z)` — total weight of all baskets of size `n` — is then a convolution of the per-row
polynomials across rows, and `f(z) = sum_n exp(-rho_0(n)) A_n(z)`.

### The truncation degree is a numerical-validity constraint

The per-row polynomial is truncated at `poly_degree`. This looks like a speed knob. It is not.

`exp(-rho_c C(n,2))` at the fitted `rho_c = -0.337` and `n = 120` is `10^1045`, against
float64's ceiling of `10^308`. The untruncated recursion returns NaN. Worse, degrees just
below overflow are **finite and meaningless**: at degree 64, `sum_j pi_j = 120.00 = n_max`
("every product is certain") when the truth is 7.6.

The safe degree therefore depends on `rho_c`, and so on the checkpoint — run404
(`rho_c = -0.211`) is safe to 48, run413 (`rho_c = -0.337`) only to 32. It must also be at
least the largest per-category count actually present in the data (26), which is the
smallest degree that gives an observed basket non-zero probability.

Calibration must run **upward from that floor**, on the weights that will actually be used.
Calibrating downward against the untruncated polynomial cannot work — that reference is the
overflowing one, and `NaN <= tol` is false, so the search falls through and returns the
worst option.

---

## 4. The quadrature, and the budget it imposes on `phi`

`Z = E_z[f(z)]` is a `Kz`-dimensional Gaussian integral, done on a **Smolyak sparse grid**
(level `q = 8`, 681 nodes at `Kz = 4`). Sparse grids need far fewer nodes than a tensor
product for the same polynomial exactness, which matters because every node costs a full
pass over the assortment.

Smolyak weights are **signed**, and that is its failure mode: at large `rho_c` the
cancellation between positive and negative weights loses precision and the estimate
diverges. Measured here, that begins above `rho ≈ 0.7` at `Kz = 4`, which is outside the
fitted range but is why the integrator is a single switchable choice in the code
(`set_quad`) rather than hard-wired.

### `lambda_max`: a budget, not a regulariser

The Gaussian quadrature is centred at the origin with the identity as its covariance. The
integrand's curvature is `grad² log f = Cov(v_S)`, so the effective covariance of the tilted
measure is `(I - Cov(v_S))^{-1}`. That stays positive definite only while

```
lambda_max  =  sum_j pi_j ||phi_j||²  <  1
```

This is a **budget on total interaction strength**, and it is why `phi` is sparse: with
`E[n] ≈ 8`, spending the entire budget uniformly gives `||phi_j|| ≤ 0.326` and hence
`|phi_j' phi_k| ≤ 0.106`, confining pair effects to lifts in `[0.90, 1.11]`. Strong
interactions are only affordable for products with small `pi_j`.

Note what this budget is *not*: it is not a condition for the model to exist. The
Hubbard–Stratonovich integral converges for any `phi`, because Gaussian tails dominate
`exp(c|z|)`. It is a condition for *this integrator* to be accurate.

---

## 5. Sampling: exact given `z`

The model is usable as an environment because a basket can be drawn without any Markov
chain. `Z = E_z[ sum_n exp(-rho_0(n)) A_n(z) ]` and `A_n` is a convolution over rows, so the
joint factorises into a chain that is walked top-down:

1. **`z`**, from its posterior, by sampling-importance-resampling on the same proposal the
   normaliser uses. *This is the only inexact step*, and it is consistent as draws grow.
2. **the size `n`**, from `P(n | z) ∝ exp(-rho_0(n)) A_n(z)`. Exact — these are the terms
   `log f` already sums over.
3. **the split of `n` across categories**, by walking the convolution backwards. Exact.
4. **which products fill each row's slots**, by walking the ESP recursion backwards. Exact.

Steps 3 and 4 build tables over up to 1,773 products, so they must be normalised with their
log-scale carried, or the ratios underflow and the walk silently returns short baskets.

The correctness criterion is that mean sampled basket size reproduces `sum_j pi_j`, which is
exact by section 2. Measured on 576 baskets: within one standard error.

---

## 6. Price: what is identified, and how

Price is where the model earns or loses its claim to answer counterfactuals. The term is

```
b_j  ⊃  - gb_j * [ dbar  +  kappa * ( dlp_j - dbar ) ],       gb_j = sum_k gamma_{h,k} beta_{j,k}
```

where `dlp_j` is the item's log-price deviation, `dbar` a **reference price level**, `gamma`
a household loading and `beta` a product loading. The split is the whole point:

* a **uniform** price rise moves `dlp` and `dbar` together, so only `gb` acts;
* an **idiosyncratic** rise moves `dlp_j - dbar`, so `gb * kappa` acts.

That separates two elasticities the data reports very differently.

### The three elasticities, measured from the data

Estimated independently of the model, from an item-week panel with item fixed effects and
display/mailer controls (`n = 42,464` item-weeks, `990` products):

| quantity | estimate | se |
|---|---|---|
| own-price, no promo controls | −1.0037 | 0.0108 |
| **own-price, with display + mailer** | **−0.7725** | 0.0114 |
| **cross-price, rival = mean log price in the same sub-commodity** | **+0.1351** | 0.0172 |
| aggregate (basket size vs a uniform rise) | −0.121 | — |

These are targets the model is checked against, not quantities fitted to it.

### `kappa` must be initialised, not trained

`elast_own ≈ -gb * kappa` and `elast_agg ≈ -gb * Var(n)/E[n]`. A projection pins `gb` so the
aggregate matches −0.121, which leaves `kappa` to carry the own-price response.

`kappa`'s natural scale is ~40, so the structural learning rate moves it 0.005% per step;
its gradient is also small and sign-noisy across 24-trip minibatches, which Adam averages to
near nothing. Measured: it travels **1.4 units per 1,000 iterations even at 20× the rate**,
so a run would need ~50,000 extra iterations to arrive. **Where it starts decides where it
ends.**

It is identified without the model — `kappa* = (elast_own/elast_agg)·Var(n)/E[n] ≈ 45` — and
a sweep on the fitted model puts the likelihood's *own* optimum at `kappa ∈ [40, 60]`,
spanning elasticities −0.71 to −1.00. Data and likelihood agree, so the model is started at
the value they agree on (`--kappa-init 44`).

### Cross-price substitution lives in the reference, not the interaction

A rival's price rise moves `dbar`, and

```
d b_j / d dbar  =  gb_j * (kappa - 1)  =  +0.535 at kappa = 35.6
```

— strongly positive, i.e. substitution. But the magnitude is scaled by how much one rival
actually moves the reference, `n_riv / n_ref`. Referenced to the **whole assortment**
(a median of 5,312 products per store) that is tiny: the channel is diluted ~5,000× and the basket-size
effect swamps it, giving cross-price **−0.162**, the wrong sign. Referenced to the store's
own **category** (median 128 purchase-weighted) it is `d/n_c`, giving **+0.044**. Referenced to the
**sub-commodity** (median 16 purchase-weighted), **+0.502**.

The reference width is a single knob trading own-price against cross-price, and the data's
`+0.1351` sits at an effective width of ~85 products, between the last two. The shipped
model uses the category reference: correct sign, a third of the target magnitude, and the
best likelihood of the three.

### Why the interaction term cannot do this job

It is tempting to fix cross-price with `phi`. It cannot work, and the reason is exact.
Since `pi_k = d log(Z-1)/d b_k`,

```
d pi_k / d b_j  =  Cov( 1{j in S}, 1{k in S} )
```

so a pair term's leverage on a *marginal* is second order in `pi_j · pi_k ≈ 10⁻³`. Measured
on a hand-constructed best case — two popular same-sub-commodity products with embeddings
driven maximally apart — pushing `phi_j' phi_k` from −0.01 to −0.64 moved the cross-price
elasticity from −0.0125 to +0.0015. **93× short of target**, with `lambda_max` never
exceeding 0.216, so the budget of section 4 was not even the binding constraint. Raising
`Kz`, unmasking `phi`, or adding a repulsive `rho` at sub-commodity granularity would all
fail for the same reason.

---

## 7. What the co-purchase data actually says

A warning about a statistic that is easy to get wrong here.

Pairwise **lift**, `P(j,k together) / (P(j)P(k))`, conflates two things: whether a household
buys the category at all (correlated across its members, pushing lift up) and, given that it
does, whether it buys one item or several. Restricting to pairs that co-occur at least a few
times makes it worse — that conditions *on* co-occurrence, which is precisely what substitutes
do not do, and it reported a median same-sub-commodity lift of 86.8 here. Computed over the
complete matrix with zeros kept, the same statistic is 1.57.

On the margin `rho` actually acts on — the **count within a group** — households buy 2+ items
from a sub-commodity **8.6× more often than independence**, and from an affinity category
31.7× more often. That is why `rho_c` is fitted negative (complementary), and why a
*repulsive* `rho` at sub-commodity granularity would fit the opposite of the data.

Substitution is nonetheless real: it shows up under **price variation** (+0.1351 above), not
under co-occurrence. The two are not in conflict — co-occurrence is dominated by shopper
heterogeneity and category incidence, while the price response identifies the causal margin.

---

## 8. Objective and constraints

Training maximises the exact set log-likelihood `E(S) - log Z` per trip, plus:

* a **units** likelihood for how many of each item;
* a **size cross-entropy** calibrating `P(n)` against the observed size law;
* an **elasticity projection** forcing `gb` onto the aggregate target — a projection, not a
  penalty, because with Adam the step size is set by the gradient's sign and running scale,
  so a penalty with a huge gradient still only moves the parameter by about `lr` per step
  and the likelihood simply out-pushes it (measured: with weight 20 the elasticity went
  −0.765 → −4.871 over 400 iterations, the wrong way);
* a **`phi` cap and `lambda_max` projection** keeping the quadrature valid;
* a **`rho_c` floor** at −0.92, since a 2.5× pair lift is `rho_c = -0.92` and the term
  detonates numerically below that.

`phi = 0` is a **saddle**, not a minimum: `dE(S)/d phi_j = sum_{k in S} phi_k`, which is zero
when every `phi` is zero, so escape from a zero initialisation is exponential. This is why
`phi` is placed spectrally — by eigendecomposing the empirical log-lift matrix — rather than
seeded with noise. Measured, spectral placement beat 15,000 SGD updates from a noise seed.
