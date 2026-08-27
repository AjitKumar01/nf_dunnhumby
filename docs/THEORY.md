# Theory

The model, derived. Every result below is proved or computed, not asserted, and each
derivation is followed by a plain reading of what it means. Numbers quoted are measured on
this dataset and carry the formula they come from.

Equations are LaTeX and render on GitHub.

**Contents**

1. [Setup and notation](#1-setup-and-notation)
2. [The energy function](#2-the-energy-function)
3. [The normalising constant, and why it is not hopeless](#3-the-normalising-constant)
4. [The Hubbard–Stratonovich transform](#4-the-hubbardstratonovich-transform)
5. [What $\log f$ is: the cumulant generating function](#5-what-log-f-is)
6. [Evaluating $f(z)$: elementary symmetric polynomials](#6-evaluating-fz)
7. [Inclusion probabilities and their derivatives](#7-inclusion-probabilities)
8. [The quadrature and the $\lambda_{\max}$ budget](#8-the-quadrature-and-the-budget)
9. [Numerical validity: the truncation degree](#9-numerical-validity)
10. [Exact sampling](#10-exact-sampling)
11. [Price: identification and the three elasticities](#11-price)
12. [Why the interaction cannot produce substitution](#12-why-the-interaction-cannot-substitute)
13. [Objective, projections, and the saddle at $\phi=0$](#13-objective-and-projections)

---

## 1. Setup and notation

A household $h$ visits store $s$ on day $t$. The store carries an **assortment**
$\mathcal{A}_s \subseteq \{1,\dots,J\}$ of products, each at a posted price. The household
leaves with a **subset** $S \subseteq \mathcal{A}_s$.

| symbol | meaning | value here |
|---|---|---|
| $J$ | products | 5,455 |
| $N$ | households | 2,066 |
| $S_{\text{tore}}$ | stores | 115 |
| $C$ | categories (the partition defining rows) | 280 |
| $\mathcal{A}_s$ | store $s$'s assortment | median 5,312 products |
| $n = \lvert S\rvert$ | basket size | mean 7.84, median 4, $\mathrm{Var}=82.9$ |
| $n_c$ | items taken from category $c$ | |
| $x$ | context: prices, promotion, week, store, household state | |

The model is a distribution over subsets,

$$
P(S \mid x) \;=\; \frac{\exp\big(E(S,x)\big)}{Z(x)},
\qquad
Z(x) \;=\; \sum_{S \subseteq \mathcal{A}_s} \exp\big(E(S,x)\big).
$$

**Plainly.** Every possible basket gets a score $E$; the probability of a basket is its
score exponentiated, divided by the total over *all* baskets. The difficulty is entirely in
that total: there are $2^{5312}$ possible baskets at the median store.

**Why a set and not something simpler.** Treating the trip as $\lvert\mathcal{A}_s\rvert$
independent yes/no choices discards the two facts that make a basket a basket: items are
bought *together*, and the *size* of the basket is itself a decision. Treating it as a
sequence (as SHOPPER does) imposes an ordering the data does not contain.

---

## 2. The energy function

$$
E(S) \;=\;
\underbrace{\sum_{j \in S} b_j(x)}_{\text{(1) utility}}
\;+\; \underbrace{\sum_{\{j,k\} \subseteq S,\; j<k} \phi_j^{\top}\phi_k}_{\text{(2) interaction}}
\;-\; \underbrace{\sum_{c=1}^{C} \rho_c \binom{n_c}{2}}_{\text{(3) within-category}}
\;-\; \underbrace{\rho_0(n)}_{\text{(4) size}}
$$

**(1) Utility.** $b_j$ collects everything about product $j$ on its own:

$$
b_j \;=\; \lambda_j \;+\; \theta_h^{\top}\alpha_j \;+\; \xi_s^{\top}\zeta_j
\;+\; \text{season}_j \;+\; \delta^{\text{disp}} d_j + \delta^{\text{mail}} m_j
\;-\; g_j\Big[\bar{\ell} + \kappa\big(\ell_j - \bar{\ell}\big)\Big] \;+\; \psi\,\mathrm{rec}_j
$$

with $\lambda_j$ a product intercept, $\theta_h \in \mathbb{R}^{32}$ the household's taste,
$\ell_j$ the log-price deviation, $\bar\ell$ a **reference price level** (§11), and
$g_j = \sum_{p=1}^{K_p}\gamma_{h,p}\beta_{j,p}$ the household $\times$ product price loading
with $K_p = 8$.

**(2) Interaction.** $\phi_j \in \mathbb{R}^{K_z}$, $K_z = 4$. The pair term
$\phi_j^{\top}\phi_k$ is positive for items that go together, negative for items that repel.
Restricted to 30 products; §8 shows that sparsity is forced, not chosen.

**(3) Within-category.** $\rho_c < 0$ makes a second item from category $c$ *more* likely —
complementarity. §13 gives the measured evidence for the sign.

**(4) Size.** $\rho_0: \{0,\dots,n_{\max}\} \to \mathbb{R}$ is a free function, so the model
reproduces the observed size law rather than inheriting whatever the item terms imply.

**Plainly.** Score a basket by adding up how much each item is wanted, plus a bonus or
penalty for every pair in it, minus a term for taking many items from one category, minus a
term for the basket's overall size.

---

## 3. The normalising constant

$Z$ sums over $2^{\lvert\mathcal{A}_s\rvert}$ subsets. At the median store that is
$2^{5312} \approx 10^{1599}$ terms. This is not an approximation problem; it is an
impossibility unless the energy has exploitable structure.

It does. Terms (1), (3) and (4) are **separable given the counts**: (1) is a sum over
members, (3) and (4) depend on $S$ only through $n_c$ and $n$. Only term (2) genuinely
couples pairs. The next section removes it.

---

## 4. The Hubbard–Stratonovich transform

### 4.1 Completing the square

Let $v_S := \sum_{j\in S}\phi_j \in \mathbb{R}^{K_z}$.

$$
\lVert v_S\rVert^2 \;=\; \Big(\sum_{j\in S}\phi_j\Big)^{\!\top}\Big(\sum_{k\in S}\phi_k\Big)
\;=\; \sum_{j\in S}\lVert\phi_j\rVert^2 \;+\; 2\!\!\sum_{j<k \in S}\!\!\phi_j^{\top}\phi_k
$$

and therefore, exactly,

$$
\boxed{\;\sum_{j<k\in S}\phi_j^{\top}\phi_k \;=\; \tfrac{1}{2}\lVert v_S\rVert^2
\;-\; \tfrac{1}{2}\sum_{j\in S}\lVert\phi_j\rVert^2\;}
\tag{4.1}
$$

The pairwise sum has become a **squared norm of a sum** plus a term separable over members.

### 4.2 Linearising the square

For $z \sim \mathcal{N}(0, I_{K_z})$ the moment generating function is
$\mathbb{E}_z[e^{v^{\top}z}] = e^{\frac{1}{2}\lVert v\rVert^2}$, hence

$$
\boxed{\;\exp\!\big(\tfrac12\lVert v\rVert^2\big) \;=\;
\mathbb{E}_{z\sim\mathcal{N}(0,I)}\big[\exp(v^{\top}z)\big]\;}
\tag{4.2}
$$

This is exact for every $v$ — no approximation, no condition.

### 4.3 Putting them together

Substituting (4.1) and (4.2) into $\exp(E(S))$ and defining the **tilted weight**

$$
w_j(z) \;:=\; \exp\!\Big(b_j - \tfrac12\lVert\phi_j\rVert^2 + \phi_j^{\top}z\Big),
$$

we get

$$
\exp\big(E(S)\big) \;=\; \mathbb{E}_z\Big[\;\prod_{j\in S} w_j(z)\;\Big]\cdot
\exp\Big(-\sum_c \rho_c\tbinom{n_c}{2} - \rho_0(n)\Big)
$$

and, exchanging the finite sum with the expectation,

$$
\boxed{\;Z \;=\; \mathbb{E}_{z}\big[f(z)\big],\qquad
f(z) \;=\; \sum_{S\subseteq\mathcal{A}}\;\prod_{j\in S} w_j(z)\;\;
\exp\Big(-\sum_c \rho_c\tbinom{n_c}{2}-\rho_0(n)\Big)\;}
\tag{4.3}
$$

**Conditional on $z$, the items no longer interact.** The only coupling left is through the
counts $n_c$ and $n$, which §6 handles exactly.

**Plainly.** The pair term is what makes the sum impossible. Equation (4.2) says a squared
quantity can be written as an average over a Gaussian random vector $z$. Introducing that
$z$ buys independence: for any *fixed* $z$ the items stop interacting, and the intractable
$2^{5312}$ sum becomes a 4-dimensional integral over an integrand we can evaluate exactly.
A $2^{5312}$ problem has been traded for a 4-dimensional one.

---

## 5. What $\log f$ is

Write $f(z) = \sum_S g(S)\,e^{v_S^{\top}z}$ where
$g(S) = \exp\big(\sum_{j\in S}(b_j - \frac12\lVert\phi_j\rVert^2) - \sum_c\rho_c\binom{n_c}{2} - \rho_0(n)\big)$
does not depend on $z$. Define the probability measure $q(S) \propto g(S)$. Then

$$
\frac{f(z)}{f(0)} \;=\; \mathbb{E}_{S\sim q}\big[e^{v_S^{\top}z}\big]
\qquad\Longrightarrow\qquad
\boxed{\;\log f(z) - \log f(0) \;=\; K_{v_S}(z)\;}
\tag{5.1}
$$

the **cumulant generating function** of the random vector $v_S$ under $q$. Immediately,

$$
\nabla \log f(0) \;=\; \mathbb{E}_q[v_S], \qquad
\nabla^2 \log f(0) \;=\; \operatorname{Cov}_q(v_S).
\tag{5.2}
$$

**Plainly.** $\log f$ is not an arbitrary function to be evaluated numerically — it is a
CGF, so its slope at the origin is a mean and its curvature is a covariance. Anything that
would otherwise need simulation is available by differentiating the object the normaliser
already computes. This is what makes the model debuggable, and §8 uses the curvature
directly.

---

## 6. Evaluating $f(z)$

### 6.1 Elementary symmetric polynomials

For weights $w_1,\dots,w_m$ the $r$-th elementary symmetric polynomial is

$$
e_r(w_1,\dots,w_m) \;=\; \sum_{\lvert T\rvert = r,\; T\subseteq\{1..m\}}\;\prod_{j\in T} w_j
$$

— "the total weight of every way to choose exactly $r$ of them". It obeys

$$
e_r^{(k)} \;=\; e_r^{(k-1)} \;+\; w_k\, e_{r-1}^{(k-1)},
\qquad e_0^{(k)} = 1,\quad e_r^{(0)} = 0\;(r>0)
\tag{6.1}
$$

*Proof.* Split the $r$-subsets of $\{1..k\}$ by whether they contain $k$: those that do not
are counted by $e_r^{(k-1)}$; those that do contribute $w_k$ times an $(r-1)$-subset of
$\{1..k-1\}$. $\square$

Cost: $O(mR)$ for orders up to $R$.

### 6.2 Rows, and why the category term slots in

Group the assortment into $(\text{store},\text{category})$ **rows**. Because term (3)
depends on row $c$ only through its count $r = n_c$, it multiplies straight into the
polynomial:

$$
G_c(z, r) \;=\; e_r\big(\{w_j(z)\}_{j\in c}\big)\cdot \exp\!\Big(-\rho_c \tbinom{r}{2}\Big)
\tag{6.2}
$$

Total weight of all baskets of size $n$ is the convolution across rows,

$$
A_n(z) \;=\; \sum_{r_1 + \cdots + r_C = n}\; \prod_{c=1}^{C} G_c(z, r_c),
\qquad
f(z) \;=\; \sum_{n\ge 0} e^{-\rho_0(n)} A_n(z).
\tag{6.3}
$$

### 6.3 The ragged layout

Rows are ragged — median 3 products, maximum 1,773, and **128 weighted by where purchases
actually fall**. Padding every row to the maximum would waste roughly $20\times$ the work,
so items are held in one flat array with a row index and only the short category axis is
padded. A purchased product is stored as its **position within its row**, because that is
what (6.1) indexes.

**Plainly.** Given $z$, "what is the total weight of all baskets that take exactly 3 items
from dairy?" is answered by a simple recursion. Do that per category, then convolve across
categories to get "all baskets of size $n$", then weight by the size potential. Nothing is
enumerated.

---

## 7. Inclusion probabilities

### 7.1 First derivative

Since $Z = \sum_S \exp(\sum_{j\in S} b_j + \cdots)$,

$$
\frac{\partial \log Z}{\partial b_j}
= \frac{1}{Z}\sum_S \exp(E(S))\,\mathbb{1}\{j\in S\}
= P(j \in S) =: \pi_j .
$$

The empty basket is not an observed trip and $\exp(E(\varnothing)) = 1$, so it is removed:

$$
\boxed{\;\pi_j \;=\; \frac{\partial \log (Z-1)}{\partial b_j}\;}
\tag{7.1}
$$

Consequently $\sum_j \pi_j = \mathbb{E}[n]$ **by construction**, which is the consistency
check used throughout this project.

### 7.2 Second derivative

$$
\boxed{\;\frac{\partial \pi_k}{\partial b_j}
\;=\; \frac{\partial^2 \log Z}{\partial b_j \partial b_k}
\;=\; \operatorname{Cov}\big(\mathbb{1}\{j\in S\},\,\mathbb{1}\{k\in S\}\big)\;}
\tag{7.2}
$$

the standard exponential-family result: second derivatives of the log-partition function are
covariances of sufficient statistics. §12 turns this into a hard limit.

**Plainly.** The chance that item $j$ ends up in the basket is the slope of $\log Z$ in that
item's utility. How much item $k$'s chance moves when item $j$'s utility moves is the
*covariance* of the two items' presence. Both come free from the normaliser.

---

## 8. The quadrature and the budget

### 8.1 The integrator

$Z = \mathbb{E}_z[f(z)]$ is a $K_z$-dimensional Gaussian integral, evaluated on a **Smolyak
sparse grid** (level $q=8$, **681 nodes** at $K_z=4$). Sparse grids reach a given polynomial
exactness with far fewer nodes than a tensor product, which matters because each node costs
a full pass over the assortment.

Smolyak weights are **signed**. That is the failure mode: at large $\rho$ the cancellation
between positive and negative weights loses precision and the estimate diverges — measured
here above $\rho \approx 0.7$ at $K_z=4$, outside the fitted range, which is why the
integrator is a single switchable choice (`set_quad`) rather than hard-wired.

### 8.2 Deriving $\lambda_{\max}$

By (5.1), $f(z) = f(0)\exp(K(z))$ with $K$ the CGF of $v_S$. Expanding to second order,
$K(z) \approx a^{\top}z + \frac12 z^{\top}Bz$ with $a = \mathbb{E}[v_S]$,
$B = \operatorname{Cov}(v_S)$. Then

$$
\mathbb{E}_z[f(z)] \;\approx\; f(0)\,(2\pi)^{-K_z/2}\!\int
\exp\!\Big(-\tfrac12 z^{\top}(I - B)z + a^{\top}z\Big)\,dz
$$

which is finite iff $I - B \succ 0$, i.e.

$$
\boxed{\;\lambda_{\max}(B) < 1\;}
\tag{8.1}
$$

Under a Poisson-binomial approximation ($\mathbb{1}\{j\in S\}$ roughly independent
Bernoulli($\pi_j$)),
$B \approx \sum_j \pi_j(1-\pi_j)\,\phi_j\phi_j^{\top}$, so

$$
\lambda_{\max}(B) \;\le\; \operatorname{tr}(B) \;\approx\; \sum_j \pi_j\lVert\phi_j\rVert^2
\;=:\; \lambda_{\max}^{\text{est}} .
\tag{8.2}
$$

### 8.3 What the budget costs

With $\mathbb{E}[n] = \sum_j\pi_j \approx 8$, spending the entire budget uniformly gives

$$
8\,\lVert\phi\rVert^2 < 1 \;\Longrightarrow\; \lVert\phi_j\rVert < 0.354
\;\Longrightarrow\; \lvert\phi_j^{\top}\phi_k\rvert \le 0.125,
$$

confining pair effects to lift factors $e^{\pm 0.125} \in [0.88,\,1.13]$. Strong pair
effects are affordable only for products with small $\pi_j$.

**Important:** (8.1) is a condition on the **integrator**, not on the model. The exact
integral (4.3) converges for *any* $\phi$, because Gaussian tails $e^{-\lVert z\rVert^2/2}$
dominate $e^{c\lVert z\rVert}$. What fails past $\lambda_{\max}=1$ is the Gaussian
quadrature centred at the origin.

**Plainly.** The integration rule assumes the thing it is integrating does not curve upward
faster than the Gaussian falls off. $\lambda_{\max}$ measures that curvature, and it is a
*total budget* across all products: strong interactions everywhere is not affordable, which
is why $\phi$ is restricted to 30 products.

---

## 9. Numerical validity

The per-row polynomial is truncated at degree $d$ (`poly_degree`). This looks like a speed
knob. It is not.

The category factor in (6.2) is $\exp(-\rho_c\binom{r}{2})$. At the fitted
$\rho_c = -0.337$ and $r = 120$:

$$
\binom{120}{2} = 7140,\qquad
0.337 \times 7140 = 2406.2,\qquad
e^{2406.2} = 10^{1045}
$$

against float64's ceiling $\approx 10^{308}$. The untruncated recursion returns `NaN`.

Worse, degrees just below overflow are **finite and meaningless**. Measured on the shipped
checkpoint, using the identity $\sum_j\pi_j = \mathbb{E}[n]$ from (7.1):

| degree | $\sum_j \pi_j$ | verdict |
|---|---|---|
| 26 | 6.47 | correct |
| 32 | 6.49 | correct |
| 40 | 102.41 | garbage |
| 64 | 120.00 | garbage ( $= n_{\max}$, "every product certain") |
| 96, 120 | NaN | overflow |

The safe degree depends on $\rho_c$ and therefore on the **checkpoint**: run404
($\rho_c=-0.211$) is safe to 48; run413 ($\rho_c=-0.337$) only to 32.

**It must be calibrated upward from the data floor.** The floor is the largest per-category
count actually observed — 26 here — which is the smallest degree that gives an observed
basket non-zero probability. Calibrating *downward* against the untruncated polynomial
cannot work: that reference is the overflowing one, and `NaN <= tol` is false, so the search
falls through and returns the worst option. Above the floor there is no accuracy gain on
observed data, only unobserved tail mass, so candidates are bounded to $1.5\times$ the floor.

---

## 10. Exact sampling

The model is usable as an environment because a basket is drawn without any Markov chain.
From (4.3) and (6.3), the joint factorises into a chain walked top-down:

| level | draw | exact? |
|---|---|---|
| 1 | $z$ from its posterior $\propto p(z)f(z)$, by sampling-importance-resampling | **the only inexact step**; consistent as draws grow |
| 2 | size $n \sim e^{-\rho_0(n)}A_n(z)$ | exact — these are the terms $f$ already sums |
| 3 | split of $n$ across rows, walking the convolution (6.3) backwards | exact |
| 4 | which products fill each row's slots, walking (6.1) backwards | exact |

For level 4, having chosen $r$ items from a row of $m$, item $m$ is included with
probability

$$
P(m \in T) \;=\; \frac{w_m\, e_{r-1}^{(m-1)}}{e_r^{(m)}}
\tag{10.1}
$$

then recurse on $(m-1, r - \mathbb{1}\{\text{included}\})$.

**Numerical care.** Levels 3–4 build tables over up to 1,773 products; the raw recursion
underflows. Each row is normalised and its log-scale carried, so (10.1) is evaluated as

$$
\frac{w_m\,\tilde e_{r-1}^{(m-1)}}{\tilde e_r^{(m)}}\;\exp\big(\ell_{m-1}-\ell_m\big)
$$

with $\tilde e$ the normalised table and $\ell$ its accumulated log-scale. Without this the
walk silently returns baskets shorter than the $n$ it drew.

**Correctness criterion.** Mean sampled basket size must equal $\sum_j\pi_j$, which is exact
by (7.1). Measured over 576 baskets: within one standard error (bias $+1.4\%$, se $0.41$).

---

## 11. Price

### 11.1 The functional form

$$
b_j \;\supset\; -\,g_j\Big[\;\bar{\ell} \;+\; \kappa\,\big(\ell_j - \bar{\ell}\big)\Big],
\qquad g_j = \sum_{p} \gamma_{h,p}\beta_{j,p}
\tag{11.1}
$$

$\ell_j$ is product $j$'s log-price deviation and $\bar\ell$ a **reference price level**
averaged over some group. The split is the whole point: a *uniform* price move shifts
$\ell_j$ and $\bar\ell$ together so only $g$ acts; an *idiosyncratic* move shifts
$\ell_j - \bar\ell$ so $g\kappa$ acts.

### 11.2 Aggregate elasticity

Under $\ell_j \mapsto \ell_j + \delta$ for all $j$ (so $\bar\ell \mapsto \bar\ell+\delta$),
(11.1) gives $\partial b_j/\partial\delta = -g_j$. Using (7.2),

$$
\frac{d\,\mathbb{E}[n]}{d\delta}
= \sum_{j}\sum_{k}\frac{\partial \pi_j}{\partial b_k}\frac{\partial b_k}{\partial \delta}
= -\sum_{j,k} \operatorname{Cov}(\mathbb{1}_j,\mathbb{1}_k)\, g_k
\;\overset{g_k \approx \bar g}{=}\; -\bar g\,\operatorname{Var}(n)
$$

since $\sum_{j,k}\operatorname{Cov}(\mathbb{1}_j,\mathbb{1}_k) = \operatorname{Var}(\sum_j \mathbb{1}_j) = \operatorname{Var}(n)$. Hence

$$
\boxed{\;\varepsilon_{\text{agg}} \;=\; \frac{d\log\mathbb{E}[n]}{d\delta}
\;=\; -\,\bar g\;\frac{\operatorname{Var}(n)}{\mathbb{E}[n]}\;}
\tag{11.2}
$$

### 11.3 Own-price elasticity

Under $\ell_j \mapsto \ell_j+\delta$ for one product, with $\bar\ell$ essentially unmoved,
$\partial b_j/\partial\delta = -g_j\kappa$, and since
$\partial\pi_j/\partial b_j = \operatorname{Var}(\mathbb{1}_j) = \pi_j(1-\pi_j)$,

$$
\boxed{\;\varepsilon_{\text{own}} \;=\; \frac{d\log \pi_j}{d\delta}
\;=\; -\,g_j\,\kappa\,(1-\pi_j)\;\approx\; -g_j\kappa\;}
\tag{11.3}
$$

### 11.4 Cross-price elasticity

Raise the price of $n_{\text{riv}}$ rivals sharing target $k$'s reference group of size
$n_{\text{ref}}$. Then $\bar\ell$ moves by $\delta\,n_{\text{riv}}/n_{\text{ref}}$ while
$\ell_k$ does not, so from (11.1),

$$
\Delta b_k = -g_k\big[\Delta\bar\ell + \kappa(0-\Delta\bar\ell)\big]
= g_k(\kappa-1)\,\Delta\bar\ell
$$

$$
\boxed{\;\varepsilon_{\text{cross}} \;=\; g_k(\kappa-1)\,\frac{n_{\text{riv}}}{n_{\text{ref}}}\,(1-\pi_k)\;}
\tag{11.4}
$$

**Substitution enters through the reference, and only there.** At the fitted $\kappa=35.6$
and $g=0.01545$, $g(\kappa-1) = +0.535$ — strongly positive — but it is scaled by
$n_{\text{riv}}/n_{\text{ref}}$. Measured across three reference widths:

| reference | median width | $\varepsilon_{\text{cross}}$ |
|---|---|---|
| whole assortment | 5,312 | $-0.162$ (wrong sign; the size effect dominates) |
| **store's category** | **128** | $\mathbf{+0.044}$ |
| sub-commodity | 16 | $+0.502$ |
| **data target** | — | $\mathbf{+0.099}$ |

### 11.5 Why $\kappa$ is initialised, not trained

The projection pins $\bar g$ so (11.2) matches the aggregate target, leaving $\kappa$ to
carry the own-price response through (11.3). But $\kappa$'s natural scale is $\approx 40$,
so the structural learning rate $0.002$ moves it $0.005\%$ per step, and its gradient is
small and sign-noisy across 24-trip minibatches, which Adam averages to nearly nothing.
**Measured: 1.4 units per 1,000 iterations even at $20\times$ the rate** — some 50,000 extra
iterations to arrive. Where it starts decides where it ends.

It is identified without the model. From (11.2) and (11.3),

$$
\kappa^{*} \;=\; \frac{\lvert\varepsilon_{\text{own}}\rvert}{\bar g},
\qquad
\bar g \;=\; \lvert\varepsilon_{\text{agg}}\rvert\,\frac{\mathbb{E}[n]}{\operatorname{Var}(n)}
\tag{11.5}
$$

With $\varepsilon_{\text{own}}=-0.789$, $\varepsilon_{\text{agg}}=-0.121$,
$\mathbb{E}[n]=7.80$, $\operatorname{Var}(n)=82.8$: $\bar g = 0.0114$ and
$\kappa^{*} = 69.2$. A sweep on the fitted model puts the *likelihood's own* optimum at
$\kappa\in[40,60]$, spanning $\varepsilon_{\text{own}}\in[-1.00,-0.71]$ — so data and
likelihood agree to within the width of that interval.

### 11.6 The targets are external, and rest on one estimator

`src/basket/elasticity_targets.py` estimates them from the data and `train.sh` consumes the
result. Two estimators are run on variation that shares nothing:

| estimator | identification | $\varepsilon_{\text{own}}$ | $n$ |
|---|---|---|---|
| **A** item-week panel, item FE, promo controlled | an item's own price over weeks | $-0.7886$ (se $0.0117$) | 42,464 |
| **B** cross-store within item-week | stores pricing the same item differently in the same week | $-0.1309$ (se $0.0262$) | 2,719 |

**They disagree at 22.9σ.** B is identified off store deviations covering 0.53% of the
item–store–week grid, and demeaning within item-week discards every item-week observed at a
single store — thin, and consistent with attenuation from a noisy regressor. A is adopted.
**The calibration therefore rests on one credible estimator, not two.**

The aggregate estimator returns $+0.285$ — a price rise growing baskets. The store "price
level" is a mean over whichever items happen to have an observed deviation, and observation
is non-random, so this is composition, not price. The script refuses it and falls back to
$-0.121$ with a printed warning.

---

## 12. Why the interaction cannot substitute

It is tempting to fix cross-price with $\phi$. It cannot work, and the reason is exact.

By (7.2), $\partial\pi_k/\partial b_j = \operatorname{Cov}(\mathbb{1}_j,\mathbb{1}_k)$, so
positive cross-price elasticity **requires negative covariance** between the two items. And
a pair term's leverage on a *marginal* is second order in $\pi_j\pi_k \approx 10^{-3}$:

$$
\operatorname{Cov}(\mathbb{1}_j,\mathbb{1}_k)\Big/\pi_k \;=\; P(j\mid k) - P(j),
\qquad \big\lvert P(j\mid k) - P(j)\big\rvert \le P(j)
$$

so the achievable cross elasticity is bounded by $P(j)\cdot g\kappa$, which for a typical
$P(j)\sim 10^{-3}$ is negligible.

**Measured on a hand-built best case** — two popular same-sub-commodity products with
embeddings driven maximally apart:

| $\phi_j^{\top}\phi_k$ | $\lambda_{\max}$ | $\varepsilon_{\text{cross}}$ |
|---|---|---|
| as fitted | — | $-0.0125$ |
| $-0.106$ (full budget) | 0.157 | $-0.0094$ |
| $-0.640$ | 0.216 | $+0.0015$ |

**93× short of the target**, with $\lambda_{\max}$ never exceeding 0.216 — so the budget of
§8 was not even the binding constraint. Raising $K_z$, unmasking $\phi$, or adding a
repulsive $\rho$ at sub-commodity granularity all fail for the same reason.

A repulsive $\rho$ would also contradict the data. On the count margin $\rho$ acts on,
households buy $\ge 2$ items from a sub-commodity **8.6× more often than independence**
(31.7× for affinity categories) — complementary, not repulsive.

**A warning about lift.** Pairwise lift $P(j,k)/(P(j)P(k))$ conflates category *incidence*
with within-category *count*, and restricting to pairs that co-occur at least a few times
conditions **on** co-occurrence — precisely what substitutes do not do. That filter reported
a median same-sub-commodity lift of 86.8 here; over the complete matrix with zeros kept it
is 1.57. Substitution is real but shows up under **price variation** (§11.6), not under
co-occurrence.

---

## 13. Objective and projections

Training maximises the exact set log-likelihood $E(S) - \log Z$ per trip, plus a units
likelihood, a size cross-entropy calibrating $P(n)$, and:

**The elasticity is projected, not penalised.** With Adam the step size is set by the
gradient's sign and running scale, so a penalty with a huge gradient still moves the
parameter by about $\text{lr}$ per step and the likelihood out-pushes it. Measured: with
weight 20 the elasticity went $-0.765 \to -4.871$ over 400 iterations — the wrong way, fast.
The projection is a closed-form step that cannot be out-pushed. Under `--price-soft`,
$\gamma,\beta$ *are* the coefficients, so it is a multiplication
$\gamma,\beta \mapsto \gamma,\beta\sqrt{\tau/\text{cur}}$ (exact to $10^{-16}$), not the
subtraction that is only multiplicative when $\mathrm{softplus}(x)\approx e^x$.

**$\rho_c$ is floored** at $-0.92$: a $2.5\times$ pair lift is $\rho_c = -0.92$, and §9
shows the term detonates below that.

**$\phi = 0$ is a saddle.** From (4.1),

$$
\frac{\partial E(S)}{\partial \phi_j} \;=\; \sum_{k\in S,\,k\neq j}\phi_k
$$

which vanishes when every $\phi$ is zero. Escape from a zero initialisation is therefore
exponential, which is why $\phi$ is placed **spectrally** — by eigendecomposing the
empirical log-lift matrix, since $\log\text{lift}_{jk} \approx \phi_j^{\top}\phi_k$ — rather
than seeded with noise. Measured: spectral placement beat 15,000 SGD updates from a noise
seed by 0.022 nats.

**Parameter scales differ by $400\times$.** Unconstrained, $\gamma \sim 0.02$ while
$\kappa \sim 40$. One learning rate cannot serve both: at $\text{lr}=0.002$, $\gamma$ moves
10% of its own value per step (51× the constrained effective step,
$0.002\cdot\sigma(-3.92) = 3.9\times10^{-5}$) and diverges, while $\kappa$ moves 0.005%.
Hence separate optimiser groups — $0.05\times$ for price, $5$–$20\times$ for $\kappa$.
