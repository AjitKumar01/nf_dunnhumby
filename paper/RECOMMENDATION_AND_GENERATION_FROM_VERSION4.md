# Recommendation and basket generation from the Version-4 probability law

- Status: **theory note for presentation and audit**
- Scope: **distinct-product incidence baskets, conditional on a non-empty trip**
- Foundation: **the unchanged Version-4 energy and joint basket probability**

This document derives two separate functions of the fitted Version-4 model:

1. basket-completion recommendation; and
2. unconditional non-empty basket generation.

Both begin from the same joint basket probability. Basket generation does **not** follow
from the recommendation softmax, and it does not repeatedly use recommendation scores.
The recommendation calculation is one conditional query of the model. The generator is a
direct sampler for the complete joint basket law.

The note also explains why contextual utility uses a change in log price. No new energy
term, size factor, interaction law, or recommendation-specific training objective is
introduced.

---

## 1. Objects and support

Let:

- $x=(h,t,s,\ldots)$ be a purchase context, including household, date, store, prices,
  promotions, and other declared covariates;
- $\mathcal A_x$ be the offered assortment in context $x$;
- $J_x=|\mathcal A_x|$, with up to 5,455 products in the present catalogue;
- $S\subseteq\mathcal A_x$ be an unordered set of distinct purchased products;
- $N=|S|$ be total basket size;
- $c(j)$ be the category of product $j$;
- $N_c(S)=|\{j\in S:c(j)=c\}|$ be the count from category $c$;
- $\phi_j\in\mathbb R^r$ be the interaction embedding of product $j$; and
- $n_{\max}=120$ be the declared maximum supported basket size.

The incidence support is

\[
\Omega_x^+
=
\left\{
S\subseteq\mathcal A_x:
1\le |S|\le n_{\max}
\right\}.
\tag{1}
\]

The superscript $+$ records that the empty basket is excluded. The model therefore means

\[
p_\Theta(S\mid x,S\neq\varnothing),
\tag{2}
\]

although the non-empty conditioning is suppressed below.

This distinction matters in production. The fitted model answers:

> Given that a recorded purchase occasion occurred, what non-empty basket was purchased?

It does not estimate whether a household visits the retailer or buys nothing. Section 13
shows how a separately identified purchase-occurrence probability can be composed with
the Version-4 conditional basket law.

The present model is also an incidence model: it generates distinct SKUs, not unit
quantities. Unit, revenue, and profit simulation require a separately fitted conditional
quantity and margin layer.

---

## 2. The Version-4 energy and joint probability

For $S\in\Omega_x^+$, the basket energy is

\[
\boxed{
\begin{aligned}
E_\Theta(S;x)
={}&
\sum_{j\in S}b_j(x)
+\sum_{\substack{j<k\\j,k\in S}}\phi_j^\top\phi_k\\
&-\sum_c\rho_c{N_c(S)\choose2}
-\rho_0(|S|).
\end{aligned}
}
\tag{3}
\]

The terms have different roles:

- $b_j(x)$ is contextual product utility;
- $\phi_j^\top\phi_k$ is the learned pair interaction;
- $\rho_c{N_c\choose2}$ controls broad within-category crowding or attraction; and
- $\rho_0(N)$ is the total basket-size potential.

The complete Version-4 basket probability is

\[
\boxed{
p_\Theta(S\mid x)
=
\frac{\exp\{E_\Theta(S;x)\}}{Z_+(x)}
},
\qquad
Z_+(x)
=
\sum_{T\in\Omega_x^+}\exp\{E_\Theta(T;x)\}.
\tag{4}
\]

Likelihood, recommendation, marginal incidence, counterfactuals, and generation are
different mathematical queries of Equation (4).

---

## 3. Contextual utility and log price

The Version-4 item utility is

\[
\begin{aligned}
b_{jht}
={}&
\lambda_j
+\theta_h^\top\alpha_j
-g_{hj}\,\Delta\log\operatorname{Price}_{jt}\\
&+w_j^{\mathrm{dsp}}D_{jt}
+w_j^{\mathrm{mlr}}M_{jt}
+\mu_j^\top\delta_{w(t)}
+\zeta_j^\top\xi_{s(t)}
+\psi_j^\top r_{jht},
\end{aligned}
\tag{5}
\]

where

\[
g_{hj}
=
\operatorname{softplus}(\gamma_h)^\top
\operatorname{softplus}(\beta_j)
\ge 0.
\tag{6}
\]

Here $\operatorname{Price}_{jt}$ is a monetary price, not a probability. The contextual
price feature is

\[
\Delta\log\operatorname{Price}_{jt}
=
\log\left(
\frac{\operatorname{Price}_{jt}}
{\operatorname{Price}^{\mathrm{ref}}_j}
\right).
\tag{7}
\]

Using explicit $\operatorname{Price}$ notation in slides prevents the symbol $p$ from
being mistaken for a probability.

### 3.1 Why use a logarithm?

First, log change measures a proportional price movement. A move from 2 to 2.20 and a move
from 20 to 22 are both 10% increases:

\[
\log(2.20/2)=\log(22/20)=\log(1.1).
\tag{8}
\]

Second, it is invariant to the currency unit. For every $a>0$,

\[
\log\left(
\frac{a\operatorname{Price}_{jt}}
{a\operatorname{Price}^{\mathrm{ref}}_j}
\right)
=
\log\left(
\frac{\operatorname{Price}_{jt}}
{\operatorname{Price}^{\mathrm{ref}}_j}
\right).
\tag{9}
\]

Third, for a small relative price change,

\[
\Delta\log\operatorname{Price}
\approx
\frac{\Delta\operatorname{Price}}{\operatorname{Price}}.
\tag{10}
\]

The coefficient $g_{hj}$ therefore has an elasticity-like meaning on the energy scale.

### Proposition 1 — direct product utility is non-increasing in own price

Holding the other contextual variables fixed,

\[
\frac{\partial b_{jht}}
{\partial\,\Delta\log\operatorname{Price}_{jt}}
=-g_{hj}\le0.
\tag{11}
\]

#### Proof

Equation (5) contains the price feature through
$-g_{hj}\Delta\log\operatorname{Price}_{jt}$. Equation (6) makes $g_{hj}$ nonnegative.
Differentiation gives Equation (11). $\square$

For a discount fraction $d\in[0,1)$,

\[
\operatorname{Price}'=(1-d)\operatorname{Price}
\tag{12}
\]

and hence

\[
\log\operatorname{Price}'-\log\operatorname{Price}
=\log(1-d)<0.
\tag{13}
\]

The direct utility change is

\[
\Delta b_j=-g_{hj}\log(1-d)\ge0.
\tag{14}
\]

This proves the sign of the direct utility response. The final marginal purchase
probability also depends on basket size, category competition, interactions, and all other
products, so it is not a single-product logit elasticity.

---

# Part I — basket-completion recommendation

## 4. The precise conditional question

Suppose an observed basket has one product hidden. Let $R$ be the revealed remainder. The
evaluation asks:

> Conditional on the final basket containing every product in $R$ and containing exactly
> one additional product, which product is the addition?

Define the one-item-completion event

\[
\mathcal C_R
=
\left\{
S=R\cup\{k\}:
k\in\mathcal A_x\setminus R
\right\}.
\tag{R.1}
\]

For candidate $j\notin R$, the exact probability being evaluated is

\[
P_\Theta\left(
S=R\cup\{j\}
\mid
S\in\mathcal C_R,x
\right).
\tag{R.2}
\]

This is not the unconditional incidence probability $P(j\in S\mid x)$ and it is not an
unconditional next-item transition.

## 5. Deriving the recommendation softmax

Define the add-one energy increment

\[
s_j(R,x)
=
E_\Theta(R\cup\{j\};x)-E_\Theta(R;x).
\tag{R.3}
\]

### Proposition 2 — exact one-item-completion probability

\[
\boxed{
P_\Theta\left(
S=R\cup\{j\}
\mid S\in\mathcal C_R,x
\right)
=
\frac{e^{s_j(R,x)}}
{\displaystyle\sum_{k\in\mathcal A_x\setminus R}e^{s_k(R,x)}}
}.
\tag{R.4}
\]

#### Proof directly from the Version-4 probability

Conditional probability gives

\[
P_\Theta(S=R\cup\{j\}\mid S\in\mathcal C_R,x)
=
\frac{
P_\Theta(S=R\cup\{j\}\mid x)
}{
\displaystyle\sum_{k\notin R}
P_\Theta(S=R\cup\{k\}\mid x)
}.
\tag{R.5}
\]

Substitute Equation (4):

\[
\begin{aligned}
&P_\Theta(S=R\cup\{j\}\mid S\in\mathcal C_R,x)\\
&\quad=
\frac{
\exp\{E_\Theta(R\cup\{j\};x)\}/Z_+(x)
}{
\displaystyle\sum_{k\notin R}
\exp\{E_\Theta(R\cup\{k\};x)\}/Z_+(x)
}\\
&\quad=
\frac{
\exp\{E_\Theta(R\cup\{j\};x)\}
}{
\displaystyle\sum_{k\notin R}
\exp\{E_\Theta(R\cup\{k\};x)\}
}.
\end{aligned}
\tag{R.6}
\]

The original partition function cancels because every candidate is compared under the
same context and support. By Equation (R.3),

\[
E_\Theta(R\cup\{j\};x)
=
E_\Theta(R;x)+s_j(R,x).
\tag{R.7}
\]

The common factor $e^{E_\Theta(R;x)}$ also cancels:

\[
\frac{
e^{E_\Theta(R;x)}e^{s_j(R,x)}
}{
\sum_{k\notin R}e^{E_\Theta(R;x)}e^{s_k(R,x)}
}
=
\frac{e^{s_j(R,x)}}{\sum_{k\notin R}e^{s_k(R,x)}}.
\tag{R.8}
\]

This proves Equation (R.4). $\square$

No numerical normalizer, quadrature rule, Monte Carlo sample, or separate recommender is
needed for this particular conditional probability.

## 6. Expanding the add-one score

Adding candidate $j$ changes the pair interaction by

\[
\sum_{k\in R}\phi_j^\top\phi_k.
\tag{R.9}
\]

Its category count changes from $N_{c(j)}(R)$ to $N_{c(j)}(R)+1$, and

\[
{N_{c(j)}(R)+1\choose2}
-{N_{c(j)}(R)\choose2}
=N_{c(j)}(R).
\tag{R.10}
\]

The total-size potential changes by

\[
\rho_0(|R|+1)-\rho_0(|R|).
\tag{R.11}
\]

Therefore

\[
\boxed{
\begin{aligned}
s_j(R,x)
={}&
b_j(x)
+\sum_{k\in R}\phi_j^\top\phi_k
-\rho_{c(j)}N_{c(j)}(R)\\
&-\left[
\rho_0(|R|+1)-\rho_0(|R|)
\right].
\end{aligned}
}
\tag{R.12}
\]

All candidates produce a basket of the same size $|R|+1$. The size increment in the last
line is therefore common to every candidate and cancels from their relative ranking.
Using

\[
m_R=\sum_{k\in R}\phi_k,
\tag{R.13}
\]

the ranking score is

\[
\boxed{
s_j^{\mathrm{rank}}(R,x)
=
b_j(x)
+\phi_j^\top m_R
-\rho_{c(j)}N_{c(j)}(R).
}
\tag{R.14}
\]

This separates the recommendation signal into:

- contextual relevance through $b_j(x)$;
- complementarity with the revealed basket through $\phi_j^\top m_R$; and
- within-category competition or attraction through $\rho_{c(j)}N_{c(j)}(R)$.

### 6.1 Completion probability versus marginal incidence

The marginal incidence probability is

\[
\pi_j(x)
=P_\Theta(j\in S\mid x)
=\frac{\partial\log Z_+(x)}{\partial b_j(x)}.
\tag{R.15}
\]

It averages over all supported baskets. Equation (R.4), by contrast, conditions on a
specific $R$ and exactly one missing item. The correct score depends on the question:

- use $\pi_j(x)$ for an unconditional trip-level ranking;
- use $s_j(R,x)$ for locked one-item basket completion.

### 6.2 MRR

Let $r_i$ be the rank of the hidden true product in evaluation case $i$. Then

\[
\operatorname{MRR}
=
\frac1M\sum_{i=1}^M\frac1{r_i},
\tag{R.16}
\]

and

\[
\operatorname{MRR@K}
=
\frac1M\sum_{i=1}^M
\frac{\mathbf 1\{r_i\le K\}}{r_i}.
\tag{R.17}
\]

MRR measures ranking, not probability calibration. Joint likelihood can improve through
size, category, or tail calibration without changing the top candidate ranks. There is no
theorem requiring MRR to increase monotonically with joint likelihood.

This completes the recommendation derivation.

---

# Part II — basket generation directly from the joint law

## 7. The generation target

Generation starts again from the original Version-4 probability:

\[
\boxed{
S\sim p_\Theta(S\mid x)
=
\frac{\exp\{E_\Theta(S;x)\}}{Z_+(x)},
\qquad S\in\Omega_x^+.
}
\tag{G.1}
\]

No equation from Part I is needed. In particular, an exact generator does not repeatedly
sample from the recommendation softmax.

A naive add-one procedure would be wrong because Version-4 assigns probability to a final
unordered set, not to an item ordering. It would also omit the probability of stopping
and the energy mass of all valid future completions. The generator below instead
factorizes Equation (G.1) itself.

## 8. Hubbard--Stratonovich augmentation

The pair interaction satisfies

\[
\sum_{\substack{j<k\\j,k\in S}}\phi_j^\top\phi_k
=
\frac12\left\|\sum_{j\in S}\phi_j\right\|^2
-\frac12\sum_{j\in S}\|\phi_j\|^2.
\tag{G.2}
\]

Define

\[
\widetilde b_j(x)
=
b_j(x)-\frac12\|\phi_j\|^2
\tag{G.3}
\]

and, for $z\in\mathbb R^r$,

\[
w_j(z;x)
=
\exp\left\{
\widetilde b_j(x)+z^\top\phi_j
\right\}.
\tag{G.4}
\]

The Gaussian identity

\[
\mathbb E_{z\sim\mathcal N(0,I_r)}
\left[e^{z^\top v}\right]
=
e^{\|v\|^2/2}
\tag{G.5}
\]

makes the quadratic interaction additive conditional on $z$.

### Proposition 3 — positive augmentation with the exact basket marginal

Let $\varphi_r(z)$ be the standard $r$-dimensional Gaussian density. Define

\[
\boxed{
\begin{aligned}
p_\Theta(S,z\mid x)
=
\frac{\varphi_r(z)}{Z_+(x)}
\exp\Bigg\{
&\sum_{j\in S}
\left[
b_j(x)-\frac12\|\phi_j\|^2+z^\top\phi_j
\right]\\
&-\sum_c\rho_c{N_c(S)\choose2}
-\rho_0(|S|)
\Bigg\}.
\end{aligned}
}
\tag{G.6}
\]

Then

\[
\int_{\mathbb R^r}p_\Theta(S,z\mid x)\,dz
=p_\Theta(S\mid x).
\tag{G.7}
\]

#### Proof

Let

\[
v_S=\sum_{j\in S}\phi_j.
\tag{G.8}
\]

The $z$-dependent integral in Equation (G.6) is

\[
\begin{aligned}
&\int\varphi_r(z)
\exp\left\{
z^\top v_S-\frac12\sum_{j\in S}\|\phi_j\|^2
\right\}dz\\
&\quad=
\exp\left\{
\frac12\|v_S\|^2
-\frac12\sum_{j\in S}\|\phi_j\|^2
\right\}\\
&\quad=
\exp\left\{
\sum_{\substack{j<k\\j,k\in S}}
\phi_j^\top\phi_k
\right\},
\end{aligned}
\tag{G.9}
\]

where the first equality uses Equation (G.5), and the second uses Equation (G.2). Every
remaining exponent in Equation (G.6) is the corresponding term in the original energy
(3). Dividing by the same $Z_+(x)$ therefore gives Equation (G.7). $\square$

The augmentation is exact. The vector $z$ is an auxiliary numerical variable; it need
not have a retailer interpretation.

## 9. Exact conditional mass at fixed auxiliary state

At fixed $z$, define the category ESP

\[
e_{c,k}(z,x)
=
\sum_{\substack{A\subseteq\mathcal A_{xc}\\|A|=k}}
\prod_{j\in A}w_j(z;x),
\qquad e_{c,0}=1.
\tag{G.10}
\]

Attach the Version-4 category energy:

\[
q_c(k;z,x)
=
e^{-\rho_c{k\choose2}}e_{c,k}(z,x).
\tag{G.11}
\]

Define the category polynomial

\[
G_c(u;z,x)
=
\sum_{k=0}^{\min(|\mathcal A_{xc}|,n_{\max})}
q_c(k;z,x)u^k.
\tag{G.12}
\]

Multiplying the category polynomials gives

\[
\prod_cG_c(u;z,x)
=
\sum_{n=0}^{n_{\max}}A_n(z,x)u^n.
\tag{G.13}
\]

The positive non-empty mass conditional on $z$ is

\[
F_+(z;x)
=
\sum_{n=1}^{n_{\max}}
e^{-\rho_0(n)}A_n(z,x).
\tag{G.14}
\]

### Proposition 4 — $A_n$ enumerates all size-$n$ basket mass at fixed $z$

\[
A_n(z,x)
=
\sum_{\substack{S\subseteq\mathcal A_x\\|S|=n}}
\left[
\prod_c e^{-\rho_c{N_c(S)\choose2}}
\right]
\left[
\prod_{j\in S}w_j(z;x)
\right].
\tag{G.15}
\]

#### Proof

Choosing degree $k$ from category polynomial $G_c$ chooses $k$ distinct products from
category $c$. The ESP sums the product weights of all such subsets, and the multiplier
$e^{-\rho_c{k\choose2}}$ attaches the original category energy. Multiplication over
categories combines disjoint category subsets. The power of $u$ records the total number
of selected products. Therefore the coefficient of $u^n$ contains every size-$n$ basket
exactly once, proving Equation (G.15). $\square$

It follows from Equations (G.6) and (G.14) that

\[
\boxed{
p_\Theta(z\mid x)
=
\frac{\varphi_r(z)F_+(z;x)}{Z_+(x)}
}
\tag{G.16}
\]

and

\[
\boxed{
p_\Theta(S\mid z,x)
=
\frac{
e^{-\rho_0(|S|)}
\prod_c e^{-\rho_c{N_c(S)\choose2}}
\prod_{j\in S}w_j(z;x)
}{
F_+(z;x)
}.
}
\tag{G.17}
\]

Equations (G.16)--(G.17) are a direct factorization of the Version-4 basket probability:

\[
p_\Theta(S\mid x)
=
\int
p_\Theta(z\mid x)\,
p_\Theta(S\mid z,x)\,dz.
\tag{G.18}
\]

## 10. Exact discrete basket draw conditional on $z$

Assume temporarily that $z\sim p_\Theta(z\mid x)$ is available. The conditional law
(G.17) can be sampled in three exact stages.

### Stage 1 — total basket size

\[
\boxed{
P_\Theta(N=n\mid z,x)
=
\frac{
e^{-\rho_0(n)}A_n(z,x)
}{
F_+(z;x)
},
\qquad 1\le n\le n_{\max}.
}
\tag{G.19}
\]

This is where the basket-size potential directly controls generation. In recommendation,
all candidates had the same final size and its increment cancelled. In unconditional
generation, size is unknown and must be drawn from Equation (G.19).

### Stage 2 — category counts

Conditional on $N=n$, the count vector $r=(r_1,\ldots,r_C)$ satisfies

\[
\boxed{
P_\Theta((N_c)_c=r\mid N=n,z,x)
=
\frac{
\mathbf 1\{\sum_cr_c=n\}
\prod_c q_c(r_c;z,x)
}{
A_n(z,x)
}.
}
\tag{G.20}
\]

It is unnecessary to enumerate all count vectors. Define suffix masses

\[
H_c(m)
=
[u^m]\prod_{d=c}^{C}G_d(u;z,x),
\tag{G.21}
\]

with $H_{C+1}(0)=1$ and $H_{C+1}(m)=0$ for $m\neq0$. If $m$ products remain to be
allocated at category $c$, draw $N_c=k$ with

\[
\boxed{
P(N_c=k\mid m,z,x)
=
\frac{
q_c(k;z,x)H_{c+1}(m-k)
}{
H_c(m)
}.
}
\tag{G.22}
\]

The numerator is the total mass of every continuation that selects $k$ products from the
current category and $m-k$ products from later categories. Summing over feasible $k$
recovers $H_c(m)$, so Equation (G.22) is normalized. Multiplying the successive
backtracking probabilities telescopes to Equation (G.20).

### Stage 3 — products within each category

Given $N_c=k$, a subset $B_c\subseteq\mathcal A_{xc}$ with $|B_c|=k$ has probability

\[
\boxed{
P_\Theta(B_c\mid |B_c|=k,z,x)
=
\frac{
\prod_{j\in B_c}w_j(z;x)
}{
e_{c,k}(z,x)
}.
}
\tag{G.23}
\]

One exact method uses suffix ESP values. Index the category products by
$1,\ldots,M_c$. If $k$ selections remain before product $i$, then

\[
P(i\text{ included}\mid k\text{ remain},z,x)
=
\frac{
w_i(z;x)e_{k-1}(w_{i+1},\ldots,w_{M_c})
}{
e_k(w_i,\ldots,w_{M_c})
}.
\tag{G.24}
\]

If product $i$ is included, reduce $k$ by one. Otherwise leave it unchanged. The ESP
identity

\[
e_k(w_i,\ldots,w_M)
=
e_k(w_{i+1},\ldots,w_M)
+w_i e_{k-1}(w_{i+1},\ldots,w_M)
\tag{G.25}
\]

shows that the inclusion and exclusion probabilities sum to one. Their product along the
chosen path telescopes to Equation (G.23).

An equivalent exact implementation uses independent Bernoulli variables

\[
Q_j
=
\frac{e^{\eta_j+a}}{1+e^{\eta_j+a}},
\qquad
\eta_j=\log w_j,
\tag{G.26}
\]

and conditions on exactly $k$ successes. For a size-$k$ set $B_c$,

\[
\prod_{j\in B_c}Q_j
\prod_{j\notin B_c}(1-Q_j)
\propto
\prod_{j\in B_c}e^{\eta_j}.
\tag{G.27}
\]

The scalar shift $a$ cancels after conditioning and may be chosen to improve acceptance.
If a rejection limit is used, exhausting it must raise an explicit failure or switch to
exact suffix-ESP sampling; silently skipping the context would change the target data
distribution.

Finally return

\[
S=\bigcup_cB_c.
\tag{G.28}
\]

### Proposition 5 — the three stages reproduce $p_\Theta(S\mid z,x)$

Let a particular basket $S$ have size $n$, category counts $r_c=N_c(S)$, and category
subsets $B_c=S\cap\mathcal A_{xc}$. Multiplying Equations (G.19), (G.20), and (G.23) gives

\[
\begin{aligned}
&\frac{e^{-\rho_0(n)}A_n}{F_+}
\times
\frac{\prod_c[
e^{-\rho_c{r_c\choose2}}e_{c,r_c}
]}{A_n}
\times
\prod_c
\frac{\prod_{j\in B_c}w_j}{e_{c,r_c}}\\
&\qquad=
\frac{
e^{-\rho_0(n)}
\prod_c e^{-\rho_c{r_c\choose2}}
\prod_{j\in S}w_j
}{
F_+
},
\end{aligned}
\tag{G.29}
\]

which is exactly Equation (G.17). $\square$

All discrete stages after $z$ are therefore exact and retain all 5,455 products and every
declared size $1,\ldots,120$. There is no 20-product support restriction.

## 11. Sampling the outer interaction distribution

The remaining target is Equation (G.16):

\[
p_\Theta(z\mid x)
\propto
\varphi_r(z)F_+(z;x).
\tag{G.30}
\]

It is generally not Gaussian. Smolyak quadrature can accurately integrate functions
against this target, but a Smolyak combination can contain negative weights. Signed nodes
must not be interpreted as sampling probabilities.

The positive generation method used by the selected theory tempers only the interaction
already present in Equation (3). Define

\[
p_\beta(S\mid x)
\propto
\exp\{
E_0(S;x)+\beta V_\Phi(S)
\},
\qquad 0\le\beta\le1,
\tag{G.31}
\]

where

\[
E_0(S;x)
=
\sum_{j\in S}b_j(x)
-\sum_c\rho_c{N_c(S)\choose2}
-\rho_0(|S|)
\tag{G.32}
\]

and

\[
V_\Phi(S)
=
\sum_{\substack{j<k\\j,k\in S}}
\phi_j^\top\phi_k.
\tag{G.33}
\]

At $\beta=0$, contextual utilities, categories, the complete size law, and full catalogue
support remain. Only the Gram interaction is absent. This base basket law is sampled
exactly with the same size/category/product dynamic program using weights $e^{b_j(x)}$.
At $\beta=1$, Equation (G.31) is exactly the full Version-4 law.

Choose

\[
0=\beta_0<\beta_1<\cdots<\beta_L=1.
\tag{G.34}
\]

For each particle basket, the incremental positive weight is

\[
W_\ell(S)
=
\exp\{
(\beta_\ell-\beta_{\ell-1})V_\Phi(S)
\}.
\tag{G.35}
\]

After normalization and resampling, apply a blocked Gibbs mutation. Let

\[
m(S)=\sum_{j\in S}\phi_j.
\tag{G.36}
\]

Draw

\[
z\mid S,x,\beta_\ell
\sim
\mathcal N\left(
\sqrt{\beta_\ell}\,m(S),I_r
\right),
\tag{G.37}
\]

then draw a new basket from Stages 1--3 using

\[
w_j(z,\beta_\ell,x)
=
\exp\left\{
b_j(x)
-\frac{\beta_\ell}{2}\|\phi_j\|^2
+\sqrt{\beta_\ell}\,z^\top\phi_j
\right\}.
\tag{G.38}
\]

### Proposition 6 — the blocked mutation preserves the bridge distribution

The transition $S\rightarrow z\rightarrow S'$ in Equations (G.37)--(G.38) leaves
$p_{\beta_\ell}(S\mid x)$ invariant.

#### Proof

At fixed $\beta$, define the positive augmented density proportional to

\[
\exp\left\{
E_0(S;x)
-\frac12\|z\|^2
+\sqrt\beta\,z^\top m(S)
-\frac\beta2\sum_{j\in S}\|\phi_j\|^2
\right\}.
\tag{G.39}
\]

Completing the square in $z$ gives the Gaussian conditional in Equation (G.37). Holding
$z$ fixed gives the additive product weights in Equation (G.38), for which
Stages 1--3 are exact. Alternating exact full conditional draws is a Gibbs kernel, so it
preserves the augmented distribution and its basket marginal. Integrating $z$ using the
same calculation as Equation (G.9) gives
$\exp\{E_0(S;x)+\beta V_\Phi(S)\}$, which is the bridge target. $\square$

At finite particle count, the SMC output is not an independent exact sample because
resampling creates shared ancestry. Under standard positivity, finite-moment, and
ergodicity conditions, its empirical basket distribution is consistent as particle count
increases. A final $\beta=1$ Gibbs rejuvenation improves diversity without changing the
target.

Effective sample size detects concentration among represented particles:

\[
\frac{\operatorname{ESS}_\ell}{P}
=
\frac{1}{
P\sum_{p=1}^P(\overline W_\ell^{(p)})^2
}.
\tag{G.40}
\]

A small value proves degeneracy. A large value does not prove that an unvisited mode is
absent. Independent replicates, schedule comparisons, tail checks, and held-out
calibration remain necessary.

## 12. Complete generation algorithm

For a declared purchase context $x$:

1. Construct the offered assortment and every feature used in $b_j(x)$.
2. Load the fitted $\phi_j$, $\rho_c$, and $\rho_0$ without changing the energy.
3. Compute the $\beta=0$ ESP/category dynamic program.
4. Draw exact non-interaction base baskets using total size, category counts, and
   within-category product draws.
5. Traverse the fixed bridge schedule in Equation (G.34).
6. At each bridge, apply the positive weight in Equation (G.35), resample, and perform the
   invariant blocked mutation in Equations (G.37)--(G.38).
7. Apply final full-interaction rejuvenation at $\beta=1$.
8. Return the final distinct-SKU baskets and all particle diagnostics.
9. If unit quantities are required, use a separately trained and certified conditional
   quantity model after incidence generation.

Inside every conditional basket draw, the exact discrete factorization is

\[
z
\longrightarrow
N
\longrightarrow
(N_c)_c
\longrightarrow
(B_c)_c
\longrightarrow
S.
\tag{G.41}
\]

This chain is derived from Equations (G.6)--(G.18), not from recommendation.

### 12.1 Computational cost

For a revealed basket $R$, recommendation precomputes $m_R$ and scores all products in
$O(J_xr)$ interaction arithmetic. It needs no outer estimator.

At a fixed $z$, a conservative bound for one complete conditional generation table is

\[
O(J_xn_{\max}+C_xn_{\max}^2),
\tag{G.42}
\]

where $C_x$ is the number of nonempty categories. Degree caps, sparse categories, and
degree-aware polynomial multiplication reduce realized work.

With $P$ particles and $L$ bridges, the outer SMC repeats conditional table work. Computing
the interaction statistic itself costs

\[
O(PL\,\bar n r),
\tag{G.43}
\]

where $\bar n$ is average basket size. No step enumerates the $2^{J_x}$ possible subsets.
Vectorizing particles and reusing context-only terms changes runtime, not the probability
law.

---

## 13. Incorporating no-purchase probability

Because Version-4 is fitted to observed non-empty trips, it cannot identify the probability
of no purchase from those records alone.

Suppose a separate opportunity model, trained with valid exposure denominators, estimates

\[
q(x)=P(B=1\mid x),
\tag{O.1}
\]

where $B=1$ denotes that a purchase occurs. A complete occurrence-plus-basket law is

\[
P(S=\varnothing\mid x)=1-q(x),
\tag{O.2}
\]

\[
P(S=T\mid x)
=
q(x)\,
p_\Theta(T\mid x,S\neq\varnothing),
\qquad T\in\Omega_x^+.
\tag{O.3}
\]

Production sampling then has two levels:

1. draw $B\sim\operatorname{Bernoulli}(q(x))$;
2. return the empty basket if $B=0$;
3. otherwise run the Version-4 generator in Section 12.

Estimating $q(x)$ requires a defensible definition of purchase opportunities, such as
household-week exposure, store visits, or loyalty-app sessions. Purchase-only transactions
do not supply the denominator.

---

## 14. Counterfactual basket generation

Let action $a$ change prices, promotions, or another declared contextual variable. It
creates context $x_a$ and energy $E_\Theta(S;x_a)$. The counterfactual basket law is still
Version-4:

\[
p_\Theta(S\mid x_a)
=
\frac{\exp\{E_\Theta(S;x_a)\}}{Z_+(x_a)}.
\tag{C.1}
\]

The direct procedure is:

1. recompute every affected contextual utility $b_j(x_a)$;
2. preserve the fitted structural parameters unless the intervention definition changes
   them;
3. run the same sampler from Section 12 under $x_a$; and
4. compare factual and counterfactual basket functionals on the same context population.

For reuse of factual particles, define

\[
\Delta_a(S;x)
=
E_\Theta(S;x_a)-E_\Theta(S;x).
\tag{C.2}
\]

If only item utilities change,

\[
\Delta_a(S;x)
=
\sum_{j\in S}
\left[
b_j(x_a)-b_j(x)
\right].
\tag{C.3}
\]

For any basket statistic $f$,

\[
\mathbb E_{x_a}[f(S)]
=
\frac{
\mathbb E_x[
f(S)e^{\Delta_a(S;x)}
]
}{
\mathbb E_x[
e^{\Delta_a(S;x)}
]
}.
\tag{C.4}
\]

This exact density-ratio identity permits efficient reweighting for modest actions with
good overlap. When action-weight ESS is poor, the model should be sampled again under
$x_a$ rather than relying on concentrated or clipped weights.

Equation (C.4) is a probability identity, not causal identification. A causal price or
promotion claim additionally needs randomization, exogenous variation, or another credible
causal design.

---

## 15. Validation obligations

Different uses of Equation (4) need different numerical checks.

### 15.1 Likelihood

- Score identical held-out baskets with identical catalogue support.
- Audit the outer integral at increasing quadrature fidelity.
- Report paired uncertainty across the same baskets.
- Do not interpret a quadrature rule as a generator.

### 15.2 Recommendation

- Lock the hidden-item protocol and candidate assortment.
- Use the exact add-one score from Equation (R.14).
- Report MRR, MRR@$K$, Recall@$K$, and paired model differences.
- Compare the full model with its additive ablation to isolate interaction contribution.

### 15.3 Generation

Generated and observed baskets must be compared under the same context mixture. At minimum,
audit:

- mean, variance, quantiles, and tails of basket size;
- category-count distributions;
- product incidence probabilities;
- pair co-incidence and learned-complement statistics;
- household- or segment-conditional moments;
- bridge ESS and particle ancestry;
- independent replicate stability;
- support violations and sampling failures; and
- counterfactual overlap for every candidate action.

A good held-out likelihood does not automatically certify the finite-particle sampler.
Conversely, a sampler can reproduce the fitted model accurately while the model remains
miscalibrated against held-out data. Numerical sampling error and statistical model error
must be reported separately.

---

## 16. Final logical map

The theory has two independent branches after the common Version-4 law:

\[
\boxed{
p_\Theta(S\mid x)
=
\frac{e^{E_\Theta(S;x)}}{Z_+(x)}
}
\tag{L.1}
\]

leads to recommendation by conditioning on the one-item-completion event:

\[
p_\Theta(S=R\cup\{j\}\mid S\in\mathcal C_R,x)
=
\frac{e^{s_j(R,x)}}{\sum_{k\notin R}e^{s_k(R,x)}},
\tag{L.2}
\]

while the same starting law independently leads to generation through

\[
p_\Theta(S\mid x)
=
\int p_\Theta(z\mid x)
\,
p_\Theta(N\mid z,x)
\,
p_\Theta((N_c)_c\mid N,z,x)
\,
\prod_c p_\Theta(B_c\mid N_c,z,x)
\,dz.
\tag{L.3}
\]

Equation (L.2) is an analytical conditional probability. Equation (L.3) is the direct
generative factorization of the full basket probability. Neither is an approximation to
the other, and the basket sampler never uses the recommendation softmax.

The foundational Version-4 theory therefore remains intact. The H--S identity and ESP
dynamic program make its joint normalization and conditional draws tractable; positive
outer sampling handles generation; and signed deterministic quadrature remains confined
to integration and high-precision likelihood evaluation.
