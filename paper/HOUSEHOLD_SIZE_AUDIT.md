# Conditional basket-size audit and rank-one household correction

Status: **theory checked; constrained pilot passed; fresh converged fit not yet run**

## 1. Question

The corrected rank-5 candidate passes aggregate tail calibration but assigns majority
probability to \(N\ge60\) in a small number of ordinary contexts. This audit asks:

1. Is \(\rho_0\) too small a parameterization?
2. Does the error originate in the additive parent or the Gram interaction?
3. Can an explicitly identified low-rank term inside \(b_{jv}\) learn
   household-specific basket size?
4. Does that change the Version-4 normalizer or sampler?

No foundational energy term is replaced in this branch.

## 2. How many parameters \(\rho_0\) has

The implementation stores

\[
\rho_0(1),\ldots,\rho_0(120)
\]

as 120 trainable scalar values and fixes \(\rho_0(0)=0\). Because the likelihood is
conditioned on a non-empty basket, adding one constant to every
\(\rho_0(1),\ldots,\rho_0(120)\) cancels between numerator and normalizer. Thus there are
at most 119 effective non-empty-support degrees of freedom after gauge fixing.

This is already a saturated size curve. The problem is not too few size parameters.
Rather, one context-independent curve cannot determine which household/occasion should
occupy its rare tail.

The fitted tail is also rough. For \(n\ge40\), the root-mean-square discrete second
difference of the additive parent is \(0.2612\), with maximum magnitude \(0.6613\).
Individual context size laws consequently show roughly 30 small local modes, including
peaks at rare observed sizes such as 93, 101, 103 and 107. Simple post-hoc smoothing
removes those small modes, but does not remove the majority-tail phase without losing
substantial likelihood. Tail roughness is therefore a secondary regularization issue,
not the sole cause.

## 3. Enlarged population audit

All 160,007 training contexts were screened at q6. The 2,048 highest-risk contexts were
then confirmed at q7, up from the earlier 384.

- 12 trip rows have \(P(N\ge60\mid x)\ge0.5\).
- They correspond to 11 household-store-day contexts and 9 households.
- Ten observed baskets contain fewer than 40 products; the other two contain 42 and 46.
- None of the 12 observed baskets contains 60 products.
- Their observed mean is \(21.92\); model expected mean is \(67.96\).
- Their average \(P(N\ge60)\) is \(0.59345\), with maximum \(0.73837\).

The conservative upper envelope for every omitted context is \(0.17428\), so enlarging
the panel does not reveal another hidden set of majority-tail contexts.

There are genuine large baskets:

| Split | \(N\ge60\) | Rate |
|---|---:|---:|
| Training | 300 / 160,007 | 0.1875% |
| Validation | 27 / 17,351 | 0.1556% |
| Test | 21 / 23,340 | 0.0900% |

Across all splits, 348 baskets contain at least 60 distinct products, 49 contain at least
80, nine contain at least 100, and one contains 120. These rows are exact checkout-level
distinct-product counts, not clipped basket sizes. A correct fix must retain this rare
tail while assigning it to the right contexts.

## 4. Block ablation

Only \(\Phi\) and \(\rho_0\) differ between the exact additive parent and the candidate.
On the 12 pathological contexts:

| Model block | Expected size | Mean \(P(N\ge60)\) |
|---|---:|---:|
| Exact additive parent | 87.53 | 0.8100 |
| Parent plus fitted size correction | 55.14 | 0.4251 |
| Parent plus interaction only | 98.03 | 0.9180 |
| Full candidate | 67.96 | 0.5934 |

Therefore:

- the pathology already exists before Gram interactions;
- positive interactions amplify it by about 10.5 expected products;
- the two-parameter update inside \(\rho_0\) corrects about 30 products, but not enough.

For the 46 genuinely large baskets present in the enlarged high-risk panel, the full
candidate predicts mean \(30.96\) against observed \(72.22\). The model therefore does
not merely have too much tail mass; it places tail mass on the wrong occasions.

## 5. Similar context audits

The full q6 screen was aggregated by household, store and week.

- Household observed/predicted mean-size correlation is \(0.9167\), but individual
  household biases range above 15 products.
- Store-level positive bias is at most 1.93 products.
- Week-level positive bias is at most 0.85 products.
- Trip-level observed/predicted correlation is only \(0.4050\).

The dominant systematic axis is therefore household, while the remaining error is
occasion-level variation. Several affected households exhibit fill-in and stock-up trips
in the same history; a single deterministic mean cannot reveal which mission occurred
without an additional observed context feature.

## 6. Rank-one household size channel

Reserve one existing household-taste coordinate for a catalogue-common loading:

\[
b_{jv}
=
\widetilde b_{jv}
+\kappa_{h(v)},
\qquad
\frac1J\sum_j\widetilde b_{jv}=0
\quad\text{for the household-composition block}.
\tag{1}
\]

Across the household-by-product utility matrix, this is

\[
\kappa\,\mathbf 1_J^\top,
\]

which has rank one. It uses one of the existing \(K=32\) taste coordinates, leaving 31
catalogue-centred composition coordinates. It adds no parameters to the existing
\(\theta^\top\alpha\) factorization and does not increase taste rank.

For a basket \(S\),

\[
\sum_{j\in S}\kappa_h
=
|S|\kappa_h.
\tag{2}
\]

Hence Eq. (1) is equivalently

\[
\rho_{0h}(n)=\rho_0(n)-n\kappa_h.
\tag{3}
\]

This is a reparameterization inside the original item utility, not a new size law or a
change to the H--S theorem.

### Proposition 1: fixed-size composition is unchanged

For any two baskets \(S,T\) with \(|S|=|T|=n\),

\[
\frac{P_{\kappa}(S\mid x,N=n)}
     {P_{\kappa}(T\mid x,N=n)}
=
\frac{\exp\{E_0(S)+n\kappa_h\}}
     {\exp\{E_0(T)+n\kappa_h\}}
=
\frac{P_0(S\mid x,N=n)}
     {P_0(T\mid x,N=n)}.
\]

Thus \(\kappa_h\) changes household basket size but does not distort recommendation or
composition conditional on size. \(\square\)

## 7. Concave fit and deterministic safety bound

Given a base conditional size law \(p_0(n\mid x)\), the tilted law is

\[
p_\kappa(n\mid x)
=
\frac{p_0(n\mid x)e^{n\kappa_h}}
{\sum_{m=1}^{120}p_0(m\mid x)e^{m\kappa_h}}.
\tag{4}
\]

For household \(h\), fit \(\kappa_h\) by

\[
\max_{\kappa_h}
\sum_{v:h(v)=h}
\left[
n_v\kappa_h
-
\log\sum_{m=1}^{120}p_0(m\mid x_v)e^{m\kappa_h}
\right]
-
\frac{\lambda}{2}\kappa_h^2.
\tag{5}
\]

Its second derivative is

\[
-\sum_{v:h(v)=h}\operatorname{Var}_{\kappa_h}(N\mid x_v)-\lambda<0,
\]

so every household solve is one-dimensional and strictly concave.

Moreover \(P_\kappa(N\ge60\mid x)\) is monotone in \(\kappa_h\). Therefore the safety
condition is an interval constraint

\[
\kappa_h\le u_h,
\qquad
u_h=\sup\left\{
k:
\max_{v:h(v)=h}P_k(N\ge60\mid x_v)\le0.35
\right\}.
\tag{6}
\]

The fitted solution is simply

\[
\widehat\kappa_h^{\rm safe}
=
\min(\widehat\kappa_h^{\rm ridge},u_h).
\tag{7}
\]

There is no stochastic penalty weight, retry loop or non-convex search in Eq. (7).
The screen threshold \(0.35\) leaves room for the measured q7-minus-q6 tail envelope
\(0.11645\), keeping the confirmed value below \(0.5\).

## 8. Pilot result on the frozen candidate

Alternating trips within each household produced a two-fold cross-fit. Ridge values were
selected only by held-out size likelihood; the optimum was \(\lambda=4800\).

The constrained cross-fitted gain was

\[
0.002834\pm0.000216
\quad\text{nats/basket},
\]

with lower 95% bound \(0.002410\). On the full training panel the constrained fit gained
\(0.00994\) size nats/basket and capped only six households.

On the independent q7 high-risk panel:

| Diagnostic | Before | Rank-one constrained pilot |
|---|---:|---:|
| Contexts with \(P(N\ge60)\ge0.5\) | 12 | 0 |
| Maximum \(P(N\ge60)\) | 0.7384 | 0.4413 |
| Contexts with \(E[N]\ge40\) | 189 | 113 |
| Mean \(P(N\ge60)\) | 0.10146 | 0.06929 |

For genuinely observed 60+ baskets, average model tail probability rises from \(0.10425\)
to \(0.11462\). The correction therefore does not obtain safety by uniformly deleting
the real tail.

This is a frozen-law pilot, not evidence from a fresh converged rank-one model. The branch
must still run the complete pipeline and repeat likelihood, recommendation, generation,
counterfactual and population gates.

## 9. Sampling and computational cost

No new latent variable or quadrature dimension is introduced. Once the ordinary recursion
has produced size coefficients \(Z_n(x,z)\), apply

\[
Z_n^{(\kappa)}(x,z)=e^{n\kappa_h}Z_n(x,z).
\tag{8}
\]

Sampling remains:

1. sample/integrate the existing H--S variable \(z\);
2. sample basket size from
   \[
   P(N=n\mid x,z)\propto
   e^{n\kappa_h-\rho_0(n)}Z_n(x,z);
   \]
3. sample category counts and products conditional on \(n,z\) with the existing backward
   recursion.

Because \(e^{n\kappa_h}\) cancels conditional on \(n\), steps for category and product
composition are unchanged. Evaluation adds one multiply/add per size coefficient:
\(O(n_{\max})\) per context, negligible beside the existing
\(O(Jn_{\max})\) recursion. Memory increases by no model tensors in the constrained
rank-32 implementation; one existing taste coordinate is assigned an identifiable role.

## 10. Branch implementation

Branch **version4-household-size-rank1** implements the structural decomposition behind
Eq. (1):

- one existing taste coordinate is the fixed all-product loading;
- the other 31 product loadings are catalogue-centred;
- gauge projection preserves utilities exactly;
- the initialization artifact records the parameterization;
- downstream checkpoint loading restores it from metadata;
- training logs report the household-size standard deviation and maximum; and
- the production confirmation panel is increased from 384 to 2,048 contexts.

Unit tests establish the rank-one common shift, gauge-preserving projection and the
exponential size-tilt/sampling identity. A ten-update full-catalogue integration smoke
test completed successfully. No full converged run has been started from this branch.
