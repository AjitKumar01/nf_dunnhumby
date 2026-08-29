# Pipeline decision for the Version-4 model

## 1. The unit of comparison

A checkpoint is an output, not a method. A numbered experiment can look better because it
used a different parent, panel, split, support, rank, or quadrature rule. Therefore this
project compares **pipelines** under one contract:

\[
\mathcal P=(\text{data},\text{initialization},\text{objective},
\text{normalizer},\text{optimizer},\text{acceptance gates}).
\]

Two outputs are comparable only when these components and the evaluation manifest agree.
Historical numeric labels are retained only in old provenance; they are not model names.

## 2. Non-negotiable model contract

Every admissible pipeline fits the joint law in `version4.html`:

\[
p_\theta(S\mid x)=\frac{\exp\{E_\theta(S,x)\}}{Z_\theta(x)},
\quad
Z_\theta(x)=\sum_{1\leq |A|\leq120}\exp\{E_\theta(A,x)\}.
\]

The catalogue has 5,455 products. The Gram term is

\[
\sum_{i<j\in S}\phi_i^\top\phi_j
=\frac12\left(\left\|\sum_{j\in S}\phi_j\right\|^2
-\sum_{j\in S}\|\phi_j\|^2\right).
\]

No pipeline may replace the joint law with a conditional-size model, remove interactions,
truncate the catalogue to convenient products, or optimize MRR instead of likelihood.

## 3. Decision order

Pipeline selection is lexicographic, not a hand-tuned weighted average:

1. unchanged law and complete support;
2. fresh, reproducible completion without estimator aborts;
3. numerical score fidelity at the accepted rank;
4. population basket-size and generation calibration;
5. paired held-out likelihood against the exact additive parent and external baselines;
6. interaction contribution to held-out recommendation;
7. wall time and memory.

This ordering prevents a small likelihood gain from buying a pathological simulator.

## 4. Pipelines evaluated

| Pipeline family | Normalizer/training idea | What the evidence establishes | Decision |
|---|---|---|---|
| End-to-end RQMC joint SGD | randomized Gaussian integral on every update | high latency, retries/aborts, and no completed convincing convergence path | reject as default |
| Long joint Smolyak SGD | q8 large-batch score plus q9 correction and q10 audit | stable quadrature and positive likelihood gain, but roughly 1.1 s/update and rare size phases remain | optional research refinement |
| Post-fit scalar/context size corrections | tilt existing \(\rho_0\) or household-common direction | changes are cheap, but measured gains are marginal and the full-population extreme tail worsens | reject |
| Exact additive + rank score + projected Fisher + original \(\rho_0\) solve | exact dynamic program first; deterministic Smolyak only for final interaction law | fresh completion, full-catalogue rank learning, positive likelihood direction, valid recommendation/generation path, much less expensive interaction optimization | **selected** |

No evaluated pipeline has yet passed the new full-population extreme-tail gate. Therefore
the selected pipeline is the best **research and certification pipeline**, not a declaration
that the current fitted parameters are production-ready.

## 5. Selected pipeline

### Stage A — data and support

Raw transactions are converted to trips, item/day prices, store price deviations,
promotions, recency state, and a training-only affinity partition. Availability is
store-specific by category and chain-wide within category. An observed training basket
outside this support is a hard error.

### Stage B — exact additive maximum likelihood

Set \(\Phi=0\). The H--S integral disappears and the category/cardinality dynamic program
computes \(Z(x)\) and its gradients exactly for all products and sizes 1 through 120. This
stage fits all original non-Gram incidence parameters. It is the fast convergence phase.
The 12,000-update setting is only a safety ceiling: validation plateaus lower the learning
rate, and convergence is declared only after the minimum rate stops producing new bests.

### Stage C — rank identification

At \(\Phi=0\), the ordinary gradient with respect to \(\Phi\) is zero because the energy is
quadratic in \(\Phi\). The informative local object is instead the pair-statistic score

\[
R=\mathbb E_{\rm data}[P(S)]-\mathbb E_{p_{\rm add}}[P(S)].
\]

Positive eigenvectors of \(R\) are locally improving positive-semidefinite Gram
directions. Ranks 8 down to 4 are tested on independent training halves. The largest rank
whose mean squared subspace overlap is at least 0.5 is accepted. Rank is capacity; it is
not quadrature accuracy.

### Stage D — projected Fisher interaction fit

In the accepted basis \(U\), write

\[
K=\Phi\Phi^\top=UCU^\top,\qquad C\succeq0.
\]

Only \(r(r+1)/2\) coordinates are fitted. The score and Fisher covariance are accumulated
over the full supported training population. Ridge values are selected by swapped-half
predicted likelihood gain, and a candidate is rejected unless both directions improve.
This replaces thousands of expensive noisy updates with one statistically meaningful
second-order solve.

### Stage E — original size-potential recalibration

Positive interactions alter basket-size moments. Holding other parameters fixed, the
pipeline fits a low-dimensional correction inside the existing \(\rho_0(n)\):

\[
\Delta\rho_0(n)=a n+c n^2.
\]

The solve uses 32 within-context draws because the objective contains a nonlinear log
average. It must pass swapped-half likelihood gates. This is parameter estimation inside
the original model, not a new factorization or theorem.

### Stage F — certification

The candidate must pass:

- q9 paired validation and test likelihood, with q10 numerical audit;
- exact conditional add-one MRR and recall on a fixed test manifest;
- SMC validity: no duplicates or unavailable products and adequate ESS;
- monotone uniform-price counterfactual response;
- segment-specific generation and price-response diagnostics;
- q8 screening over every supported training context and q9 confirmation of the
  highest-risk size laws.

The population gate rejects a candidate when its mean tail probability is incompatible
with observed \(N\geq60\) frequency or when a context with observed size below 40 assigns
more than half its mass to \(N\geq60\). This directly targets the rare phase transition
that small validation panels missed.

## 6. Complexity

Let \(B\) be batch size, \(J_x\) the offered products in a context, \(C_x\) its nonempty
affinity groups, \(n_{\max}=120\), accepted rank \(r\), and Smolyak node count \(M_q(r)\).

- Exact additive update: approximately
  \(O(B[J_x n_{\max}+C_x n_{\max}^2])\), with no quadrature nodes.
- Pair-score construction: sparse observed/generated co-incidence plus a sparse leading
  eigensolve; the dense \(5455^2\) Gram matrix is never materialized.
- Projected Fisher accumulation:
  \(O(TD[r^2+r^4])\) for \(T\) contexts and \(D\) draws, with only
  \(r(r+1)/2\) fitted coordinates.
- Smolyak likelihood audit:
  \(O(T M_q(r)[J_x n_{\max}+C_x n_{\max}^2])\).
- Recommendation: \(O(J_x r)\) add-one scoring; \(\log Z\) cancels, so no Smolyak cost.

For rank 7, q8/q9/q10 use 15/127/785 nodes. This is why q8 is used for a population
screen, q9 for reported likelihood, and q10 only for a small numerical certification
panel.

## 7. What “best” means now

The selected pipeline is best because it is the simplest one that preserves the complete
theory, learns full-catalogue interactions, completes reproducibly, and exposes the known
failure rather than hiding it behind a favorable checkpoint. If the population-tail gate
fails, the next scientific task is parameter/regularization work within this pipeline;
it is not another estimator lottery or a comparison of numbered runs.
