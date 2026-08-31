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
| Long joint Smolyak SGD from an arbitrary checkpoint | q8 large-batch score plus q9 correction and q10 audit | stable quadrature and positive likelihood gain, but poor initialization wastes updates and rare size phases remain | reject as a standalone pipeline |
| Post-fit scalar/context size corrections | tilt existing \(\rho_0\) or household-common direction | changes are cheap, but measured gains are marginal and the full-population extreme tail worsens | reject |
| Exact additive + rank score + constrained natural-parameter MCLE | exact parent draws define one deterministic likelihood-ratio objective in \(C\) and a correction inside the original \(\rho_0\) | full-catalogue rank learning, monotone optimization, cross-fit selection, and no quadrature inside training | **selected corrected pipeline** |

No evaluated pipeline has yet passed the new full-population extreme-tail gate. Therefore
the selected pipeline is the best **research and certification pipeline**, not a declaration
that the current fitted parameters are production-ready.

## 5. Selected pipeline

### Stage A — data and support

Raw transactions are converted to checkout baskets, modal item/week prices, modal store
price deviations, promotions, recency state, and a training-only affinity partition. The
product and household cohorts are defined using training weeks only. Because the source
has no stock feed, likelihood support is the complete declared 5,455-product chain
catalogue at each of the 115 modeled stores. An observed basket in *any* split outside this
support is a hard error; outcomes are never deleted to fit the support. Promotion-missing
weeks outside 9--101 are excluded. A raw/derived digest manifest and an independent
outcome/price reconstruction audit must pass before initialization.

### Stage B — exact additive maximum likelihood

Set \(\Phi=0\). The H--S integral disappears and the category/cardinality dynamic program
computes \(Z(x)\) and its gradients exactly for all products and sizes 1 through 120. This
stage fits all original non-Gram incidence parameters. It is the fast convergence phase.
The 30,000-update setting is only a safety ceiling: validation plateaus lower the learning
rate, and convergence is declared only after the minimum rate stops producing new bests.

The original category parameter is fitted subject to the complete-support admissibility
constraint

\[
(-\rho_c)_+{m_c\choose2}\le1.5,
\]

where (m_c) is the largest available count of category (c) on support 1 through 120.
This prevents a broad affinity group from being extrapolated as a 120-product attractive
clique. It changes neither the Version-4 energy nor the exact dynamic program and costs
(O(C)) per update.

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

### Stage D — constrained interaction and size MCLE

In the accepted basis \(U\), write

\[
K=\Phi\Phi^\top=UCU^\top,\qquad C\succeq0.
\]

Let \(S_{md}\) be fixed exact draws from the fitted additive law for context \(m\),
and let

\[
h_{C,a,c}(S)=\operatorname{tr}\{C F_U(S)\}
-a\frac{|S|}{10}-c\left(\frac{|S|}{10}\right)^2.
\]

The sampled log-likelihood gain over the additive parent is

\[
\widehat G(C,a,c)=\frac1M\sum_{m=1}^M\left[
h(S_m^{\rm obs})-log\left\{\frac1D\sum_{d=1}^D
e^{h(S_{md})}\right\}\right].
\]

This is concave because a linear term minus log-sum-exp is concave. The feasible set

\[
0\preceq C\preceq \sigma_{\max}^2I,\qquad
c\ge0,\qquad a+(n_{\max}/10)c\ge0
\]

is convex. The last two inequalities prevent the correction from creating an attractive
large-size tail while permitting the negative linear coefficient needed to preserve the
mean. A diagonal conditional-Fisher preconditioner removes the scale mismatch between
size and pair statistics; projection and Armijo backtracking still decide acceptance, so
the recorded objective is monotone.

Ridge is selected by swapped context halves. Both held-out directions must improve and
importance effective sample size must pass its declared floor. The full solve then
recovers \(\Phi=U C^{1/2}\). This trains an interaction vector for every one of the 5,455
products without optimizing 5,455 by \(r\) unidentified factor coordinates.

Positive interactions alter basket-size moments, so the same solve fits a low-dimensional
correction inside the existing \(\rho_0(n)\):

\[
\Delta\rho_0(n)=a n+c n^2.
\]

The correction is not a new size factor or a change to the Version-4 joint law. It is a
two-direction update of the already-defined unrestricted size potential. There is no
initialization search: concavity supplies one global sampled optimum.

### Stage E — certification

The candidate must pass:

- paired validation and test likelihood at \(q=r+2\), with a \(q=r+3\) numerical audit;
- exact conditional add-one MRR and recall on a fixed test manifest;
- SMC validity: no duplicates or unavailable products and adequate ESS;
- monotone uniform-price counterfactual response;
- segment-specific generation and price-response diagnostics;
- \(q=r+1\) screening over every supported training context and \(q=r+2\)
  confirmation of the highest-risk size laws.

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

For rank 7, q8/q9/q10 use 15/127/785 nodes. In general the selected pipeline uses
\(q=r+1\) for the population screen, \(q=r+2\) for reported likelihood and high-risk
confirmation, and \(q=r+3\) only for a small numerical certification panel. The corrected
rank-5 fit therefore uses q6/q7/q8; the rank-relative accuracy contract is unchanged.

## 7. What “best” means now

The selected pipeline is best because it is the simplest one that preserves the complete
theory, learns full-catalogue interactions, completes reproducibly, and exposes the known
failure rather than hiding it behind a favorable checkpoint. If the population-tail gate
fails, the next scientific task is parameter/regularization work within this pipeline;
it is not another estimator lottery or a comparison of numbered runs.

## 8. Corrected full-run outcome

The selected pipeline was executed from fresh initialization on the corrected cohort on
2026-08-31. It completed preprocessing, exact additive convergence, rank selection,
natural-parameter interaction fitting, locked validation/test likelihood, recommendation,
generation, segmentation and the full-population audit. The final nonzero process status
was produced deliberately by Stage E after the safety gate rejected the candidate; it was
not an optimizer or estimator abort.

| Gate | Measured outcome | Decision |
|---|---|---|
| Exact additive convergence | best fixed-panel validation LL \(-44.748944\) at update 13,500; convergence at 14,700 | pass |
| Rank stability | rank 5 overlap \(0.549018\); ranks 6--8 below \(0.5\) | rank 5 |
| Interaction cross-fit | mean gain \(0.024326\); minimum half gain \(0.023863\); median ESS fraction \(0.9981\) | pass |
| Validation likelihood | paired gain \(0.021630\pm0.001581\) nats | pass |
| Test likelihood | paired gain \(0.023908\pm0.001647\) nats | pass |
| Numerical audit | error upper bounds \(0.000510\) validation, \(0.000720\) test | pass |
| Recommendation interaction effect | MRR gain \(0.000247\pm0.000372\) | not established |
| Generator mechanics | no unavailable products or duplicates; minimum normalized ESS \(0.99945\) | pass |
| Price response | incidence and expected size decrease monotonically as price rises | pass |
| Aggregate \(N\ge60\) calibration | calibrated upper \(0.002745 < 0.004250\) allowed | pass |
| Local extreme-tail safety | 12 confirmed contexts with \(P(N\ge60)\ge0.5\); maximum \(0.73837\) for observed \(N<40\) | **fail** |

The \(C\) solution has all five eigenvalues at the declared spectral cap. This is useful
diagnostic evidence but not permission to enlarge the cap: the same candidate already
fails a local tail gate. A subsequent pipeline revision must tie any increase in
interaction capacity to a provable or directly enforced conditional-tail constraint,
then repeat a fresh end-to-end fit.

The detailed empirical record is
[CORRECTED_PIPELINE_RESULTS.md](CORRECTED_PIPELINE_RESULTS.md). The complete console log
is artifacts/pipeline_corrected_full.log; the exact-additive trace is
out/v3_pipeline_additive.log.
