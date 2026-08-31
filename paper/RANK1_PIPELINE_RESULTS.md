# Certified rank-one pipeline result

## 1. Scope and lineage

This document records the completed result of the selected Version-4 pipeline on the
corrected dunnhumby cohort. The execution began from `artifacts/initialization.pt`; no
learned checkpoint from an earlier numbered run was resumed. The fitted law remains the
law in `version4.html`. The only structural reparameterization assigns one existing
household/product utility direction to a catalogue-common loading, so that the household
coordinate changes the size marginal without changing composition conditional on size.

The final checkpoint is `artifacts/candidate_rank1.pt` and the complete console record is
`artifacts/pipeline_household_rank1_full.log`.

## 2. Training result

The exact additive parent converged at update 14,300; its best fixed-panel validation
score was \(-44.736334\) at update 13,100. The split-half spectral audit selected rank 5
with overlap \(0.516813\); ranks 6--8 did not satisfy the stability gate.

The constrained natural-parameter interaction solve selected ridge \(0.0003\). Its
cross-fitted likelihood gain was \(0.024333\) nats/basket, the weaker split gained
\(0.023898\), and median proposal ESS fraction was \(0.99821\). All five eigenvalues of
the fitted interaction matrix reached the declared spectral cap. That saturation is
reported as a capacity diagnostic, not used to relax the cap after looking at test data.

The identified household-size block selected ridge 4,800 by within-household cross-fit:

\[
\Delta\ell_{\mathrm{size}}^{\mathrm{CF}}
=0.00320684\pm0.00022071,
\qquad
\operatorname{LCB}_{95}=0.00277425.
\]

The full-panel gain was \(0.01026735\) nats/basket and five households reached the
predeclared conditional-tail cap.

## 3. Locked likelihood

All values below use complete support \(1\le |S|\le120\), the same 4,096 trips for parent
and child, q7 for the interaction model, and an exact dynamic program for the additive
parent.

| Split | Rank-one model | Exact additive parent | Paired gain |
|---|---:|---:|---:|
| Validation | \(-43.687816\) | \(-43.714530\) | \(0.026714\pm0.002107\) |
| Test | \(-46.064895\) | \(-46.097646\) | \(0.032750\pm0.002393\) |

On independent 128-trip panels, q7-minus-q8 error had 95% absolute upper bound
\(0.000318\) nats on validation and \(0.000468\) on test. After charging those bounds
against the paired gains, both lower confidence bounds remain positive. The interaction
model therefore has a statistically and numerically established likelihood gain over its
matched additive parent.

## 4. Recommendation

The locked test protocol hides one bought product and ranks all contemporaneously
available products by conditional add-one energy. The normalizer cancels, so these metrics
do not depend on q7.

| Metric | Result |
|---|---:|
| Eligible cases | 1,615 |
| MRR | \(0.095246\pm0.006075\) |
| MRR@5 | \(0.081899\) |
| MRR@10 | \(0.087099\) |
| MRR@20 | \(0.090316\) |
| Recall@5 | \(0.122601\) |
| Recall@10 | \(0.160991\) |
| Recall@20 | \(0.207430\) |

The interaction-minus-additive MRR gain is
\(0.0011649\pm0.0006271\), with 95% interval
\([-0.000064,0.002394]\). Total recommendation quality is useful, but this panel does not
establish a separate interaction-only recommendation improvement at the 5% level.

## 5. Population basket-size certification

The cheap q6 rule screened all 160,007 modeled contexts. It estimated observed/model mean
sizes \(7.64925/7.59039\) and observed/model \(N\ge60\) rates
\(0.001875/0.001660\). No screened context had majority extreme-tail probability.

The q7 confirmation evaluated 2,048 high-risk contexts and a separate random calibration
panel of 2,048 contexts:

- maximum confirmed \(P(N\ge60\mid x)\): \(0.402682\);
- contexts with \(P(N\ge60\mid x)\ge0.5\): zero;
- contexts with confirmed \(E[N\mid x]\ge60\): zero; and
- calibrated population-tail 95% upper bound: \(0.002013\), below the allowed
  \(0.004250\).

All four declared population gates pass. The mean absolute q6--q7 expected-size gap is
\(0.8002\) products, with maximum \(5.2889\); the one-item q6 fidelity diagnostic therefore
does not pass. Safety is based on q7 confirmation and conservative coverage envelopes,
not on pretending q6 is exact.

## 6. Generation and counterfactuals

Generation produced no unavailable products and no duplicate items. Minimum normalized
SMC ESS was \(0.99938\), and the price audit showed the required monotone direction.

The small segment-balanced generation panel nevertheless exposes a material calibration
caveat:

| Size moment | Generated | Selected observed baskets |
|---|---:|---:|
| Mean | 7.3406 | 10.0313 |
| Variance | 63.8349 | 136.2803 |

This does not contradict the full-population mean/tail pass: the 64-context-per-segment
panel is deliberately balanced and has a different empirical context distribution. It
does mean that segment-level rollout claims need a larger, distribution-weighted
generation audit before deployment.

## 7. Promotion-policy experiment

The three-segment, 28-day budget-constrained environment reweights common factual SMC
particles under 10% and 20% five-SKU discounts. It optimizes a paired 95% lower-confidence
reward subject to expected markdown budget, ESS and extreme-tail gates. The evaluated
policies used 98.9--99.4% of their scenario budgets without overspend. The strongest
supported action was a 10% milk-bundle discount for segment 1; its per-segment-trip
incremental list-price basket value was \(0.9764\pm0.1511\), with lower bound \(0.6803\).

These results support campaign shortlisting, not autonomous pricing. The model conditions
on an arriving nonempty trip and does not contain visit probability, quantities,
wholesale costs, inventory or causal treatment identification. Full details are in
[SEGMENT_PROMOTION_MDP.md](SEGMENT_PROMOTION_MDP.md).

## 8. Decision

The pipeline passes its declared technical likelihood, numerical and population-tail
certification. It is the accepted implementation for continued research and controlled
rollout experiments. Commercial production readiness remains withheld until the
segment-level generation mismatch is resolved or bounded on a larger weighted panel and
promotion response is validated with randomized interventions and economic state.
