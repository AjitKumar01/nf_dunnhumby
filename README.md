# Version-4 energy basket model

This branch is the minimal, reproducible implementation of the model in
[`paper/version4.html`](paper/version4.html). It contains one selected end-to-end
pipeline—not a collection of numbered experiment runs.

The statistical law is unchanged:

\[
p_\theta(S\mid x)=\frac{\exp E_\theta(S,x)}{Z_\theta(x)},
\qquad 1\leq |S|\leq120,
\]

over the complete 5,455-product catalogue, with the original additive utility,
household/context effects, price response, category term, total-size potential, and
low-rank Gram interactions. Recommendation, basket generation, price counterfactuals,
and retailer simulation all come from this same joint law; there is no separate
recommendation training objective.

## Why this pipeline

The selected pipeline is:

1. deterministic data preparation and a training-only affinity partition;
2. fresh initialization—no learned checkpoint is loaded;
3. exact additive joint maximum likelihood using the category/cardinality dynamic
   program, with validation-driven learning-rate decay, a convergence gate, and a
   complete-support bound on the existing category coefficient;
4. one split-half spectral pass that certifies the largest stable rank from 8 down to 4;
5. one cross-fitted constrained Monte Carlo likelihood solve for the PSD interaction
   kernel and the original total-size potential correction, using fixed exact draws from
   the additive law;
6. one strictly concave household-size block update in an identified catalogue-common
   direction of the existing household/product utility, selected by within-household
   cross-fit and capped by the population tail screen;
7. locked complete-support likelihood, recommendation, generation, price, and
   population-tail audits.

This is the corrected end-to-end pipeline. It never searches over \(\rho_0\)
initializations and does not run stochastic Smolyak gradients. In the certified product
basis it writes \(K=UCU^\top\), optimizes \(C\) directly under
\(0\preceq C\preceq I\), and jointly solves a linear/quadratic correction inside the
original \(\rho_0(n)\). Fixed common draws make the sampled likelihood deterministic and
concave in these natural parameters. PSD/tail projection and Armijo backtracking make
every accepted step monotone. Cross-fit gain and proposal effective sample size are
checked before the independent Smolyak audit. Certification remains fail-closed: if the
optimizer does not converge, numerical fidelity fails, or the model
puts excessive mass on anomalously large baskets, the last stage exits nonzero and keeps
the candidate artifacts for diagnosis rather than calling them production-ready.

The category bound is

\[
(-\rho_c)_+{m_c\choose2}\le1.5,
\]

where (m_c) is the largest supported offered count in category (c). It preserves the
Version-4 energy exactly while preventing a broad learned affinity group from creating an
unobserved large-basket clique phase. Narrow two-item groups retain the original
(-1.5) coefficient range.

The evidence and decision rule are in [`paper/PIPELINE.md`](paper/PIPELINE.md). The full
model derivation is in [`paper/THEORY.md`](paper/THEORY.md), and estimator details are in
[`paper/ESTIMATOR.md`](paper/ESTIMATOR.md).
The completed corrected-data fit and its fail-closed production decision are reported in
[`paper/CORRECTED_PIPELINE_RESULTS.md`](paper/CORRECTED_PIPELINE_RESULTS.md).
The exact-enumeration interaction recovery benchmark and its real-data diagnosis are in
[`paper/SYNTHETIC_INTERACTION_AUDIT.md`](paper/SYNTHETIC_INTERACTION_AUDIT.md).
The conditional size failure, rank-one household reparameterization, constrained pilot,
and unchanged sampling recursion are derived in
[`paper/HOUSEHOLD_SIZE_AUDIT.md`](paper/HOUSEHOLD_SIZE_AUDIT.md).

## Current empirical status

The parent corrected pipeline has completed. On locked 4,096-trip panels, the rank-5 model
improves over its matched exact additive parent by \(0.02163\pm0.00158\) nats/basket on
validation and \(0.02391\pm0.00165\) on test; both gains remain positive after the
higher-level numerical audit. Total test recommendation MRR is \(0.09514\pm0.00608\), but
the interaction-only MRR gain is not significant.

Production certification is **rejected**: although the aggregate \(N\ge60\) rate passes,
the high-accuracy audit finds localized contexts with majority probability on extreme
baskets. The candidate is therefore suitable for research diagnosis, not production
generation or retailer policy simulation. Historical baseline numbers from the old
preprocessing are not treated as corrected-data comparisons.

This branch adds the audited rank-one household-size correction. Its frozen-law pilot
removes all 12 confirmed majority-tail contexts while improving cross-fitted size
likelihood, but those pilot numbers are not substituted for a fresh converged end-to-end
result. The full branch pipeline must pass the same locked gates before certification.

## Requirements

- Python 3.11 or newer
- a C++ compiler compatible with the active Python installation
- about 12 GB free RAM for the full-population stages
- dunnhumby *The Complete Journey* CSV files (not redistributed here)

Install:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pytest -q
```

## Data

Download the dunnhumby CSV bundle and point `NF_RAW_DIR` at the directory containing
`transaction_data.csv`, `product.csv`, and `causal_data.csv`:

```bash
export NF_RAW_DIR="/absolute/path/to/dunnhumby_The-Complete-Journey CSV/"
```

Derived files under `data/` and `basket_input/` are deliberately ignored by Git because
the source-data license does not permit redistribution.

The cohort is defined using training weeks 9--82 only: the top 5,455 products by training
line frequency, followed by households with 20--300 training shopping days. Validation is
weeks 83--90 and test is weeks 91--101. Weeks outside 9--101 are excluded because the
promotion file does not cover them. No held-out purchase line is removed after cohort
definition; in the absence of a stock feed, every modeled store uses the complete declared
5,455-product chain catalogue.

Every pipeline invocation runs `scripts/data/audit_preprocessing.py` before fitting. It
checks the locked raw-file digests, independently reconstructs cohort membership and every
basket-product outcome, verifies modal chain/store prices and promotion coverage, and
writes `basket_input/preprocessing_manifest.json`. See
[`paper/PREPROCESSING_AUDIT.md`](paper/PREPROCESSING_AUDIT.md).

## Check before spending compute

Print the exact command graph without training:

```bash
python scripts/run_pipeline.py --from-raw --dry-run
```

The dry run is the recommended first command after cloning. It verifies paths and shows
every stage, fixed panel size, rank gate, quadrature level, and output location.

## Full end-to-end execution

From raw CSVs:

```bash
mkdir -p artifacts
python scripts/run_pipeline.py --from-raw 2>&1 | tee artifacts/pipeline.log
```

If `data/` and `basket_input/` already exist:

```bash
python scripts/run_pipeline.py 2>&1 | tee artifacts/pipeline.log
```

If the machine stops during the exact-additive stage, continue the same from-scratch run
without resetting Adam or replaying minibatches:

```bash
python scripts/run_pipeline.py --resume-additive out/v3_pipeline_additive.pt \
  2>&1 | tee -a artifacts/pipeline.log
```

The full profile has a 30,000-update safety ceiling but must satisfy the convergence gate;
reaching that ceiling is a nonzero pipeline failure and no interaction/evaluation stage is
then allowed to run.

To stop after a stage:

```bash
python scripts/run_pipeline.py --stop-after additive
python scripts/run_pipeline.py --stop-after rank
python scripts/run_pipeline.py --stop-after evaluation
```

`--profile smoke` uses tiny panels, fixes rank 4, and relaxes statistical fit gates solely
to exercise every code path. It still writes the population-tail report but does not fail
the integration test when that report rejects the deliberately undertrained model. Smoke
output is not statistically valid and must never be used for reporting results.

## Outputs and logs

| Path | Meaning |
|---|---|
| `artifacts/pipeline.log` | complete console log when invoked with `tee` as above |
| `out/v3_pipeline_additive.log` | exact additive optimizer log |
| `out/v3_pipeline_additive_best.pt` | best additive parent |
| `artifacts/interaction_basis_rank*.{npz,json}` | rank audits; rejected ranks remain auditable |
| `artifacts/candidate.{pt,json}` | interaction candidate before the household-size block update |
| `artifacts/candidate_rank1.{pt,json}` | final rank-one household-size candidate and cross-fit/safety audit |
| `reports/likelihood_{validation,test}.json` | paired, complete-support likelihood |
| `reports/recommendation.json` | locked exact add-one MRR, MRR@5/10/20 and Recall@5/10/20; no normalizer is used |
| `reports/generation_counterfactual.json` | SMC validity and price response |
| `reports/customer_segments.json` | segment structure, generation, and price response |
| `reports/population_size.json` | full-population size/tail certification |

The population audit checkpoints its screen under
`reports/population_size.screen-<digest>.*`. An interruption resumes at the last durable
boundary. If the cheap signed rule is invalid for an individual context, only that context
is escalated to the next quadrature level and the trip ID is recorded in the final report.

The last audit may reject a fitted candidate. In the full profile that rejection makes
the pipeline exit nonzero, while preserving the reports and candidate for diagnosis. It
means the fitted law is not safe for production simulation; it is not an estimator crash.
Only the smoke profile converts that statistical rejection into a successful integration
exit, because smoke is deliberately too small to certify a model.

## Reading the recommendation metrics

For each eligible held-out basket, one bought item is hidden. Every contemporaneously
available product is ranked by its exact conditional add-one energy. `MRR` averages
\(1/r\), where \(r\) is the hidden product's rank. `MRR@k` is \(1/r\) only when
\(r\leq k\), otherwise zero. The normalizer cancels in this conditional ranking, so MRR
does not use or tune the Smolyak level.

## External baseline suite

The repository includes independently trainable implementations of five basket models:

| Baseline | Normalized basket law |
|---|---|
| Multinomial | empirical training-only \(P(n)\) times an exact distinct-item ESP composition, restricted to \(1\le n\le120\) |
| Bernoulli | exact independent-product set law restricted to \(1\le n\le120\) |
| DPP | exact non-empty low-rank determinantal law; probability beyond size 120 is bounded in the report |
| NDPP | exact non-empty nonsymmetric determinantal law; probability beyond size 120 is bounded in the report |
| SHOPPER | sequential interaction model converted to a set probability by summing small order spaces exactly and using reproducible sampled orderings otherwise; checkout is forced at 120 |

These baselines are deliberately a separate command from the main pipeline. This prevents
an ordinary model fit from silently spending compute on five competitors. Print the full
command graph first:

```bash
python scripts/run_baselines.py --dry-run
```

The default profile trains every baseline from a fresh lineage until it satisfies the
same explicit optimization contract: at least two epoch-equivalents of minibatch exposure,
trips, validation-driven learning-rate reductions, the declared learning-rate floor, and
four further stale validation checks at that floor. It then scores each validation-selected
checkpoint on the main model's locked test trips:

```bash
python scripts/run_baselines.py --profile converged \
  2>&1 | tee artifacts/baselines_converged.log
```

The 60,000-update setting is only a safety ceiling. Reaching it without satisfying the
certificate exits nonzero and produces no test comparison. An interrupted fresh lineage
can resume without resetting its optimizer, validation state, or random streams:

```bash
python scripts/run_baselines.py --profile converged --resume-training \
  2>&1 | tee -a artifacts/baselines_converged.log
```

The historical equal-update experiment remains reproducible with
`--profile published-1000`, but it is explicitly not a converged-model comparison. Use
`--skip-training` to rescore checkpoints belonging to the selected profile.

For a quick installation check, `--profile smoke` trains each model for two updates and
scores only the first 16 locked trips. Its relaxed certificate exercises the control flow
only and must not be reported as model quality. The converged comparison writes
`reports/baselines_converged.json` plus `reports/baselines_converged_per_trip.npz`; the JSON
records checkpoint hashes, selected and terminal iterations, convergence certificates,
manifest hash, likelihoods, paired standard errors, and support diagnostics.

## Reproducibility contract

- The first learned stage starts only from `artifacts/initialization.pt`.
- The interaction rank is selected using training split halves, not validation/test.
- All 5,455 product rows remain active in the interaction fit.
- Final likelihood uses support 1 through 120 and fixed identical trip panels.
- Test data never selects a checkpoint or estimator.
- A candidate is not production-certified unless every final gate passes.
