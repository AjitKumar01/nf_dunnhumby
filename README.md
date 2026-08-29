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
   program, with validation-driven learning-rate decay and a convergence gate;
4. one split-half spectral pass that certifies the largest stable rank from 8 down to 4;
5. full-population projected-Fisher fitting in the accepted interaction subspace;
6. cross-fitted recalibration of the original total-size potential \(\rho_0\);
7. locked complete-support likelihood, recommendation, generation, price, and
   population-tail audits.

This is the best current pipeline-level trade-off. End-to-end QMC training was slower and
did not produce a completed, stable convergence path. Long joint Smolyak SGD was
numerically stable but added latency and did not solve the rare basket-size phase. Post-fit
scalar size tilts gave only marginal likelihood changes and worsened the extreme tail.
The selected staged estimator gets the useful interaction direction without thousands of
expensive quadrature-gradient updates. Certification remains fail-closed: if the model
puts excessive mass on anomalously large baskets, the last stage exits nonzero and keeps
`artifacts/candidate.pt` as a research candidate rather than calling it production-ready.

The evidence and decision rule are in [`paper/PIPELINE.md`](paper/PIPELINE.md). The full
model derivation is in [`paper/THEORY.md`](paper/THEORY.md), and estimator details are in
[`paper/ESTIMATOR.md`](paper/ESTIMATOR.md).

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
| `artifacts/interaction.pt` | accepted projected-Fisher interaction candidate |
| `artifacts/candidate.pt` | size-recalibrated joint-law candidate |
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

Train every baseline from scratch for exactly 1,000 optimizer updates and score the
iteration-1,000 checkpoints on the main model's locked test trips:

```bash
python scripts/run_baselines.py --profile published-1000 \
  2>&1 | tee artifacts/baselines.log
```

This is a matched-update experiment, not a claim that every model has converged. To rerun
only the paired evaluation using existing checkpoints:

```bash
python scripts/run_baselines.py --profile published-1000 --skip-training
```

For a quick installation check, `--profile smoke` trains each model for two updates and
scores only the first 16 locked trips. Smoke output tests execution only and must not be
reported as model quality. The comparison writes `reports/baselines.json` plus
`reports/baselines_per_trip.npz`; the JSON records checkpoint hashes, manifest hash,
per-basket and per-line likelihood, paired standard errors, and support diagnostics.

## Reproducibility contract

- The first learned stage starts only from `artifacts/initialization.pt`.
- The interaction rank is selected using training split halves, not validation/test.
- All 5,455 product rows remain active in the interaction fit.
- Final likelihood uses support 1 through 120 and fixed identical trip panels.
- Test data never selects a checkpoint or estimator.
- A candidate is not production-certified unless every final gate passes.
