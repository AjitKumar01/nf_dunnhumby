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
4. split-half interaction-rank certification, trying ranks 8 down to 4;
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

To stop after a stage:

```bash
python scripts/run_pipeline.py --stop-after additive
python scripts/run_pipeline.py --stop-after rank
python scripts/run_pipeline.py --stop-after evaluation
```

`--profile smoke` uses tiny panels to exercise plumbing; it is not statistically valid
and must never be used for reporting results.

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
| `reports/recommendation.json` | MRR, MRR@5/10/20 and Recall@5/10/20 |
| `reports/generation_counterfactual.json` | SMC validity and price response |
| `reports/customer_segments.json` | segment structure, generation, and price response |
| `reports/population_size.json` | full-population size/tail certification |

The last audit is intentionally allowed to fail. A failure means the code ran correctly
and found that the fitted law is not safe for production simulation; it is not an
estimator crash.

## Reading the recommendation metrics

For each eligible held-out basket, one bought item is hidden. Every contemporaneously
available product is ranked by its exact conditional add-one energy. `MRR` averages
\(1/r\), where \(r\) is the hidden product's rank. `MRR@k` is \(1/r\) only when
\(r\leq k\), otherwise zero. The normalizer cancels in this conditional ranking, so MRR
does not use or tune the Smolyak level.

## Reproducibility contract

- The first learned stage starts only from `artifacts/initialization.pt`.
- The interaction rank is selected using training split halves, not validation/test.
- All 5,455 product rows remain active in the interaction fit.
- Final likelihood uses support 1 through 120 and fixed identical trip panels.
- Test data never selects a checkpoint or estimator.
- A candidate is not production-certified unless every final gate passes.
