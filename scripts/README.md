# Scripts

Organised around the two documents. `paper/model_spec.html` defines the model;
`paper/experiments.html` reports what it does. Everything in this directory produces one
or the other. Everything that does not is in `../archive/`.

Run each script from its own directory. They resolve the repository root relatively, and
`eval/` scripts add `../model` to `sys.path` so `27_nested_basket` and
`28_nested_counterfactual` import by their bare module names.

```
scripts/
  pipeline/   raw dunnhumby  ->  data/  ->  basket_input/
  model/      the model itself, and generation
  eval/       one script per section of experiments.html
```

## pipeline/

| script | reads | writes |
|---|---|---|
| `01_build_base.py` | raw dunnhumby CSVs | `data/` — `tx`, `trips`, `price_week`, `price_store_week` |
| `22_basket_data.py` | `data/tx.parquet`, raw `product.csv` | `basket_input/` — the model's only input |

`basket_input/` is what every script below reads. Rebuilding it from scratch is
`01` then `22`; nothing else in the repository is on that path.

## model/

| script | what it is |
|---|---|
| `27_nested_basket.py` | Eqs. 4–21 of the specification, and the fitting loop |
| `28_nested_counterfactual.py` | price derivatives (Appendix C) and the §12 sampler |

Key flags: `--l2-incidence 1e-4` (**required** — the shared penalty collapses
`c_user`/`c_cat` to effective rank 5 and 4 of 64), `--no-persist`, `--spec-edges`,
`--placebo-price permute`, `--w-incidence`.

## eval/

Each maps to a section of `experiments.html`.

| script | section | produces |
|---|---|---|
| `33_verify_equations.py` | §2 | 30 equation checks; must print `all equations verified` |
| `43_bootstrap.py` | §3 | household block bootstrap, paired CIs |
| `44_baselines_exact.py` | §5 | HPF and B-Emb on the exact conditional |
| `45_shopper.py` | §5 | SHOPPER, scored within-category and full-catalogue |
| `34_generator_eval.py` | §7 | basket shape, co-occurrence, recovered price response |
| `39_representation_eval.py` | §8, §9 | embedding probes and the price-sensitivity block |
| `40_embedding_structure.py` | §8 | hierarchy agreement and the t-SNE figure |
| `47_embeddings_verified.py` | §8 | purity against a permutation null and a random embedding |
| `41_targeting.py` | §10 | quintiles, the sign screen, and the recency transition |
| `46_horizon.py` | §11 | whether the novelty excess compounds over a rollout |
| `42_limitations.py` | §12 | assortment sensitivity, temperature, the classifier two-sample test |

Two arguments are not defaults and matter for reproducing the published numbers:
`34_generator_eval.py` was run with `--n-trips 24768 --top-items 500`, and every model
was fitted with `--l2-incidence 1e-4`.

## A caution about labels

`42_limitations.py` defaults to `--label spec_nested`, a superseded fit. Pass `--label`
explicitly. `33_verify_equations.py` previously ignored `--label` entirely, which is how a
failing check went unnoticed for a day; it now reads it.

## ../archive/

Not used by either document. Kept because the results are cited historically, not because
anything runs.

| directory | what it is |
|---|---|
| `normalizing_flow/` | the earlier project phase — a normalizing-flow demand model on `model_input/`, plus `nf_torch.py` and stages 02–20 |
| `flat_basket/` | stage 23's flat basket model and its evaluation, superseded by `model/27` |
| `superseded_eval/` | EDA and evaluation stages replaced by `eval/` — sampled-metric benchmarks, context ablations, the earlier exact-likelihood check |
| `runners/` | old shell drivers |

`normalizing_flow/04_extras.py` is the one archived script worth knowing about: it is the
only code that reads dunnhumby's `causal_data.csv` (display and mailer placement per
product × store × week). Wiring that into `pipeline/22_basket_data.py` is the open step
§13 of `experiments.html` calls highest-value, because it is the only real instrument for
price in this dataset.
