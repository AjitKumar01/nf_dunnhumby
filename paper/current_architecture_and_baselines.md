# Current version-4 architecture and baseline audit

**Status date:** 2026-08-21  
**Primary implementation:** `scripts/v3/`  
**Governing model specification:** `paper/version4.html`

## 1. Scope and non-negotiable model contract

The current production model is the original version-4 joint basket model. The work in
`scripts/v3/` changes how its normalising constant is computed at full catalogue scale; it
does **not** replace the version-4 probability law.

In particular:

- all 5,455 products are present in every calculation in which they are available in the
  trip's store assortment;
- every product has a rank-32 interaction vector;
- the declared basket support is `1 <= |S| <= 120`;
- the 280-row affinity partition is selected with `V3_AFFINITY=1`;
- the original contextual size response is retained through the joint normaliser and
  `rho_0(n)`;
- `factored_size_enabled` is false; no empirical, context-free size distribution replaces
  the main model's size law;
- the earlier 20-product/rank-4 Smolyak restriction is not used by the current run.

The optional factored-size branch left in the code is an ablation/reproduction path. Runs
127--139 used that different model and must not be mixed with version-4 results. The
`--require-version4 1` guard rejects this drift in the production command.

## 2. Data universe and support

| Quantity | Current value |
|---|---:|
| Products | 5,455 |
| Households | 2,066 |
| Stores | 115 |
| Affinity rows | 280 |
| Training trips | 157,464 |
| Validation trips | 16,948 |
| Main validation audit | 384 fixed trips |
| Maximum basket size | 120 |
| Per-affinity-row degree cap `R` | 120 |

The training and validation support filters drop zero trips. Since total basket size is at
most 120, `R=120` also makes the per-affinity-row restriction non-binding. It is retained
as an explicit polynomial-recursion limit.

Trips are temporally split. The current code uses the full training split and evaluates a
fixed permutation across the validation split rather than an early time-ordered prefix.
The 384-trip audit manifest is produced by:

```text
all in-support validation IDs
-> numpy.default_rng(12345).permutation(...)
-> first 384 IDs
```

Its ordered-ID SHA-256 is
`01d0500942dfa12f1aa1a22afc7580c8e5fd17335f54f00cfacd2009a9fea963`.
It contains 3,303 purchased lines and basket sizes from 1 to 103.

## 3. Version-4 set model

For trip context `x=(h,t)` and a non-empty basket `S`, the conditional set law is

```math
P(S\mid x,S\ne\varnothing)
=\frac{\exp E_x(S)}{Z_x-1},
```

where the empty basket has energy zero and

```math
E_x(S)
= \sum_{j\in S} b_j(x)
+ \sum_{j<k;\,j,k\in S}\phi_j^\top\phi_k
- \sum_c \rho_c {n_c(S)\choose 2}
- \rho_0(|S|).
```

Here `n_c(S)` is the number of products from affinity row `c`. The pair feature is
`g(k)=k(k-1)/2` over the entire support; it is not saturated at an old implementation cap.

The interaction has two parts:

1. `phi_j' phi_k` is a learned rank-32 product interaction, able to share statistical
   strength across products.
2. `-rho_c` is a within-affinity-row pair effect. With the sign convention above, negative
   `rho_c` represents attraction and positive `rho_c` represents repulsion.

The strict no-interaction member of this family is obtained by setting
`phi=0` and `rho_c=0`. This nesting fact concerns model capacity and the maximised training
objective. It does not guarantee that an incompletely optimised full model must have better
held-out likelihood at every finite update count.

### 3.1 Trip-product utility

All paths that need item utility call the same `RaggedModel.b_at` implementation. This is
important: the observed-basket energy and the normaliser cannot silently apply different
price or context terms.

The active utility is

```math
\begin{aligned}
b_j(x)={}&\lambda_j
+(\theta_h-\bar\theta)^\top\alpha_j\\
&-\langle \operatorname{softplus}(\gamma_h),
          \operatorname{softplus}(\beta_j)\rangle
  \{\bar p_x+\operatorname{softplus}(\kappa)(\Delta\log p_{jx}-\bar p_x)\}\\
&+w^{dsp}_j\,display_{jx}+w^{mlr}_j\,mailer_{jx}\\
&+\mu_j^\top(\delta_w-\bar\delta)
+\zeta_j^\top(\xi_s-\bar\xi).
\end{aligned}
```

`bar p_x` is the assortment-level mean price deviation for the trip. Splitting the common
price level from the product-relative deviation permits aggregate basket-size elasticity
and within-basket substitution to have different scales. This is a parameterisation of
`b_j(x)` inside the original energy, not a replacement of the set law or its normaliser.
The softplus factors make own-price utility response non-positive by construction.

The model contains a four-dimensional recency loading `psi_j`, but run155 uses `--no-rec 1`;
that block is zero and frozen. Household, week and store context factors are centred to fix
bilinear gauge freedoms and leave `lambda_j` as the unique product intercept.

### 3.2 Parameter blocks

| Block | Shape | Role |
|---|---:|---|
| `lambda` | 5,455 | Product intercept |
| `alpha`, `theta` | 5,455x32; 2,066x32 | Household-product taste |
| `phi` | 5,455x32 | Set interaction embedding |
| `rho_c` | 280 | Within-affinity pair effect |
| `rho_0_free` | 120 | Total-size potential for sizes 1--120; size 0 is gauge-fixed to zero |
| `gamma`, `beta` | 2,066x8; 5,455x8 | Incidence-price sensitivity |
| `price_kappa` | 1 | Relative-price multiplier |
| `w_dsp`, `w_mlr` | 5,455 each | Promotion effects |
| `mu`, `delta` | 5,455x8; 53x8 | Product-week seasonality |
| `zeta`, `xi` | 5,455x4; 115x4 | Product-store effect |
| `psi` | 5,455x4 | Recency effect; frozen at zero in run155 |
| `a_q`, `gamma_q`, `beta_q`, `log_r` | product/household rank-8 blocks | Conditional units model |

The model reports 645,954 scalar parameters before excluding frozen blocks. At run155
iteration 1,000 all 5,455 `phi` rows are active and the effective interaction rank is 32.

### 3.3 Conditional units model

Quantity is a separate factor conditional on inclusion:

```math
q_j-1\mid j\in S,x \sim
\operatorname{NegativeBinomial}
\left(\mu_{jh}=\exp\{a^q_j-
\langle\operatorname{softplus}(\gamma^q_h),
\operatorname{softplus}(\beta^q_j)\rangle\Delta\log p_{jx}\},
r=\operatorname{softplus}(\log r)\right).
```

Thus

```math
P(S,q\mid x)=P(S\mid x)\prod_{j\in S}P(q_j\mid j\in S,x).
```

The baseline claim in this document concerns **set log likelihood only**. Units are reported
separately and are not added when deciding whether interactions improve the basket-set law.

## 4. Full-catalogue normaliser

The architecture is easiest to read as the following pipeline:

```text
trip/store assortment + context
              |
              v
       b_j(x), phi_j, rho_c
              |
              v
 scrambled Sobol z in R^32 -> exact per-category ESP polynomials
              |                         |
              +-------------------------+
                          |
             multiply category polynomials
                          |
             apply rho_0(n), sum n=0..120
                          |
          positive-weight RQMC estimate of Z_x
                          |
 observed energy E_x(S) - log(Z_x - 1)
```

### 4.1 Exact Gaussian identity

Let `v_S=sum_{j in S} phi_j`. The Hubbard--Stratonovich identity gives

```math
\exp\{\tfrac12\|v_S\|^2\}
=\mathbb E_{z\sim N(0,I)}\exp(z^\top v_S).
```

Using

```math
\sum_{j<k\in S}\phi_j^\top\phi_k
=\tfrac12\left\|\sum_{j\in S}\phi_j\right\|^2
-\tfrac12\sum_{j\in S}\|\phi_j\|^2,
```

define

```math
w_j(z,x)=\exp\{b_j(x)-\tfrac12\|\phi_j\|^2+\phi_j^\top z\}.
```

Then the version-4 normaliser is exactly

```math
Z_x=\mathbb E_{z\sim N(0,I)} f_x(z),
\qquad
f_x(z)=\sum_{n=0}^{120}e^{-\rho_0(n)}A_{x,n}(z).
```

For affinity row `c`, the exact row polynomial is

```math
G_{x,c}(u;z)=\sum_{r=0}^{120}
e^{-\rho_c {r\choose2}}
e_r(\{w_j(z,x):j\in c\})u^r,
```

and `A_{x,n}(z)` is the degree-`n` coefficient of the product over all row polynomials.
The elementary symmetric polynomials and category-polynomial products are computed by
vectorised recursions. Therefore the discrete sum over baskets is exact for each `z`; no
basket enumeration or product subsampling is used.

This is the scaling insight: catalogue size affects the vectorised ESP work, while the only
remaining numerical integral has dimension `Kz=32`, not 5,455.

### 4.2 Current RQMC rule

The outer Gaussian expectation uses a size-stratified, mode-centred, scrambled-Sobol rule
with positive weights:

- training uses 8 total one-mode nodes split across four independent scrambles, or 16 total
  nodes when a second mode is retained;
- each training update refreshes the scramble seed;
- six coarse size bands plus the full size sum are screened with three vectorised fixed-point
  mode steps;
- a second proposal mode is retained only when it is within 4 pointwise nats of the first
  and separated by at least 1 in latent space;
- the two-mode rule uses the balance-mixture importance denominator, so combining modes does
  not double-count overlap;
- flagged training trips whose replicate log-`Z` SE exceeds 0.015 are retried with an
  independent 64-node rule; only unresolved bad trips skip that optimizer update;
- checkpoint evaluation uses 128 total one-mode nodes across four fixed scrambles (256 total
  for a retained two-mode rule), seed 2,000,003;
- node evaluation is chunked in blocks of 32.

Proposal centres and frames are detached; gradients still pass through every evaluation of
the model integrand. The current run uses unit proposal scale in the eigenframe of
`Phi'Phi`, avoiding expensive finite-difference curvature probes whose measured scales were
effectively one.

### 4.3 Geometry and failure protection

The estimator can miss a remote large-basket mode if many interaction rows align. The
operator bound

```math
\left\|\Phi^\top x_S\right\|^2
\le \lambda_{max}(\Phi^\top\Phi)|S|
```

motivates projection to `lambda_max(Phi'Phi) <= 2`. This limits catalogue-wide clique
amplification without changing `rho_0` or the version-4 size law. Individual rows are also
capped at norm 0.96. Raising rank adds independent interaction directions; raising every
row norm would instead recreate the remote-mode failure.

The following diagnostics are evaluated at checkpoints:

- four-scramble replicate SE for log `Z`;
- an independent `N` versus `2N` log-`Z` gap;
- expected basket size and size variance;
- sampled-versus-analytic size agreement;
- operator norm, row norms and effective interaction rank;
- full data/support coverage and price elasticity.

At the frozen run155 iteration-1,000 checkpoint, the log-`Z` audit reported
`2*SE=0.0013` nats and an independent node-count gap of `0.0005` nats; all gates passed.
These checks validate the numerical integral at that checkpoint. They do not imply that
the model parameters have converged after only 0.152 epochs.

## 5. Training configuration

Run155 is a fresh fit (`seed=0`), not a warm start or resume. Its production schedule is:

| Setting | Value |
|---|---:|
| Batch size | 24 trips |
| Total updates | 30,000 |
| Learning rate, updates 1--19,999 | 0.002 |
| Learning rate, updates 20,000--25,999 | 0.001 |
| Learning rate, updates 26,000--30,000 | 0.0005 |
| Product-intercept learning rate | 0.0001 initially (`0.05x`) |
| Weight decay | `1e-5` |
| Gradient clipping | 2.0 |
| Checkpoint evaluation cadence | 200 updates |

Initial product intercepts use training-only incidence divided by store-assortment exposure.
`rho_0` is initialized against the empirical training size law, but remains trainable in the
original joint objective. `phi` and household taste start at scale 0.03.

The loss also contains declared calibration/regularisation terms for the size distribution,
reverse KL, expected size, product pooling, price-factor calibration and aggregate elasticity
(target `-0.121`). These stabilize an under-one-epoch early fit; they do not substitute an
empirical size factor for the joint normaliser.

The isolated timing probe at startup measured 2.069 seconds/update at batch 24. Baseline
audits running simultaneously can slow observed wall-clock time, so equal-update results are
not equal-compute results.

## 6. Evaluation metrics

### 6.1 Likelihood

For each observed basket the evaluator reports:

- set log likelihood per basket;
- set log likelihood per purchased line;
- `log P(n|x)` (size);
- `log P(S|n,x)` (conditional composition);
- conditional-units log likelihood, separately.

The exact identity

```math
\log P(S\mid x)=\log P(|S|\mid x)+\log P(S\mid |S|,x)
```

is used to localise a gap. In particular, an interaction loss in conditional composition is
not evidence of a bad size normaliser.

### 6.2 MRR and recall

The logged MRR is a complete-the-basket metric, not MRR@5, MRR@10 or MRR@20. For every
eligible basket one purchased item is hidden with a fixed random seed. Given the remaining
set `R`, each candidate `j` in the store assortment receives the exact fixed-final-size
incremental score

```math
score(j\mid R)=b_j(x)
+\phi_j^\top\sum_{k\in R}\phi_k
-\rho_{c(j)}\left[g(n_{c(j)}+1)-g(n_{c(j)})\right].
```

With `g(k)=k(k-1)/2`, the last increment is `rho_{c(j)} n_{c(j)}`. Already revealed items
are excluded. Rank is one plus the number of admissible candidates with a strictly larger
score, and

```math
MRR=\frac1T\sum_{t=1}^T\frac1{rank_t}.
```

The fixed-size `rho_0` term and the normaliser cancel, so MRR requires no QMC. Recall@5,
Recall@10 and Recall@20 are the fractions with rank at most 5, 10 and 20 respectively; they
are separate metrics and are not encoded in the scalar MRR.

The incidence probability

```math
\pi_j=\frac{\partial\log(Z-1)}{\partial b_j}=P(j\in S\mid x,S\ne\varnothing)
```

is available through `pi_quad`, and `sum_j pi_j=E[|S|]`. It is not what the current MRR
routine ranks: MRR uses the exact conditional completion score above.

## 7. Baselines

### 7.1 Which multinomial is the decisive ablation?

The decisive no-interaction comparison is the **strict nested version-4 multinomial**:

- instantiate the same `RaggedModel`;
- retain the same utility, `rho_0`, units block, support, optimizer, minibatches, penalties
  and evaluator;
- set `phi=0` and `rho_c=0`, and freeze them;
- train from a fresh initialization.

Its normaliser is exact because the Gaussian interaction disappears. This is the only
baseline that isolates the contribution of the two version-4 interaction blocks.

There is also a BEMB-style external multinomial in `baselines2.py`:

```math
P(S\mid n,x)=\frac{\prod_{j\in S}\exp b_j(x)}{e_n(\{\exp b_k(x)\})},
\qquad P(n)=P_{empirical}(n).
```

This is a useful external baseline but **not** the nested version-4 model. Its supplied,
context-free `P(n)` differs from version 4's
`P(n|x) proportional to exp(-rho_0(n)) A_n(x)`. It must not be used to claim that adding or
removing version-4 interactions helps.

### 7.2 Common external utility layer

Bernoulli, DPP, NDPP and SHOPPER use `LinearIndex`: product intercept, rank-32 household
taste, rank-8 price factors, item-specific display and mailer effects, rank-8 seasonality
and rank-4 store effects. Their absolute popularity initializer retains its level because
these models do not have version 4's `lambda`/`rho_0` common-shift gauge.

This layer covers the same observed no-recency covariates, but it is not byte-identical to
the current `RaggedModel` utility. In particular, its price bilinear form is unconstrained
and it lacks the main model's price common/relative split and centring transforms. Hence
these are broader architecture comparisons; the strict nested run remains the controlled
interaction ablation.

### 7.3 Baseline definitions

| Baseline | Probability law and interaction | Normalisation/support | Parameters | Main caveat |
|---|---|---|---:|---|
| Frequency | Training-only store-product incidence/exposure with Laplace smoothing | Independent Bernoulli odds, conditioned non-empty | 0 fitted | Diagnostic floor; the current paired JSON does not store a separate `n>120` tail audit for this row |
| Truncated Bernoulli | `P(S) proportional to product exp(b_j)` | Exact ESP sum over `1<=n<=120` | 383,541 | No item-item interaction |
| Symmetric DPP | `L=diag(exp(q))+VV'`, rank 16; determinant induces repulsion | Exact determinant-lemma normaliser, conditioned non-empty | 470,821 | Standard PSD DPP is primarily repulsive |
| NDPP | `L=D+VV'+BCB'`, with rank-16 symmetric and 8 skew 2x2 blocks | Exact determinant-lemma normaliser, conditioned non-empty | 558,109 | More general attraction/repulsion, but a different interaction geometry |
| SHOPPER | Sequential softmax over remaining products with running-mean interaction and checkout | Forced checkout at 120; set probability sums over orderings | 558,102 | Set likelihood uses an ordering estimator above size 6 |
| Strict nested version-4 | Full version-4 model with `phi=rho_c=0` | Exact additive version-4 joint normaliser on `1<=n<=120` | Same allocated architecture; interaction blocks frozen | Controlled no-interaction ablation |

For DPP and NDPP the generative law formally permits cardinalities above 120. The comparison
keeps them only after a per-trip Chernoff bound shows that omitted mass is negligible. At
iteration 1,000 the maximum bounds were `1.416e-35` for DPP and `4.458e-33` for NDPP.

SHOPPER assigns a probability to an ordered sequence and checkout. Its set probability is

```math
P(S)=\sum_{\pi\in permutations(S)}P(\pi)
=|S|!\,\mathbb E_{\pi\sim Uniform}P(\pi).
```

Evaluation sums all orderings exactly through basket size 6 and uses 8,192 uniform sampled
orderings for larger baskets. The probability estimator is unbiased, while its logarithm
is biased downward by Jensen's inequality. The 512-versus-8,192 ordering audit moved the
aggregate score by only about 0.0002 nats. This is a faithful local reimplementation, not a
claim to reproduce every detail of a paper's variational training procedure.

## 8. Fair-comparison protocol

The matched audits enforce the following:

1. Fresh initialization; no resumption from the 400-update checkpoints and no transferred
   backbone.
2. Exactly 400 or 1,000 optimizer updates for every fitted row.
3. Batch size 24, learning rate 0.002, weight decay `1e-5`, seed 0.
4. Identical supported training split with SHA-256
   `9728da264f4fa183b35ae6db693c1a9e9f53aeb1fe4c0aa2ac67ec0578250ff4`.
5. Identical ordered 384-trip validation manifest.
6. Full 5,455-product universe, affinity partition and basket support.
7. Paired per-trip differences and paired standard errors.
8. Set likelihood alone for the interaction claim; units excluded.

Equal updates give equal minibatch exposure (24,000 trip presentations at 1,000 updates,
about 0.152 training epochs). They do not give equal wall-clock time, and 1,000 updates do
not constitute convergence for these models.

The zero-parameter frequency row is naturally exempt from the update count. Its score and
paired SE are recorded in `paper/version4_estimator_audit.html`; unlike the four fitted
external baselines, it is not stored as a format-3 checkpoint row in
`v3_other_baselines_matched*.json`. It should therefore be read as a diagnostic floor, not
as one of the provenance-complete fitted-model comparisons.

## 9. Matched results

Higher (less negative) log likelihood is better.

### 9.1 Exactly 400 fresh updates

| Model | Validation set LL / basket | Full minus baseline, paired SE |
|---|---:|---:|
| Strict nested version-4 | -56.32414 | -0.00455 +/- 0.01081 |
| **Full version-4** | **-56.32869** | -- |
| SHOPPER | -56.38268 | +0.05399 +/- 0.02994 |
| Frequency | -58.60414 | +2.27545 +/- 0.53995 |
| NDPP | -59.54138 | +3.21269 +/- 0.46010 |
| Truncated Bernoulli | -59.57427 | +3.24558 +/- 0.52975 |
| Symmetric DPP | -59.77394 | +3.44525 +/- 0.50780 |

At 400 updates, full and strict nested are statistically tied. The full model's conditional
composition advantage is `+0.00372 +/- 0.01035` nats, offset by a size difference of
`-0.00827 +/- 0.00564`.

### 9.2 Exactly 1,000 fresh updates

| Model | Validation set LL / basket | Full minus baseline, paired SE |
|---|---:|---:|
| **SHOPPER** | **-55.15249** | -0.84921 +/- 0.08511 |
| Strict nested version-4 | -55.64096 | -0.36075 +/- 0.04277 |
| Full version-4 | -56.00170 | -- |
| NDPP | -58.57987 | +2.57817 +/- 0.45442 |
| Frequency | -58.60414 | +2.60244 +/- 0.53147 |
| Truncated Bernoulli | -58.68744 | +2.68573 +/- 0.51363 |
| Symmetric DPP | -58.74187 | +2.74017 +/- 0.50467 |

The full-versus-strict-nested decomposition at 1,000 updates is:

| Component / basket | Full | Strict nested | Paired full minus nested |
|---|---:|---:|---:|
| Joint set | -56.00170 | -55.64096 | -0.36075 +/- 0.04277 |
| Size | -3.03390 | -3.01856 | -0.01534 +/- 0.00749 |
| Conditional composition | -52.96780 | -52.62240 | -0.34540 +/- 0.04041 |

The early deficit is predominantly conditional composition, not size/log-`Z` error. At this
budget the interaction blocks have slowed optimization and have not yet earned their extra
capacity. The nesting theorem says the optimised training objective of the full family
cannot be worse; it does not make this 0.152-epoch held-out result impossible. A mature
convergence comparison must match data exposure and optimisation quality, then separately
check generalisation and interaction regularisation.

For context, run155's own checkpoint series was:

| Update | Set LL / basket | MRR | Training epochs |
|---:|---:|---:|---:|
| 0 | -56.411 | 0.0248 | 0.000 |
| 400 | -56.329 | 0.0252 | 0.061 |
| 1,000 | -56.002 | 0.0329 | 0.152 |
| 1,400 | -55.668 | 0.0350 | 0.213 |
| 1,600 | -55.533 | 0.0371 | 0.244 |

The 1,200--1,600 rows are live production progress, not equal-budget baseline comparisons.

## 10. Checkpoints, logs and reproducibility

The frozen full-model comparison checkpoint is
`out/v3_run155_v4_original_stochastic_full_at1000.pt`, SHA-256
`6956ecd995048d8596075019bb336aff3bef58edcaadff65d0d8749de69344fd`.

The matched iteration-1,000 baseline checkpoints are:

```text
out/v3_run158_v4_multinomial_fair1000.pt
out/v3_bl_verified_bernoulli_iter1000.pt
out/v3_bl_verified_dpp_iter1000.pt
out/v3_bl_verified_ndpp_iter1000.pt
out/v3_bl_verified_shopper_iter1000.pt
```

Result and per-trip audit artifacts:

```text
out/v3_run155_vs_multinomial_matched400.json
out/v3_run155_vs_multinomial_matched400_per_trip.npz
out/v3_other_baselines_matched400.json
out/v3_other_baselines_matched400_per_trip.npz
out/v3_run155_vs_multinomial_matched1000.json
out/v3_run155_vs_multinomial_matched1000_per_trip.npz
out/v3_other_baselines_matched1000.json
out/v3_other_baselines_matched1000_per_trip.npz
```

Training logs:

```text
out/v3_run155_v4_original_stochastic_full.log
out/v3_run158_v4_multinomial_fair1000.log
out/v3_bl1000_bernoulli.log
out/v3_bl1000_dpp.log
out/v3_bl1000_ndpp.log
out/v3_bl1000_shopper.log
```

Key source files:

```text
paper/version4.html                         frozen theory and original experiment record
paper/version4_estimator_audit.html         estimator and fair-baseline audit
paper/sampling_version4_theory.md           mathematical estimator analysis
scripts/v3/ragged.py                        set model, exact ESP kernel and QMC normaliser
scripts/v3/fit.py                           training, checkpointing, MRR and diagnostics
scripts/v3/run_version4_full.sh              guarded production command
scripts/v3/baselines.py                     common index, Bernoulli, DPP, frequency
scripts/v3/baselines2.py                    NDPP, SHOPPER, external BEMB adaptation
scripts/v3/train_baseline_verified.py        provenance-complete fresh baseline trainer
scripts/v3/audit_multinomial_fair.py         strict nested paired audit
scripts/v3/audit_other_baselines_fair.py     other paired audits and support checks
scripts/v3/test_qmc.py                       exact small-universe normaliser tests
scripts/v3/test_baselines2.py                baseline probability/support tests
```

The current baseline unit suite passes 6 tests. The QMC suite has 21 exact small-catalogue
tests covering normaliser values, gradients, sizes, incidence, common-shift response and
rollout consistency. These tests establish implementation identities on enumerable cases;
the fixed high-node audits establish numerical accuracy on selected full-catalogue trips.

## 11. What the present evidence does and does not establish

Established:

- the current main run implements the original version-4 joint law on complete support;
- the old Smolyak product/rank restriction has been removed;
- the discrete catalogue sum is exact conditional on the auxiliary Gaussian draw;
- QMC integration error is measured and small at the frozen 1,000-update checkpoint;
- current results beat Bernoulli, DPP and NDPP at equal 1,000-update exposure and beat the
  zero-parameter frequency diagnostic on the matched manifest;
- current results do not beat strict nested version-4 or SHOPPER at that early budget.

Not yet established:

- convergence of any fitted model after only 0.152 epochs;
- a mature held-out advantage from version-4 interactions;
- published-SOTA status for this local SHOPPER implementation;
- that equal optimizer updates are an equal-compute or equal-convergence comparison.

The next decisive experiment is therefore not another change to the foundation theorem. It
is a convergence-controlled comparison of the full and strict nested version-4 models,
using the same complete-support estimator and identical fresh lineages, with training loss,
validation size/composition likelihood and interaction norms audited together.
