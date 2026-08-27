#!/usr/bin/env bash
# Train the basket model from scratch, exactly as run413 was configured.
#
# run413 is the model this repo ships: price_soft with the category price reference,
# 30-product rank-4 interaction, Smolyak q=8 normaliser.  Its measured results are in
# README.md.  This script reproduces the CONFIGURATION; the published checkpoint was
# reached by three staged runs (see README "How the shipped model was reached"), because
# kappa and the price block converge far more slowly than the rest of the model.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT/src/basket"

export V3_AFFINITY=1            # 280-category affinity partition -- LOAD-BEARING, see README
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

LABEL="${1:-run_local}"
ITERS="${2:-40000}"

python3 -u fit.py \
    --label "$LABEL" --iters "$ITERS" --lr 0.002 --lr-milestones 25000,33000 \
    --K 32 --Kp 8 --Kz 4 --R 120 --adapt-draws 1 --allow-factored-ablation 0 \
    --aniso 2.0 --antithetic 0 --batch 24 --beta-cal-w 0.1 --beta-target "$ROOT"/basket_input/v3_beta_target.npz \
    --budget-f 1.0 --c-max 0.0 --cd 0 --cd-draws 0 --clip 2.0 --comp-ce-w 0.0 \
    --composition-stage 0 --cosine 0 --ctx-shrink 1.0 --draws 16 --elast-every 20 \
    --elast-target -0.121 --elast-w 20.0 --en-w 0.005 --ess-floor 0.3 --ess-floor-min 0.15 \
    --eval-every 500 --eval-initial 1 --factored-size 0 --freeze-rho0 0 --freeze-rho-c 0 \
    --gap-project 0.0 --init-popularity 0 --init-rho0 0 --interaction-stage 0 \
    --lam-centre 1 --lam-floor 0.0 --lam-lr-scale 0.05 --lam-project 1 --lam-q 0.9 \
    --lam-sd-max 0.0 --lam-target 0.85 --lam-up-max 1.15 --lr-floor 0.02 --lr-gamma 0.5 \
    --lz-gap 0.02 --lz-strikes 3 --min-keep 0.5 --mix-lam 1.0 --mix-scales-hi 2.0 \
    --mix-scales-lo 1.0 --mode-steps 1 --n-rec 192 --n-val 384 --neg-per-trip 64 \
    --nmax 120 --no-rec 1 --objective full --phi-centre 0 --phi-deg-cap 2.5 --phi-init 0.03 \
    --phi-l1 0.0 --phi-mask "$ROOT"/basket_input/v3_phimask_lift30.npy --phi-max 0.6 \
    --phi-op-max 2.0 --phi-pool sum --phi-step-scale 1.0 --phi-topk 0.0 --phi-whiten 0.0 \
    --pi-project-every 0 --poly-degree-tol 0.001 --pool-beta 0.0 --pool-ctx 0.0 \
    --pool-prod 1.45 --price-hinge-w 10.0 --price-soft 1 --probe 10 --proj-ema 1 \
    --pseudo 0 --qmc-en-max 0.0 --qmc-eval-n 0 --qmc-mix-n 0 --qmc-mode-logtol 8.0 \
    --qmc-mode-sep 1.0 --qmc-n 0 --qmc-refresh-every 0 --qmc-reps 4 --qmc-retry-n 0 \
    --qmc-seed 0 --qmc-size-bands 0 --qmc-size-steps 2 --qmc-step-se 0.0 --quad-chunk 32 \
    --quad-probe 0 --quad-q 8 --quad-steps 2 --reinit-interactions 0 --reinit-rho0-after-warm 0 \
    --require-version4 0 --rho0-curv 0.0 --rho-c-floor -0.92 --rho-c-step-scale 0.05 \
    --rkl-eps 0.0001 --rkl-w 10.0 --seed 0 --size-bands 3 --size-ipf-damp 0.5 \
    --size-ipf-steps 0 --size-ipf-trips 256 --size-kl 1.0 --size-stage 0 --taste-init 0.03 \
    --units 1 --var-damp 0.15 --var-project 0 --var-target -1.0 --var-w 0.0 --wd 1e-05 \
    --xi-shrink 0.0 --zero-phi 0 --zero-rho-c 0 --price-lr-scale 0.05 --poly-degree 32 \
    --kappa-lr-scale 5.0 --price-ref category --kappa-init 0.0 \
    --metrics-jsonl "$ROOT/out/${LABEL}_metrics.jsonl"
