#!/usr/bin/env bash
#
# Train the shipped model from scratch, in one run.
#
# This is the FINAL configuration.  The published checkpoint was reached across several
# runs only because the settings below were discovered one at a time; there is nothing
# staged about them, and starting here goes straight to the same place.  What matters and
# why is in docs/THEORY.md section 6 (price identification) -- in particular kappa, which
# moves ~1.4 units per 1,000 iterations and so is INITIALISED at the value the data implies
# rather than trained toward it.
#
#   usage:  ./src/run/train.sh [label] [iterations]
#   e.g.    OMP_NUM_THREADS=8 ./src/run/train.sh my_run 40000
#
# ~0.7 s/iteration on 8 CPU threads, so 40,000 iterations is about 8 hours.  Writes
# out/v3_<label>.pt, out/v3_<label>_best.pt and out/<label>_metrics.jsonl.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# The 280-category co-purchase affinity partition.  Omitting this silently trains a
# DIFFERENT model on 188 merchandiser commodities, and the two are not comparable
# (rho_c changes shape).  See docs/ARCHITECTURE.md.
export V3_AFFINITY=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

LABEL="${1:-run_local}"
ITERS="${2:-40000}"

cd "$ROOT/src/basket"
python3 -u fit.py \
    --label "$LABEL" --iters "$ITERS" \
    --lr 0.002 --lr-milestones 25000,33000 \
    --K 32 --Kp 8 --Kz 4 --R 120 --adapt-draws 1 --allow-factored-ablation 0 \
    --aniso 2.0 --antithetic 0 --batch 24 --beta-cal-w 0.1 --budget-f 1.0 --c-max 0.0 \
    --cd 0 --cd-draws 0 --clip 2.0 --comp-ce-w 0.0 --composition-stage 0 --cosine 0 \
    --ctx-shrink 1.0 --draws 16 --elast-every 20 --elast-target -0.121 --elast-w 20.0 \
    --en-w 0.005 --ess-floor 0.3 --ess-floor-min 0.15 --eval-every 500 --eval-initial 1 \
    --factored-size 0 --freeze-rho0 0 --freeze-rho-c 0 --gap-project 0.0 --init-popularity 0 \
    --init-rho0 0 --interaction-stage 0 --lam-centre 1 --lam-floor 0.0 --lam-lr-scale 0.05 \
    --lam-project 1 --lam-q 0.9 --lam-sd-max 0.0 --lam-target 0.85 --lam-up-max 1.15 \
    --lr-floor 0.02 --lr-gamma 0.5 --lz-gap 0.02 --lz-strikes 3 --min-keep 0.5 \
    --mix-lam 1.0 --mix-scales-hi 2.0 --mix-scales-lo 1.0 --mode-steps 1 --n-rec 192 \
    --n-val 384 --neg-per-trip 64 --nmax 120 --no-rec 1 --objective full --phi-centre 0 \
    --phi-deg-cap 2.5 --phi-init 0.03 --phi-l1 0.0 --phi-max 0.6 --phi-op-max 2.0 \
    --phi-pool sum --phi-step-scale 1.0 --phi-topk 0.0 --phi-whiten 0.0 --pi-project-every 0 \
    --poly-degree-tol 0.001 --pool-beta 0.0 --pool-ctx 0.0 --pool-prod 1.45 \
    --price-hinge-w 10.0 --price-soft 1 --probe 10 --proj-ema 1 --pseudo 0 \
    --qmc-en-max 0.0 --qmc-eval-n 0 --qmc-mix-n 0 --qmc-mode-logtol 8.0 --qmc-mode-sep 1.0 \
    --qmc-n 0 --qmc-refresh-every 0 --qmc-reps 4 --qmc-retry-n 0 --qmc-seed 0 \
    --qmc-size-bands 0 --qmc-size-steps 2 --qmc-step-se 0.0 --quad-chunk 32 \
    --quad-probe 0 --quad-q 8 --quad-steps 2 --reinit-interactions 0 --reinit-rho0-after-warm 0 \
    --require-version4 0 --rho0-curv 0.0 --rho-c-floor -0.92 --rho-c-step-scale 0.05 \
    --rkl-eps 0.0001 --rkl-w 10.0 --seed 0 --size-bands 3 --size-ipf-damp 0.5 \
    --size-ipf-steps 0 --size-ipf-trips 256 --size-kl 1.0 --size-stage 0 --taste-init 0.03 \
    --units 1 --var-damp 0.15 --var-project 0 --var-target -1.0 --var-w 0.0 \
    --wd 1e-05 --xi-shrink 0.0 --zero-phi 0 --zero-rho-c 0 --price-lr-scale 0.05 \
    --poly-degree 32 --kappa-lr-scale 5.0 \
    --price-ref category \
    --kappa-init 44 \
    --beta-target "$ROOT/basket_input/v3_beta_target.npz" \
    --phi-mask "$ROOT/basket_input/v3_phimask_lift30.npy" \
    --metrics-jsonl "$ROOT/out/${LABEL}_metrics.jsonl"

echo
echo "done.  evaluate with:  ./src/run/evaluate.sh v3_${LABEL}_best.pt"
