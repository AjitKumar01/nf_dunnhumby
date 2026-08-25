#!/bin/sh
set -eu

# Matched fresh-start ablation for run155.  Every training and validation setting is the
# same through the requested iteration; only version-4's two interaction blocks are
# zeroed/frozen.  Optional arguments are ITERATIONS and LABEL.
audit_iterations=${1:-200}
audit_label=${2:-run156_v4_multinomial_fair200}
cd "$(dirname "$0")/../.."
V3_AFFINITY=1 python -u scripts/v3/fit.py \
  --label "$audit_label" \
  --require-version4 1 \
  --K 32 --Kp 8 --Kz 32 --R 120 --nmax 120 \
  --iters "$audit_iterations" --batch 24 --draws 16 \
  --lr 0.002 --cosine 0 --lr-milestones 20000,26000 --lr-gamma 0.5 \
  --eval-every 200 --n-val 384 --n-rec 192 --probe 10 --eval-initial 1 \
  --init-popularity 1 --init-rho0 1 --taste-init 0.03 \
  --lam-lr-scale 0.05 --lam-centre 1 --lam-project 1 --lam-sd-max 0 \
  --pi-project-every 0 --no-rec 1 --units 1 --wd 1e-5 \
  --phi-init 0.03 --phi-max 0.96 --phi-op-max 2.0 \
  --phi-whiten 0.5 --phi-centre 1 \
  --rho-c-floor -0.92 --rho-c-step-scale 0.05 \
  --pool-prod 1.45 --en-w 0.005 --rkl-w 10 --size-kl 1 \
  --beta-cal-w 0.1 --elast-w 20 --elast-target -0.121 \
  --qmc-n 8 --qmc-reps 4 --qmc-seed 0 \
  --qmc-refresh-every 1 --qmc-eval-n 128 \
  --qmc-step-se 0.015 --qmc-retry-n 64 --qmc-en-max 2.5 \
  --quad-probe -1 --quad-steps 2 --quad-chunk 32 \
  --qmc-size-bands 1 --qmc-size-steps 3 \
  --qmc-mode-logtol 4 --qmc-mode-sep 1 --qmc-mix-n 16 \
  --lz-gap 0.02 --lz-strikes 3 \
  --zero-phi 1 --zero-rho-c 1 \
  2>&1 | tee "out/v3_${audit_label}.log"
