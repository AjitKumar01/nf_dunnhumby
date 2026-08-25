#!/bin/sh
set -eu

# Exact format-2 continuation after the iteration-2000 LAPACK SVD crash. The checkpoint
# restores Adam, the learning-rate scheduler, minibatch RNG and QMC RNG. tee -a preserves
# the original log and appends the recovery record.
cd "$(dirname "$0")/../.."
V3_AFFINITY=1 python -u scripts/v3/fit.py \
  --label run155_v4_original_stochastic_full \
  --require-version4 1 \
  --resume out/v3_run155_v4_original_stochastic_full_best.pt \
  --K 32 --Kp 8 --Kz 32 --R 120 --nmax 120 \
  --iters 30000 --batch 24 --draws 16 \
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
  2>&1 | tee -a out/v3_run155_v4_original_stochastic_full.log
