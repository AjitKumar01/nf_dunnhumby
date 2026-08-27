#!/usr/bin/env bash
# Evaluate a checkpoint: likelihood, ranking, price counterfactuals, personalisation,
# segmentation, and segment-level generation.
#
# The checkpoint carries model_flags (price_soft, price_ref, poly_degree).  Every script
# here loads through evalall.load_any, which restores them -- do NOT load a checkpoint with
# a bare load_state_dict, because gamma = +0.0207 is a valid price coefficient AND a valid
# softplus pre-image, and guessing wrong scales the price block by 34x.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT/src/basket"

export V3_AFFINITY=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

CKPT="${1:?usage: evaluate.sh <checkpoint.pt>}"
# absolute, relative, or a bare name -- src/basket/paths.py resolves all three

echo "=== ranking: MRR and MRR@k ==="
python3 -u eval_mrr_cutoffs.py --ckpt "$CKPT" --n-trips 384 --nmax 120 --R 120 \
        --cutoffs 5 10 20

echo
echo "=== counterfactual / personalisation / segmentation / generation ==="
python3 -u downstream.py --ckpt "$CKPT" --n-trips 192 --n-pers 96 --n-gen 32 --k 5 \
        --out "${CKPT%.pt}_downstream.json"
