#!/usr/bin/env bash
# Build every model input from the raw dunnhumby CSVs.  Idempotent; safe to re-run.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export V3_AFFINITY=1

echo "--- stage 1/4: raw CSVs -> data/ (price reconstruction) ---"
python3 "$ROOT/src/pipeline/01_build_base.py"

echo "--- stage 2/4: data/ -> basket_input/ (the modelling universe) ---"
python3 "$ROOT/src/pipeline/22_basket_data.py"
python3 "$ROOT/src/pipeline/23_promo_data.py"

echo "--- stage 3/4: basket_input/ -> v3_index_affinity.npz (ragged assortment index) ---"
cd "$ROOT/src/basket" && python3 -c "from data import build; build()"

echo "--- stage 4/4: auxiliary inputs the training flags reference ---"
# Both are committed, so this only needs running if you change the cuts upstream.
[ -f "$ROOT/basket_input/v3_beta_target.npz" ]  || python3 beta_target.py
[ -f "$ROOT/basket_input/v3_phimask_lift30.npy" ] || {
    python3 pairmask.py --k 30
    cp "$ROOT/basket_input/v3_phimask_k30.npy" "$ROOT/basket_input/v3_phimask_lift30.npy"
}
echo "done."
