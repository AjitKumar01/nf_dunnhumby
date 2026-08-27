#!/usr/bin/env bash
# Build every model input from the raw dunnhumby CSVs.  Idempotent; safe to re-run.
#
#   NF_RAW_DIR   where the raw CSVs live (default: a sibling of this repository)
#   NF_ROOT      where data/ and basket_input/ are written (default: this repository)
#
# Stages 1-2 take tens of minutes, mostly parsing the 696 MB causal_data.csv.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export V3_AFFINITY=1

# Ask the code where things go, rather than assuming -- NF_ROOT may point elsewhere.
eval "$(python3 -c "
import sys; sys.path.insert(0, '$REPO/src/basket')
from paths import DATA, BI, RAW
print(f'DATA={DATA!r}'); print(f'BI={BI!r}'); print(f'RAW={RAW!r}')
")"
echo "raw CSVs : $RAW"
echo "data     : $DATA"
echo "inputs   : $BI"
for f in transaction_data.csv product.csv causal_data.csv; do
    [ -f "$RAW$f" ] || { echo "MISSING: $RAW$f  (set NF_RAW_DIR)" >&2; exit 1; }
done

echo "--- stage 1/4: raw CSVs -> data/  (transactions and reconstructed shelf price) ---"
python3 "$REPO/src/pipeline/01_build_base.py"

echo "--- stage 2/4: data/ -> basket_input/  (the modelling universe) ---"
python3 "$REPO/src/pipeline/22_basket_data.py"
python3 "$REPO/src/pipeline/23_promo_data.py"

# The affinity partition defines the model's 280-category row structure and is not derived
# by this pipeline -- it ships with the repository.  Stage it before building the index, or
# the index silently gets the default 188-commodity partition under the affinity filename.
if [ ! -f "$BI/items_affinity.parquet" ]; then
    cp "$REPO/basket_input/items_affinity.parquet" "$BI/"
    echo "staged items_affinity.parquet (280-category partition)"
fi

echo "--- stage 3/4: basket_input/ -> the cached ragged assortment index ---"
cd "$REPO/src/basket" && python3 -c "from data import build; build()"

echo "--- stage 4/4: auxiliary inputs the training flags reference ---"
# Both ship with the repository, so this only runs if you changed the cuts upstream
# or redirected NF_ROOT to a fresh tree.
if [ ! -f "$BI/v3_beta_target.npz" ]; then
    if [ -f "$REPO/basket_input/v3_beta_target.npz" ]; then
        cp "$REPO/basket_input/v3_beta_target.npz" "$BI/"
    else
        python3 beta_target.py
    fi
fi
if [ ! -f "$BI/v3_phimask_lift30.npy" ]; then
    if [ -f "$REPO/basket_input/v3_phimask_lift30.npy" ]; then
        cp "$REPO/basket_input/v3_phimask_lift30.npy" "$BI/"
    else
        python3 pairmask.py --k 30
        cp "$BI/v3_phimask_k30.npy" "$BI/v3_phimask_lift30.npy"
    fi
fi

echo
echo "done.  train with:  ./src/run/train.sh my_run 40000"
