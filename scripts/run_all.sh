#!/usr/bin/env bash
# Full pipeline, in the order the results depend on each other.
#
#   ./run_all.sh              everything
#   SKIP_BASE=1 ./run_all.sh  reuse data/tx.parquet (stage 01 is the slow read)
#
# Stage 11 (placebo tests) deliberately runs before anything is concluded from the
# fitted models: it is what decides whether the identification window is credible.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p ../out ../figures

L=../out
TRAIN="--K 40 --Kp 20 --price-prior-var 0.25 --intercept-var 10 \
       --iters 3000 --eval-every 100 --iters2 3000 --eval-every2 250 --device cpu"

step () { echo; echo "=== $* ==="; }

step "01 build base"
[ "${SKIP_BASE:-0}" = "1" ] && echo "skipped" || python3 01_build_base.py

step "02 select sample"
python3 02_select_sample.py

step "03 model inputs"
python3 03_make_model_inputs.py

step "04 dunnhumby extras (display / mailer / coupons)"
python3 04_extras.py

step "10 price definition audit"
python3 10_price_definition_audit.py | tail -20

step "11 placebo tests -- run before believing any elasticity"
python3 11_placebo_tests.py                > $L/log_placebo.txt 2>&1; tail -12 $L/log_placebo.txt
python3 11_placebo_tests.py --no-week-fe --tag _paperspec > $L/log_placebo_paperspec.txt 2>&1

step "05 train"
for spec in "nf|" "logit|--homogeneous" "nf_nopool|--no-pool" \
            "nf_promo|--extras display mailer coupon"; do
  lab=${spec%%|*}; extra=${spec#*|}
  # shellcheck disable=SC2086
  python3 05_train_nf.py --label "$lab" $TRAIN $extra > $L/log_$lab.txt 2>&1
  echo "  trained $lab"
done

step "07 evaluate"
python3 07_evaluate.py --labels nf logit nf_nopool nf_promo --device cpu \
  > $L/log_eval.txt 2>&1; sed -n '/label  K/,+6p' $L/log_eval.txt

step "13 placebo follow-up"
python3 13_placebo_followup.py

step "09 model-free counterfactual checks"
python3 09_counterfactual_checks.py --label nf --device cpu > $L/log_checks.txt 2>&1
tail -22 $L/log_checks.txt

step "08 / 12 reports and figures"
python3 08_data_report.py > /dev/null
python3 12_preprocessing_figures.py

step "14-17 verification: recovery, C++ cross-check, embeddings, stores"
python3 14_verify_model.py       > $L/log_verify.txt 2>&1     && echo "  parameter recovery ok"
if command -v gsl-config >/dev/null; then
  bash run_bemb_loc.sh           > $L/log_bemb.txt 2>&1 || true
  python3 15_cpp_crosscheck.py   > $L/log_crosscheck.txt 2>&1 && echo "  C++ cross-check ok"
else
  echo "  GSL not installed -- skipping the C++ cross-check (brew install gsl)"
fi
python3 16_inspect_embeddings.py > $L/log_embeddings.txt 2>&1 && echo "  embeddings ok"
python3 17_store_diagnostics.py  > $L/log_stores.txt 2>&1     && echo "  store diagnostics ok"

step "retrain on the placebo-clean subset"
python3 02_select_sample.py --exclude-placebo-failures > $L/log_clean_select.txt 2>&1
python3 03_make_model_inputs.py --outdir model_input_clean > $L/log_clean_inputs.txt 2>&1
for spec in "nf_clean|" "logit_clean|--homogeneous"; do
  lab=${spec%%|*}; extra=${spec#*|}
  # shellcheck disable=SC2086
  python3 05_train_nf.py --label "$lab" --indir model_input_clean $TRAIN $extra \
    > $L/log_$lab.txt 2>&1
  echo "  trained $lab"
done
python3 07_evaluate.py --labels nf_clean logit_clean --indir model_input_clean \
  --tag _clean --device cpu > $L/log_eval_clean.txt 2>&1
sed -n '/label  K/,+3p' $L/log_eval_clean.txt

step "restore the default sample as the primary artefacts"
python3 02_select_sample.py > $L/log_restore.txt 2>&1
python3 03_make_model_inputs.py >> $L/log_restore.txt 2>&1
python3 04_extras.py >> $L/log_restore.txt 2>&1
tail -2 $L/log_restore.txt

echo; echo "=== done ==="
