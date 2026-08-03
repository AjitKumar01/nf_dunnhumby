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

step "18-20 substitution kernel: EDA, model, test, simulator"
python3 18_substitution_eda.py   > $L/log_sub_eda.txt 2>&1 && echo "  category-variation EDA ok"
# Changes 1 and 3 together, plus the two ablations that attribute the gain, plus a
# second seed of each arm -- the effects are ~0.02 nats and run-to-run spread is
# ~0.01, so a single run of each cannot tell them apart.
for spec in "nf_split|--price-split" "nf_ks|--Ks 8" "nf_sub|--Ks 8 --price-split" \
            "nf_ctl|" "ctl_s1|--seed 1" "sub_s1|--Ks 8 --price-split --seed 1"; do
  lab=${spec%%|*}; extra=${spec#*|}
  # shellcheck disable=SC2086
  python3 05_train_nf.py --label "$lab" $TRAIN $extra > $L/log_$lab.txt 2>&1
  echo "  trained $lab"
done
python3 07_evaluate.py --labels nf nf_promo nf_nopool logit nf_ctl nf_split nf_ks nf_sub \
  --device cpu > $L/log_eval_sub.txt 2>&1
python3 19_substitution_test.py --labels nf nf_ks nf_sub > $L/log_sub_test.txt 2>&1 \
  && tail -8 $L/log_sub_test.txt
python3 20_simulate.py --label nf_sub > $L/log_simulate.txt 2>&1 && tail -6 $L/log_simulate.txt

step "21-24 the basket model: exploration, data, fit, embedding test"
python3 21_basket_eda.py     > $L/log_basket_eda.txt  2>&1 && echo "  basket EDA ok"
python3 22_basket_data.py    > $L/log_basket_data.txt 2>&1 && echo "  basket dataset ok"
BT="--tie-context --K 64 --l2 1e-2 --lr 0.005 --iters 9000 --eval-every 500"
for spec in "tied_k64_r|" "tied_noctx|--no-context" "tied_nostate|--no-state" \
            "tied_noprice|--no-price" "tied_notaste|--no-taste" "tied_s1|--seed 1"; do
  lab=${spec%%|*}; extra=${spec#*|}
  # shellcheck disable=SC2086
  python3 23_basket_model.py --label "$lab" $BT $extra > $L/log_$lab.txt 2>&1
  echo "  trained $lab"
done
python3 24_embedding_eval.py --labels tied_k64_r tied_noctx tied_nostate tied_noprice \
  tied_notaste tied_s1 --primary tied_k64_r > $L/log_embed_eval.txt 2>&1
tail -8 $L/log_embed_eval.txt

step "25-26 price causality: placebos on the data, then on the model"
python3 25_basket_placebo.py > $L/log_basket_placebo.txt 2>&1; tail -14 $L/log_basket_placebo.txt
# The price block needs its own, much lighter penalty: a single L2 tuned for the
# embedding shrinks the elasticity by an order of magnitude (see BASKET_MODEL.md 7.2).
# One configuration for everything.  Cosine decay matters: without it the validation
# sequence bounces ~0.03 nats and the saved checkpoint is a lottery, which is what made
# the embedding look like it traded off against the price coefficient.
CT="--tie-context --K 64 --l2 1e-2 --l2-price 1e-4 --lr 0.005 --lr-decay --iters 12000 --eval-every 1000"
# shellcheck disable=SC2086
python3 23_basket_model.py --label one    $CT                         > $L/log_one.txt    2>&1
# shellcheck disable=SC2086
python3 23_basket_model.py --label one_s1 $CT --seed 1                > $L/log_one_s1.txt 2>&1
# shellcheck disable=SC2086
python3 23_basket_model.py --label one_pl $CT --placebo-price permute > $L/log_one_pl.txt 2>&1
python3 24_embedding_eval.py --labels one one_s1 one_pl --primary one > $L/log_embed_eval.txt 2>&1
python3 26_price_causal.py --labels one one_s1 one_pl > $L/log_price_causal.txt 2>&1
grep -E "^\[26\]" $L/log_price_causal.txt | tail -18

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
