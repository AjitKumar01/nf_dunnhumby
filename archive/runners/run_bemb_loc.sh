#!/usr/bin/env bash
# Run the authors' C++ product-choice model (src/bemb_loc) on the emitted
# dunnhumby files.  Requires GSL:  brew install gsl
#
# Flag choices and why:
#   -likelihood 3   within-group softmax.  The "group" is the category, so this is
#                   exactly the paper's stage-1 conditional choice: the softmax runs
#                   over the items of the category the purchase came from.
#   -price 20       20 latent price components -> the gamma_i . lambda_j term.
#   -userVec 3      per-user latent vectors theta_u.
#   -UC 9           the nine columns of obsUser.tsv (household demographics).
#   -days 0         no week trend in the product stage; the paper puts the week
#                   trend in the category stage only (app. 8.2.1).
#   -shuffle 0      as recommended in bemb_loc/README.txt for a product model.
#   -keepAbove 0    keep every item; the categories were already filtered upstream.
#
# ITERS defaults to 2000.  Held-out log-likelihood peaks around iteration 300 and
# decays steadily after; 2000 shows the whole shape in ~30 minutes.  The authors'
# 6000 adds an hour of pure overfitting and changes no conclusion.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The authors' source tree.  Defaults to a sibling of the repository; override
# with NF_NF_SRC (or SRC) if the clone lives somewhere else.
SRC="${SRC:-${NF_NF_SRC:-$HERE/../../nested-factorization/src/bemb_loc}}"
DAT="${DAT:-$HERE/../model_input}"
OUTD="${OUTD:-$HERE/../out/bemb}"
K="${K:-40}"
KP="${KP:-20}"   # must match 15_cpp_crosscheck.py's --Kp for the comparison to be fair

command -v gsl-config >/dev/null || { echo "GSL not found: brew install gsl"; exit 1; }

mkdir -p "$OUTD"
if [ ! -x "$OUTD/emb" ]; then
  echo "compiling bemb_loc ..."
  g++ -std=c++11 -O2 -w -o "$OUTD/emb" "$SRC/emb.cpp" $(gsl-config --cflags --libs)
fi

"$OUTD/emb" \
  -dir "$DAT" -outdir "$OUTD" \
  -K "$K" -price "$KP" -userVec 3 -UC 9 \
  -itemIntercept -shuffle 0 -likelihood 3 \
  -max-iterations "${ITERS:-2000}" -rfreq 100 -batchsize 5000 \
  -eta 0.005 -step_schedule 0 -saveCycle 2000 \
  -s2theta 1.0 -s2alpha 1.0 -s2beta 1.0 -s2gamma 1.0 -s2lambda 1.0 \
  -valTolerance 0 -valConsecutive 100000 \
  -keepAbove 0 -disableAutoLabel -label dunnhumby-stage1

echo
echo "Output in $OUTD/emb-dunnhumby-stage1:"
echo "  train.tsv / valid.tsv / test.tsv   per-iteration log-likelihood"
echo "  param_innerProducts.tsv            val1 = intercept + theta.alpha + observables"
echo "                                     val3 = gamma_u . beta_i  (the price coefficient)"
echo
echo "NOTE: the category stage cannot be run with this binary.  It feeds the"
echo "inclusive value IV_ict into the price slot, and IV varies across users,"
echo "whereas bemb_loc stores prices as a single item x session matrix.  Use"
echo "05_train_nf.py (which fits both stages) or the authors' TTFM build."
