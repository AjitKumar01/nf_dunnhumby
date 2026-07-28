"""
Stage 13 -- Act on the placebo results.

Produces out/placebo_category_status.csv (one row per category, with which placebo
tests it failed) and re-aggregates the held-out fit statistics over the subset of
categories that survive, so the headline numbers can be read both ways.

Note on interpretation: the models were trained on all 62 categories.  This is an
*evaluation* restriction, not a retrained model.  To retrain on the clean subset,
re-run  02_select_sample.py --exclude-placebo-failures  and the stages after it.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out")


def log(m):
    print(f"[13] {m}", flush=True)


def main():
    pt = pd.read_csv(os.path.join(OUT, "placebo_tests.csv"))
    placebos = [t for t in pt.test.unique() if not t.startswith("actual")]
    p = pt[pt.test.isin(placebos)]

    status = p.pivot_table(index=["group_id", "COMMODITY_DESC"], columns="test",
                           values="p").reset_index()
    for c in placebos:
        status[f"fail_{c.replace(' ', '_')}"] = (status[c] < 0.01)
    fail_cols = [c for c in status.columns if c.startswith("fail_")]
    status["n_failed"] = status[fail_cols].sum(axis=1)
    status["fails_any"] = status.n_failed > 0
    status["fails_random_only_tests"] = status[[c for c in fail_cols if "random" in c]].any(axis=1)
    status["fails_backward_tests"] = status[[c for c in fail_cols if "backward" in c]].any(axis=1)

    real = pt[pt.test == "actual (all items)"][["group_id", "coef", "p"]].rename(
        columns={"coef": "actual_price_coef", "p": "actual_p"})
    status = status.merge(real, on="group_id", how="left")
    status.to_csv(os.path.join(OUT, "placebo_category_status.csv"), index=False)

    n = len(status)
    log(f"{n} categories: {int(status.fails_any.sum())} fail at least one placebo at 1%, "
        f"{int(status.fails_random_only_tests.sum())} fail a fully-decorrelated (random) "
        f"placebo, {int(status.fails_backward_tests.sum())} fail a backward placebo")
    clean = status.loc[~status.fails_any, "group_id"].tolist()
    clean_random = status.loc[~status.fails_random_only_tests, "group_id"].tolist()
    log(f"strictly clean categories: {len(clean)};  clean on the random placebo: "
        f"{len(clean_random)}")

    # ------------------------------- re-aggregate held-out fit on the clean subsets
    per_cat_path = os.path.join(OUT, "evaluation_per_category_raw.csv")
    if not os.path.exists(per_cat_path):
        log("evaluation_per_category_raw.csv not found; run 07_evaluate.py first")
        return
    pc = pd.read_csv(per_cat_path)

    def agg(sub, label):
        rows = []
        for lab, g in sub.groupby("label"):
            w = g.purchases
            rows.append({"label": lab, "subset": label,
                         "categories": int(len(g)),
                         "test_purchases": int(w.sum()),
                         "test_loglik": float(np.average(g.loglik, weights=w)),
                         "test_mse": float(np.average(g.mse, weights=w))})
        return pd.DataFrame(rows)

    tables = [agg(pc, f"all ({pc.group_id.nunique()} categories)"),
              agg(pc[pc.group_id.isin(clean_random)],
                  f"passes random placebo ({len(clean_random)})"),
              agg(pc[pc.group_id.isin(clean)], f"passes every placebo ({len(clean)})")]
    tab = pd.concat(tables, ignore_index=True)
    tab = tab.sort_values(["subset", "test_loglik"], ascending=[True, False])
    tab.to_csv(os.path.join(OUT, "evaluation_placebo_subsets.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\nHeld-out fit re-aggregated over placebo-surviving categories\n"
          + tab.round(4).to_string(index=False))

    summ = {
        "n_categories": n,
        "fail_any": int(status.fails_any.sum()),
        "fail_random": int(status.fails_random_only_tests.sum()),
        "fail_backward": int(status.fails_backward_tests.sum()),
        "clean_all": clean, "clean_random": clean_random,
        "clean_all_names": status.loc[~status.fails_any, "COMMODITY_DESC"].tolist(),
    }
    with open(os.path.join(OUT, "placebo_followup.json"), "w") as f:
        json.dump(summ, f, indent=2)
    log("wrote out/placebo_category_status.csv, evaluation_placebo_subsets.csv, "
        "placebo_followup.json")


if __name__ == "__main__":
    main()
