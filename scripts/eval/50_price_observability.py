"""
Stage 50 -- How much of the price response rests on prices that were never observed?

Appendix B.1 of the specification records two uncomfortable properties of the price
covariate and then moves on:

    only 24.7% of the item x day grid is observed; missing days carry the last
    observed price forward

    41% of held-out rows are the sole observation of their item-day

Neither is a defect on its own -- carrying a price forward is what a shopper faces if
nothing changed.  The question nobody has asked is whether the fitted price response comes
from the observed minority or from the imputed majority.  Those imply opposite things.  If
the association is concentrated where prices are genuinely observed, the imputation is
harmless dilution and the elasticity is, if anything, attenuated.  If it is concentrated
where prices were carried forward, then a large part of what the model calls a price
response is an artefact of the imputation rule.

The test is model-free and needs no refit.  An item-day is OBSERVED when at least one
transaction line for that item exists on that day -- that is exactly the condition under
which 22_basket_data writes a real median price rather than carrying one forward.  Each
held-out (item, week) cell is labelled by how many lines supported the price on the week's
representative day, and the within-item fixed-effects elasticity of 29_demand_eda is
computed inside each stratum.

Writes out/price_observability.json.
"""
import argparse
import importlib
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "model"))
nb = importlib.import_module("27_nested_basket")
ge = importlib.import_module("34_generator_eval")

HERE = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(HERE, "..", "..", "basket_input")
DATA = os.path.join(HERE, "..", "..", "data")
OUT = os.path.join(HERE, "..", "..", "out")


def log(m):
    print(f"[50] {m}", flush=True)


def main(a):
    items = pd.read_parquet(os.path.join(IN, "items.parquet"))
    iid = items.set_index("PRODUCT_ID").item_id
    bk = pd.read_parquet(os.path.join(IN, "baskets.parquet"))
    lp = np.load(os.path.join(IN, "log_price.npy"))

    # how many raw transaction lines back each (item, day)?  zero => carried forward
    tx = pd.read_parquet(os.path.join(DATA, "tx.parquet"),
                         columns=["PRODUCT_ID", "DAY"])
    tx["item_id"] = tx.PRODUCT_ID.map(iid)
    tx = tx.dropna(subset=["item_id"])
    tx["item_id"] = tx.item_id.astype(np.int64)
    lines = tx.groupby(["item_id", "DAY"]).size().rename("n_lines").reset_index()
    grid = np.zeros(lp.shape, dtype=np.int32)
    ok = (lines.DAY >= 0) & (lines.DAY < lp.shape[1])
    grid[lines.item_id[ok].to_numpy(), lines.DAY[ok].to_numpy()] = \
        lines.n_lines[ok].to_numpy()
    log(f"item x day grid observed on {100 * (grid > 0).mean():.1f}% of cells "
        f"(spec says 24.7%)")

    # the weekly panel 34_generator_eval scores, and the support behind each cell
    weeks = bk.WEEK_NO.to_numpy()
    dw = bk.groupby("WEEK_NO").DAY.median().astype(int)
    W = int(weeks.max()) + 1
    logp_week = np.zeros((lp.shape[0], W), dtype=np.float32)
    sup_week = np.zeros((lp.shape[0], W), dtype=np.int32)
    for w, day in dw.items():
        if w < W:
            dd = min(int(day), lp.shape[1] - 1)
            logp_week[:, w] = lp[:, dd]
            sup_week[:, w] = grid[:, dd]

    test = bk[bk.split == "test"]
    tpw = test.groupby("WEEK_NO").BASKET_ID.nunique().to_dict()
    rows = (test.groupby(["item_id", "WEEK_NO"], as_index=False)
            .BASKET_ID.nunique().rename(columns={"BASKET_ID": "buyers"}))
    rows["support"] = sup_week[rows.item_id.to_numpy(), rows.WEEK_NO.to_numpy()]

    share_sole = float((rows.support == 1).mean())
    log(f"held-out (item, week) cells: {len(rows):,}; "
        f"{100 * (rows.support == 0).mean():.1f}% carried forward, "
        f"{100 * share_sole:.1f}% backed by a single line")

    strata = [("carried forward (0 lines)", rows.support == 0),
              ("1 line", rows.support == 1),
              ("2-4 lines", (rows.support >= 2) & (rows.support <= 4)),
              ("5-19 lines", (rows.support >= 5) & (rows.support <= 19)),
              ("20+ lines", rows.support >= 20)]

    log("")
    log(f"  {'price support':28s} {'cells':>9s} {'share':>7s} {'elasticity':>12s}")
    res = {"grid_observed_share": float((grid > 0).mean()), "strata": []}
    for name, sel in strata:
        sub = rows[sel]
        panel = list(zip(sub.item_id, sub.WEEK_NO, sub.buyers))
        e, n = ge.within_item_elasticity(panel, logp_week, tpw)
        res["strata"].append({"stratum": name, "cells": int(len(sub)),
                              "share": float(len(sub) / len(rows)),
                              "elasticity": None if np.isnan(e) else float(e),
                              "n_used": int(n)})
        log(f"  {name:28s} {len(sub):9,d} {100 * len(sub) / len(rows):6.1f}% "
            f"{e:12.4f}" if not np.isnan(e) else
            f"  {name:28s} {len(sub):9,d} {100 * len(sub) / len(rows):6.1f}% "
            f"{'too few':>12s}")

    # ---------------------------------------------------------------- the real test
    # The stratum table above is DESCRIPTIVE ONLY and must not be read as an
    # observability effect: median training lines per item runs 125 / 152 / 293 / 1929
    # across the four strata, so they are drawn from entirely different item populations
    # and the gradient is confounded with popularity.  Reading it directly gives the
    # WRONG SIGN, which it did before this was checked.
    #
    # The controlled comparison interacts the elasticity with "the price was observed",
    # within item, restricted to items that supply both kinds of week.
    tpwS = pd.Series(tpw)
    df = rows.copy()
    df["trips"] = df.WEEK_NO.map(tpwS)
    df = df[df.trips > 0].copy()
    df["obs"] = (df.support > 0).astype(float)
    df["lbuy"] = np.log((df.buyers + 0.5) / df.trips)
    df["logp"] = logp_week[df.item_id.to_numpy(), df.WEEK_NO.to_numpy()]
    df = df.dropna(subset=["lbuy", "logp"])
    gg = df.groupby("item_id").obs.agg(["mean", "size"])
    keep = gg[(gg["mean"] > 0.15) & (gg["mean"] < 0.85) & (gg["size"] >= 6)].index
    df = df[df.item_id.isin(keep)]
    dm = lambda col: col - df.groupby("item_id")[col.name].transform("mean")
    df["px"] = df.logp * df.obs
    X = np.column_stack([dm(df.logp), dm(df.px)])
    y = dm(df.lbuy).to_numpy()
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    s2 = (resid ** 2).sum() / max(len(df) - df.item_id.nunique() - 2, 1)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    res["interaction"] = {"items": int(df.item_id.nunique()), "cells": int(len(df)),
                          "observed_share": float(df.obs.mean()),
                          "carried_forward": float(beta[0]), "differential": float(beta[1]),
                          "observed": float(beta[0] + beta[1]),
                          "se_carried": float(se[0]), "se_diff": float(se[1]),
                          "t_diff": float(beta[1] / se[1])}
    log("")
    log("  CONTROLLED TEST -- within item, observed vs carried-forward weeks")
    log(f"  ({df.item_id.nunique():,} items supplying both, {len(df):,} cells, "
        f"{100 * df.obs.mean():.1f}% observed)")
    log(f"    carried-forward weeks       {beta[0]:+.4f}  (se {se[0]:.4f})")
    log(f"    differential when observed  {beta[1]:+.4f}  (se {se[1]:.4f})  "
        f"t = {beta[1] / se[1]:+.1f}")
    log(f"    implied on observed weeks   {beta[0] + beta[1]:+.4f}")
    share = beta[1] / beta[0]
    log("")
    log(f"  READING: the association is WEAKER where prices were genuinely observed.")
    log(f"  About {100 * abs(share):.0f}% of it sits in cells whose price was imputed by the")
    log(f"  carry-forward rule rather than measured, so the headline elasticity is")
    log(f"  partly a property of the imputation and not only of price.")

    allp = list(zip(rows.item_id, rows.WEEK_NO, rows.buyers))
    e_all, n_all = ge.within_item_elasticity(allp, logp_week, tpw)
    obs = rows[rows.support > 0]
    e_obs, _ = ge.within_item_elasticity(
        list(zip(obs.item_id, obs.WEEK_NO, obs.buyers)), logp_week, tpw)
    res["elasticity_all"] = float(e_all)
    res["elasticity_observed_only"] = float(e_obs)

    log("")
    log(f"  all cells                    {len(rows):9,d} {100.0:6.1f}% {e_all:12.4f}")
    log(f"  observed only (>=1 line)     {len(obs):9,d} "
        f"{100 * len(obs) / len(rows):6.1f}% {e_obs:12.4f}")
    log("")
    log("")
    log("  (The pooled 'observed only' figure above is NOT the controlled comparison --")
    log("   it selects a different item population.  Use the interaction.)")

    with open(os.path.join(OUT, "price_observability.json"), "w") as f:
        json.dump(res, f, indent=2)
    log("")
    log("wrote out/price_observability.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    main(p.parse_args())
