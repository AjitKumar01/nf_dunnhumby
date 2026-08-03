"""
Stage 18 -- EDA for the substitution-kernel model (changes 1 and 3).

The category filters in 02_select_sample.py implement the paper's app. 8.1, and they
were designed for a model whose only cross-price channel is IIA.  Two of the paper's
five filters exist purely to protect that structure, and one of them (price
correlation) turns out to bind on nothing here.  A model that carries an explicit
substitution kernel psi_j . psi_k needs *different* variation, and a model that splits
price into base and promotional depth has *two* sources of it instead of one.

So this script re-asks the selection question for the new model, on all 285 candidate
categories rather than the 56 that survive today:

  1. how much own-price variation is there once base price and promotion depth are
     separated -- a category can be dead in one and alive in the other;
  2. how much *differential* within-category price movement is there -- pair-weeks in
     which item k moves and item j does not.  That is what identifies psi_j . psi_k,
     and nothing in the current pipeline measures it;
  3. how far the dropped categories actually sit from each threshold, so a filter that
     costs a lot of identifying variation can be loosened deliberately rather than
     inherited;
  4. whether unit demand -- the binding filter, 79 categories -- is violated by a
     little or a lot.

Writes out/substitution_eda.json, out/category_variation.csv and
figures/substitution_eda.png.  Changes nothing; 02_select_sample.py still decides.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "out")
FIG = os.path.join(HERE, "..", "figures")
RAW = os.path.join(os.environ.get(
    "NF_RAW_DIR",
    os.path.join(HERE, "..", "..", "dunnhumby_The-Complete-Journey",
                 "dunnhumby_The-Complete-Journey CSV")), "")


def log(m):
    print(f"[18] {m}", flush=True)


def herfindahl(v):
    v = np.asarray(v, dtype=float)
    s = v.sum()
    return float(((v / s) ** 2).sum()) if s > 0 else np.nan


def main(a):
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)

    tx = pd.read_parquet(os.path.join(DATA, "tx.parquet"),
                         columns=["household_key", "DAY", "WEEK_NO", "PRODUCT_ID",
                                  "STORE_ID", "BASKET_ID", "QUANTITY", "weekday",
                                  "unit_price", "base_price", "loyalty_price"])
    prod = pd.read_csv(RAW + "product.csv",
                       usecols=["PRODUCT_ID", "COMMODITY_DESC", "SUB_COMMODITY_DESC",
                                "BRAND", "MANUFACTURER"])
    tx = tx.merge(prod, on="PRODUCT_ID", how="left")
    log(f"{len(tx):,} lines, {tx.COMMODITY_DESC.nunique()} commodities")

    # ---------------------------------------------------------------- top items
    # Mirror 02's choice-set rule so the numbers are comparable: top J items by the
    # number of distinct trips that bought them.
    trips_per_item = (tx.groupby(["COMMODITY_DESC", "PRODUCT_ID"])
                      .BASKET_ID.nunique().rename("n_trips").reset_index())
    trips_per_item["rank"] = (trips_per_item.groupby("COMMODITY_DESC").n_trips
                              .rank(ascending=False, method="first"))
    top = trips_per_item[trips_per_item["rank"] <= a.top_j].copy()
    top_ids = set(top.PRODUCT_ID)
    log(f"top-{a.top_j} choice sets: {len(top_ids):,} items across "
        f"{top.COMMODITY_DESC.nunique()} categories")

    # ------------------------------------------------- weekly price panels, both
    # Base price and promotion depth are separate objects (see PREPROCESSING.md 1).
    # A category dead in one can be alive in the other, which the single-price
    # filter cannot see.
    t = tx[tx.PRODUCT_ID.isin(top_ids)]
    pw = (t.groupby(["PRODUCT_ID", "WEEK_NO"])
          .agg(price=("unit_price", "median"),
               base=("base_price", "median"),
               n=("unit_price", "size")).reset_index())
    pw["promo"] = (1.0 - pw.price / pw.base).clip(lower=0.0)
    pw = pw.merge(prod[["PRODUCT_ID", "COMMODITY_DESC"]], on="PRODUCT_ID", how="left")

    # week-to-week movement, per item
    pw = pw.sort_values(["PRODUCT_ID", "WEEK_NO"])
    for col in ["price", "base", "promo"]:
        pw[f"d_{col}"] = pw.groupby("PRODUCT_ID")[col].diff()
    pw["consecutive"] = pw.groupby("PRODUCT_ID").WEEK_NO.diff() == 1
    mv = pw[pw.consecutive].copy()

    rows = []
    for cat, g in mv.groupby("COMMODITY_DESC"):
        items = g.PRODUCT_ID.nunique()
        if items < 2:
            continue
        moved_p = (g.d_price.abs() > 0.01)
        moved_b = (g.d_base.abs() > 0.01)
        moved_pr = (g.d_promo.abs() > 0.02)
        rows.append({
            "COMMODITY_DESC": cat, "n_top_items": items, "item_weeks": len(g),
            "share_weeks_price_moves": float(moved_p.mean()),
            "share_weeks_base_moves": float(moved_b.mean()),
            "share_weeks_promo_moves": float(moved_pr.mean()),
            # the key one: does splitting price reveal variation the combined
            # measure hides?  A category whose base never moves can still be rich
            # in promotional variation, and vice versa.
            "share_weeks_either_moves": float((moved_b | moved_pr).mean()),
            "n_items_base_moves": int(g[moved_b].PRODUCT_ID.nunique()),
            "n_items_promo_moves": int(g[moved_pr].PRODUCT_ID.nunique()),
        })
    var = pd.DataFrame(rows).set_index("COMMODITY_DESC")
    log(f"price-variation table built for {len(var)} categories")

    # --------------------------------------- differential within-category movement
    # What identifies psi_j . psi_k: weeks where k's price moves and j's does not.
    # Under the current model this variation is unused; under the kernel it is the
    # whole point, so it should drive selection.
    diff_rows = []
    for cat, g in mv.groupby("COMMODITY_DESC"):
        piv = g.pivot_table(index="WEEK_NO", columns="PRODUCT_ID",
                            values="d_price", aggfunc="first")
        if piv.shape[1] < 2 or piv.shape[0] < 4:
            continue
        M = (piv.abs() > 0.01).to_numpy()          # [weeks, items] did it move
        n_items = M.shape[1]
        # Ordered pairs (j, k), j != k: weeks where k moved and j did not.  The
        # double loop over pairs is O(items^2) per category and dominated the
        # runtime; it collapses to a closed form.  With m_w items moving in week w
        # out of n, the count of (moved, did-not-move) ordered pairs that week is
        # m_w * (n - m_w), so the total is just the sum of that over weeks.
        m_w = M.sum(1)
        pair_n = int((m_w * (n_items - m_w)).sum())
        # how correlated are the movements?  If every item moves together the
        # kernel is not identified no matter how much movement there is.
        with np.errstate(invalid="ignore"):
            C = np.corrcoef(piv.fillna(0.0).to_numpy().T)
        iu = np.triu_indices_from(C, k=1)
        diff_rows.append({
            "COMMODITY_DESC": cat,
            "differential_cells": pair_n,
            "differential_per_pair": pair_n / max(n_items * (n_items - 1), 1),
            "mean_abs_move_corr": float(np.nanmean(np.abs(C[iu]))) if len(iu[0]) else np.nan,
        })
    diff = pd.DataFrame(diff_rows).set_index("COMMODITY_DESC")
    var = var.join(diff)
    log(f"differential-movement table built for {len(diff)} categories")

    # ------------------------------------------------------- unit-demand distance
    # The binding filter today (79 categories).  Measure how far each category sits
    # from the threshold rather than just pass/fail.
    # A lambda with .isin over ~1.5M groups dominated the runtime; flag membership
    # once as a column and let the grouped nunique do the work.
    tx["_top_id"] = tx.PRODUCT_ID.where(tx.PRODUCT_ID.isin(top_ids))
    cat_trip = (tx.groupby(["household_key", "DAY", "COMMODITY_DESC"])
                .agg(n_items=("PRODUCT_ID", "nunique"),
                     n_top=("_top_id", "nunique"))
                .reset_index())
    ud = cat_trip.groupby("COMMODITY_DESC").agg(
        cat_trips=("n_items", "size"),
        multi_any=("n_items", lambda s: float((s > 1).mean())),
        multi_top=("n_top", lambda s: float((s > 1).mean())))
    var = var.join(ud)

    # size
    size = (tx.groupby("COMMODITY_DESC")
            .agg(n_items_all=("PRODUCT_ID", "nunique")))
    var = var.join(size)
    trips_top = top.groupby("COMMODITY_DESC").n_trips.sum().rename("trips_top")
    var = var.join(trips_top)

    # existing verdict, for comparison
    # Existing verdict, for comparison.  02_select_sample.py writes an *empty*
    # drop_reason for the categories it keeps and only relabels it "KEPT" for the
    # value_counts it prints, so an empty cell here means kept -- but only for
    # categories that reached the audit at all.  Anything absent from the audit was
    # removed upstream (random-weight screen, holiday weeks) and is neither.
    fa = os.path.join(DATA, "filter_audit.csv")
    if os.path.exists(fa):
        cur = pd.read_csv(fa).set_index("COMMODITY_DESC")[["drop_reason"]]
        cur["scored"] = True
        var = var.join(cur)
        var["scored"] = var.scored.fillna(False).astype(bool)
        var["drop_reason"] = var.drop_reason.fillna("")
        var["kept_today"] = var.scored & var.drop_reason.eq("")
        var.loc[var.kept_today, "drop_reason"] = "KEPT"
        var.loc[~var.scored, "drop_reason"] = "not audited (removed upstream)"
    else:
        var["scored"], var["kept_today"] = False, False
    var.to_csv(os.path.join(OUT, "category_variation.csv"))

    # ------------------------------------------------------------------ findings
    r = {"n_categories_scored": int(var.scored.sum()),
         "n_categories_measured": int(len(var))}
    kept = var[var.kept_today]
    drop = var[var.scored & ~var.kept_today]
    r["n_kept_today"] = int(len(kept))
    log(f"measured {len(var)} categories; {int(var.scored.sum())} reached the audit, "
        f"{len(kept)} kept today, {len(drop)} dropped by a filter")

    log("")
    log("--- does splitting price reveal variation the single measure hides?")
    for name, d in [("kept today", kept), ("dropped today", drop)]:
        if not len(d):
            continue
        log(f"  {name:15s}: price moves {d.share_weeks_price_moves.median():.3f}, "
            f"base {d.share_weeks_base_moves.median():.3f}, "
            f"promo {d.share_weeks_promo_moves.median():.3f}, "
            f"either {d.share_weeks_either_moves.median():.3f}  (medians)")
    r["variation_medians"] = {
        n: {c: float(d[c].median()) for c in
            ["share_weeks_price_moves", "share_weeks_base_moves",
             "share_weeks_promo_moves", "share_weeks_either_moves"]}
        for n, d in [("kept", kept), ("dropped", drop)] if len(d)}

    log("")
    log("--- differential within-category movement (what identifies the kernel)")
    for name, d in [("kept today", kept), ("dropped today", drop)]:
        if not len(d) or d.differential_per_pair.isna().all():
            continue
        log(f"  {name:15s}: median {d.differential_per_pair.median():.1f} "
            f"differential week-cells per ordered item pair; "
            f"mean |move correlation| {d.mean_abs_move_corr.median():.3f}")
    r["differential_medians"] = {
        n: {"per_pair": float(d.differential_per_pair.median()),
            "move_corr": float(d.mean_abs_move_corr.median())}
        for n, d in [("kept", kept), ("dropped", drop)]
        if len(d) and not d.differential_per_pair.isna().all()}

    # which dropped categories are rich in exactly the variation the kernel needs?
    if len(drop):
        cand = drop[(drop.differential_per_pair >= a.min_differential)
                    & (drop.n_items_all >= a.min_items_relaxed)
                    & (drop.trips_top >= a.min_cat_trips_relaxed)
                    & (drop.multi_any <= a.max_multi_relaxed)].copy()
        cand = cand.sort_values("differential_per_pair", ascending=False)
        r["recoverable_categories"] = {
            "n": int(len(cand)),
            "criteria": {"min_differential_per_pair": a.min_differential,
                         "min_items": a.min_items_relaxed,
                         "min_cat_trips": a.min_cat_trips_relaxed,
                         "max_multi_any": a.max_multi_relaxed},
            "names": cand.index.tolist()[:40],
            "by_current_reason": cand.drop_reason.value_counts().to_dict()
            if "drop_reason" in cand else {},
        }
        log("")
        log(f"--- categories currently dropped that are rich in kernel-identifying "
            f"variation: {len(cand)}")
        for reason, n in (cand.drop_reason.value_counts().items()
                          if "drop_reason" in cand else []):
            log(f"    {n:3d}  currently dropped for: {reason}")
        if len(cand):
            log(f"    top by differential movement: {', '.join(cand.index[:6])}")

    # how far past the unit-demand threshold are the 79?
    if "drop_reason" in var:
        ud_drop = var[var.drop_reason.str.startswith("unit demand", na=False)]
        if len(ud_drop):
            r["unit_demand_violators"] = {
                "n": int(len(ud_drop)),
                "median_multi_any": float(ud_drop.multi_any.median()),
                "share_within_1pt_of_threshold": float(
                    (ud_drop.multi_any <= a.max_multi_relaxed).mean()),
            }
            log("")
            log(f"--- unit demand: {len(ud_drop)} categories dropped, median "
                f"multi-item share {ud_drop.multi_any.median():.3f} against a "
                f"0.15 threshold; {(ud_drop.multi_any <= 0.20).mean():.0%} sit "
                f"below 0.20")

    with open(os.path.join(OUT, "substitution_eda.json"), "w") as f:
        json.dump(r, f, indent=2)

    # -------------------------------------------------------------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    for d, lab, c in [(kept, "kept today", "#2d6cdf"), (drop, "dropped today", "#9aa5b1")]:
        if len(d):
            ax.scatter(d.share_weeks_base_moves, d.share_weeks_promo_moves,
                       s=18, alpha=0.7, label=lab, color=c)
    ax.set_xlabel("share of item-weeks where the base price moves")
    ax.set_ylabel("share where promotion depth moves")
    ax.set_title("Two price channels, not one\n(a category can be dead in one, alive in the other)",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    v = var.dropna(subset=["differential_per_pair"])
    for d, lab, c in [(v[v.kept_today], "kept today", "#2d6cdf"),
                      (v[v.scored & ~v.kept_today], "dropped today", "#9aa5b1")]:
        if len(d):
            ax.scatter(d.differential_per_pair, d.mean_abs_move_corr,
                       s=18, alpha=0.7, label=lab, color=c)
    ax.axvline(a.min_differential, ls="--", c="k", lw=1, alpha=0.6)
    ax.set_xscale("symlog")
    ax.set_xlabel("differential week-cells per ordered item pair  (log)")
    ax.set_ylabel("mean |correlation| of price moves")
    ax.set_title("What identifies $\\psi_j\\cdot\\psi_k$\nlower-right = movement, but all together (bad)",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    if "drop_reason" in var:
        ud_drop = var[var.drop_reason.str.startswith("unit demand", na=False)]
        if len(ud_drop):
            ax.hist(ud_drop.multi_any.dropna(), bins=30, color="#9aa5b1",
                    edgecolor="white")
            ax.axvline(0.15, color="#d1495b", ls="--", lw=1.5,
                       label="current threshold 0.15")
            ax.set_xlabel("share of category-trips buying >1 item")
            ax.set_ylabel("categories")
            ax.set_title(f"The binding filter: unit demand\n{len(ud_drop)} categories dropped",
                         fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

    fig.suptitle("Re-asking category selection for a model with an explicit substitution kernel",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "substitution_eda.png"), dpi=150, bbox_inches="tight")
    log("")
    log("wrote out/substitution_eda.json, out/category_variation.csv, "
        "figures/substitution_eda.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--top-j", type=int, default=10)
    p.add_argument("--min-differential", type=float, default=8.0,
                   help="differential week-cells per ordered item pair needed before "
                        "psi_j . psi_k is worth estimating for that category")
    p.add_argument("--min-items-relaxed", type=int, default=4)
    p.add_argument("--min-cat-trips-relaxed", type=int, default=150)
    p.add_argument("--max-multi-relaxed", type=float, default=0.20)
    main(p.parse_args())
