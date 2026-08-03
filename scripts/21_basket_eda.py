"""
Stage 21 -- Data exploration for a basket model with household state.

The earlier exploration (08, 12, 18) asked whether the data supported the paper's
model: one item per category, categories independent, no dynamics.  This one asks
the opposite question -- what does the data look like if those three assumptions are
dropped?  Every section here exists to settle a specific modelling decision:

  1. Is unit demand defensible?      -> decides whether the within-category softmax
                                        survives at all
  2. What does a basket look like?   -> decides the likelihood
  3. Do products co-occur beyond     -> decides whether an interaction term is
     chance?                            needed, and whether it must be low rank
  4. Is there household state?       -> decides whether purchase timing carries
                                        information the paper's model throws away
  5. How big is the usable           -> decides how many items an embedding can be
     catalogue?                         estimated for, and whether the
                                        sub-commodity test is even answerable
  6. Does price still vary once the  -> the expanded catalogue must retain the
     catalogue expands?                 price variation that identifies elasticity

Writes out/basket_eda.json, out/basket_eda_*.csv and figures/basket_eda_*.png.
"""
import argparse
import json
import os
from collections import Counter
from itertools import combinations

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

PALETTE = {"blue": "#2d6cdf", "grey": "#9aa5b1", "red": "#d1495b",
           "green": "#2a9d8f", "amber": "#e9c46a"}


def log(m):
    print(f"[21] {m}", flush=True)


def style(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax


def main(a):
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = {}
    tx = pd.read_parquet(os.path.join(DATA, "tx.parquet"),
                         columns=["household_key", "DAY", "WEEK_NO", "PRODUCT_ID",
                                  "BASKET_ID", "QUANTITY", "STORE_ID",
                                  "unit_price", "base_price"])
    prod = pd.read_csv(RAW + "product.csv",
                       usecols=["PRODUCT_ID", "COMMODITY_DESC", "SUB_COMMODITY_DESC",
                                "BRAND", "MANUFACTURER", "DEPARTMENT"])
    tx = tx.merge(prod, on="PRODUCT_ID", how="left")
    tx = tx.dropna(subset=["COMMODITY_DESC"])
    r["raw"] = {"lines": int(len(tx)), "households": int(tx.household_key.nunique()),
                "products": int(tx.PRODUCT_ID.nunique()),
                "baskets": int(tx.BASKET_ID.nunique()),
                "days": int(tx.DAY.nunique()),
                "commodities": int(tx.COMMODITY_DESC.nunique()),
                "sub_commodities": int(tx.SUB_COMMODITY_DESC.nunique())}
    log(f"{len(tx):,} lines, {tx.PRODUCT_ID.nunique():,} products, "
        f"{tx.BASKET_ID.nunique():,} baskets, {tx.SUB_COMMODITY_DESC.nunique():,} sub-commodities")

    # =================================================== 1. is unit demand real?
    # The paper assumes at most one item per category per trip, and 02_select_sample
    # drops any category violating it by more than 15%.  That filter removed 79 of
    # 276 categories here.  Measure the violation directly, per basket and per
    # category, on the *whole* catalogue.
    bas = tx.groupby("BASKET_ID").agg(
        n_lines=("PRODUCT_ID", "size"), n_items=("PRODUCT_ID", "nunique"),
        n_cats=("COMMODITY_DESC", "nunique"), n_subs=("SUB_COMMODITY_DESC", "nunique"),
        units=("QUANTITY", "sum"))
    multi_cat_basket = float((bas.n_items > bas.n_cats).mean())
    r["unit_demand"] = {
        "baskets_with_multiple_items_in_one_category": multi_cat_basket,
        "baskets_spanning_2plus_categories": float((bas.n_cats >= 2).mean()),
        "baskets_spanning_5plus_categories": float((bas.n_cats >= 5).mean()),
        "mean_distinct_items_per_basket": float(bas.n_items.mean()),
        "median_distinct_items_per_basket": float(bas.n_items.median()),
        "mean_units_per_basket": float(bas.units.mean()),
    }
    log(f"1. unit demand: {multi_cat_basket:.1%} of baskets hold >1 item from a single "
        f"category; {(bas.n_cats >= 2).mean():.1%} span 2+ categories")

    # per category: share of category-trips buying more than one item of it
    ct = tx.groupby(["household_key", "DAY", "COMMODITY_DESC"]).PRODUCT_ID.nunique()
    cat_multi = (ct > 1).groupby(level="COMMODITY_DESC").mean().rename("multi_share")
    cat_size = ct.groupby(level="COMMODITY_DESC").size().rename("cat_trips")
    cm = pd.concat([cat_multi, cat_size], axis=1).sort_values("multi_share", ascending=False)
    cm.to_csv(os.path.join(OUT, "basket_eda_category_multi.csv"))
    r["unit_demand"]["categories_over_15pct"] = int((cm.multi_share > 0.15).sum())
    r["unit_demand"]["categories_total"] = int(len(cm))
    r["unit_demand"]["median_category_multi_share"] = float(cm.multi_share.median())
    log(f"   per category: {int((cm.multi_share > 0.15).sum())} of {len(cm)} exceed the "
        f"paper's 15% threshold (median {cm.multi_share.median():.1%})")

    # =================================================== 2. what is a basket?
    # =================================================== 3. co-occurrence structure
    # Basket size confounds any raw co-occurrence measure: a household buying 30
    # things co-buys everything.  Stratify by basket size, and compare each pair's
    # lift against the *stratum* distribution rather than against 1.0 -- at fixed
    # basket size a random allocation already produces lift below 1.
    freq = tx.groupby("PRODUCT_ID").size()
    keep = set(freq[freq >= a.min_lines].index)
    sub = tx[tx.PRODUCT_ID.isin(keep)]
    sc = sub.groupby("BASKET_ID").SUB_COMMODITY_DESC.apply(lambda s: sorted(set(s)))
    sizes = sc.apply(len)
    strat = sc[(sizes >= 5) & (sizes <= 20)]
    n = len(strat)
    solo, pair = Counter(), Counter()
    for cats in strat:
        for c in cats:
            solo[c] += 1
        for x, y in combinations(cats, 2):
            pair[(x, y)] += 1
    rows = [{"a": x, "b": y, "n": k,
             "lift": (k / n) / ((solo[x] / n) * (solo[y] / n))}
            for (x, y), k in pair.items() if k >= a.min_pair]
    L = pd.DataFrame(rows).sort_values("lift", ascending=False)
    L.to_csv(os.path.join(OUT, "basket_eda_cooccurrence.csv"), index=False)
    r["cooccurrence"] = {
        "stratum": "baskets with 5-20 distinct sub-commodities",
        "baskets": int(n), "pairs_measured": int(len(L)),
        "median_lift": float(L.lift.median()),
        "share_lift_above_1_5": float((L.lift > 1.5).mean()),
        "share_lift_above_2": float((L.lift > 2.0).mean()),
        "share_lift_below_0_67": float((L.lift < 0.67).mean()),
        "p99_lift": float(L.lift.quantile(0.99)),
    }
    log(f"3. co-occurrence on {n:,} baskets, {len(L):,} sub-commodity pairs: "
        f"median lift {L.lift.median():.2f}, {(L.lift > 2).mean():.1%} above 2.0, "
        f"{(L.lift < 0.67).mean():.1%} below 0.67")

    # Substitutes live *inside* a sub-commodity: two items of the same kind are
    # rarely bought together.  That is the signal an interaction term must capture,
    # and it is invisible to a model that only ever sees one item per category.
    it_bask = sub.groupby("BASKET_ID").PRODUCT_ID.apply(lambda s: sorted(set(s)))
    it_bask = it_bask[it_bask.apply(len).between(2, 30)]
    p2s = prod.set_index("PRODUCT_ID").SUB_COMMODITY_DESC.to_dict()
    same_sub_pairs = diff_sub_pairs = 0
    for its in it_bask:
        for x, y in combinations(its, 2):
            if p2s.get(x) == p2s.get(y):
                same_sub_pairs += 1
            else:
                diff_sub_pairs += 1
    # expected share of same-sub pairs if items were drawn at random from the catalogue
    cnt = sub.groupby("SUB_COMMODITY_DESC").PRODUCT_ID.nunique()
    tot_items = int(cnt.sum())
    exp_same = float((cnt * (cnt - 1)).sum() / (tot_items * (tot_items - 1)))
    obs_same = same_sub_pairs / max(same_sub_pairs + diff_sub_pairs, 1)
    r["within_sub_copurchase"] = {
        "observed_share_of_pairs_same_sub": obs_same,
        "expected_if_random": exp_same,
        "ratio": obs_same / exp_same if exp_same else float("nan"),
    }
    log(f"   items from the SAME sub-commodity are {obs_same / exp_same:.2f}x "
        f"{'more' if obs_same > exp_same else 'less'} likely to share a basket than chance "
        f"({obs_same:.4f} observed vs {exp_same:.4f})")

    # =================================================== 4. household state
    # Does when you last bought predict whether you buy now?  The paper's model has
    # no state at all: two households with identical demographics and tastes get the
    # same purchase probability whether they bought the category yesterday or never.
    log("4. household state: inter-purchase timing")
    s = sub[["household_key", "DAY", "SUB_COMMODITY_DESC"]].drop_duplicates()
    s = s.sort_values(["household_key", "SUB_COMMODITY_DESC", "DAY"])
    s["prev"] = s.groupby(["household_key", "SUB_COMMODITY_DESC"]).DAY.shift()
    s["gap"] = s.DAY - s.prev
    gaps = s.gap.dropna()
    r["state"] = {
        "repeat_purchase_events": int(len(gaps)),
        "median_gap_days": float(gaps.median()),
        "p25_gap_days": float(gaps.quantile(.25)),
        "p75_gap_days": float(gaps.quantile(.75)),
    }
    log(f"   {len(gaps):,} repeat purchases; gap median {gaps.median():.0f} days "
        f"(p25 {gaps.quantile(.25):.0f}, p75 {gaps.quantile(.75):.0f})")

    # The decisive test: hazard of buying a sub-commodity today as a function of days
    # since that household last bought it, against the household's own base rate.
    # A flat curve would mean timing carries no information and state can be skipped.
    trips = sub[["household_key", "DAY"]].drop_duplicates().sort_values(["household_key", "DAY"])
    top_subs = sub.groupby("SUB_COMMODITY_DESC").household_key.nunique().nlargest(a.n_state_subs).index
    haz = []
    for sc_name in top_subs:
        buys = (sub.loc[sub.SUB_COMMODITY_DESC == sc_name, ["household_key", "DAY"]]
                .drop_duplicates().assign(bought=True))
        # Restrict to households that ever buy it: for the rest the hazard is
        # trivially zero and would only dilute the curve.
        t = trips[trips.household_key.isin(set(buys.household_key))].copy()
        t = t.merge(buys, on=["household_key", "DAY"], how="left")
        t["bought"] = t.bought.fillna(False).astype(bool)
        t = t.sort_values(["household_key", "DAY"])
        # Last purchase strictly *before* this trip: forward-fill the buy days, then
        # shift one row within the household so the current trip cannot see itself.
        t["buy_day"] = t.DAY.where(t.bought)
        t["ffilled"] = t.groupby("household_key").buy_day.ffill()
        t["prev_buy"] = t.groupby("household_key").ffilled.shift()
        t = t.dropna(subset=["prev_buy"])
        haz.append(pd.DataFrame({"since": t.DAY - t.prev_buy, "bought": t.bought}))
    H = pd.concat(haz) if haz else pd.DataFrame(columns=["since", "bought"])
    if len(H):
        bins = [0, 3, 7, 14, 21, 28, 42, 56, 84, 10 ** 6]
        H["bin"] = pd.cut(H.since, bins=bins, right=False)
        hz = H.groupby("bin", observed=True).bought.agg(["mean", "size"])
        hz.index = [f"{int(i.left)}-{int(i.right) if i.right < 10**5 else '+'}" for i in hz.index]
        hz.to_csv(os.path.join(OUT, "basket_eda_hazard.csv"))
        r["state"]["hazard_by_days_since"] = {k: float(v) for k, v in hz["mean"].items()}
        r["state"]["hazard_ratio_max_over_min"] = float(hz["mean"].max() / max(hz["mean"].min(), 1e-9))
        log(f"   repurchase hazard by days since last buy: "
            + ", ".join(f"{k} {v:.3f}" for k, v in hz["mean"].items()))
        log(f"   -> max/min hazard ratio {hz['mean'].max() / max(hz['mean'].min(), 1e-9):.2f}x "
            f"(a flat curve would mean state is unnecessary)")

    # =================================================== 5. usable catalogue
    cat_rows = []
    for thr in [20, 50, 100, 200, 500]:
        k = freq[freq >= thr]
        s2 = tx[tx.PRODUCT_ID.isin(set(k.index))]
        per_sub = s2.groupby("SUB_COMMODITY_DESC").PRODUCT_ID.nunique()
        cat_rows.append({
            "min_lines": thr, "items": int(len(k)),
            "share_of_lines": float(len(s2) / len(tx)),
            "sub_commodities": int(s2.SUB_COMMODITY_DESC.nunique()),
            "commodities": int(s2.COMMODITY_DESC.nunique()),
            # the sub-commodity clustering test needs >=2 items in a sub-commodity
            "sub_commodities_with_2plus_items": int((per_sub >= 2).sum()),
            "items_in_testable_subs": int(per_sub[per_sub >= 2].sum()),
        })
    C = pd.DataFrame(cat_rows)
    C.to_csv(os.path.join(OUT, "basket_eda_catalogue.csv"), index=False)
    r["catalogue"] = C.to_dict("records")
    log("5. usable catalogue by frequency threshold:")
    for row in cat_rows:
        log(f"   >={row['min_lines']:4d} lines: {row['items']:6,} items, "
            f"{row['share_of_lines']:.1%} of volume, {row['sub_commodities']:4d} sub-commodities, "
            f"{row['sub_commodities_with_2plus_items']:4d} testable")

    # =================================================== 6. price variation retained
    pw = (sub.groupby(["PRODUCT_ID", "WEEK_NO"])
          .agg(price=("unit_price", "median"), base=("base_price", "median"))
          .reset_index().sort_values(["PRODUCT_ID", "WEEK_NO"]))
    pw["dp"] = pw.groupby("PRODUCT_ID").price.diff()
    pw["consec"] = pw.groupby("PRODUCT_ID").WEEK_NO.diff() == 1
    mv = pw[pw.consec]
    cv = sub.groupby("PRODUCT_ID").unit_price.apply(lambda v: v.std() / max(v.mean(), 1e-9))
    r["price_variation"] = {
        "items": int(sub.PRODUCT_ID.nunique()),
        "share_item_weeks_price_moves": float((mv.dp.abs() > 0.01).mean()),
        "median_within_item_cv": float(cv.median()),
        "median_promo_depth_when_on": float(
            (1 - sub.unit_price / sub.base_price).clip(lower=0).replace(0, np.nan).median()),
    }
    log(f"6. price still moves on {(mv.dp.abs() > 0.01).mean():.1%} of item-weeks for the "
        f"{sub.PRODUCT_ID.nunique():,}-item catalogue (within-item CV {cv.median():.3f})")

    with open(os.path.join(OUT, "basket_eda.json"), "w") as f:
        json.dump(r, f, indent=2, default=float)

    # ================================================================== figures
    # --- figure A: why unit demand has to go
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    vc = bas.n_items.clip(upper=40).value_counts().sort_index()
    ax.bar(vc.index, vc.values, color=PALETTE["blue"], width=.9)
    ax.axvline(1.5, color=PALETTE["red"], ls="--", lw=1.5)
    ax.text(2.0, vc.max() * .8, "the paper models\nonly this side\nfor one category",
            fontsize=8, color=PALETTE["red"])
    style(ax, f"Distinct items per basket\nmean {bas.n_items.mean():.1f}, median "
              f"{bas.n_items.median():.0f}", "distinct items (clipped at 40)", "baskets")

    ax = axes[1]
    ax.hist(cm.multi_share, bins=40, color=PALETTE["blue"], edgecolor="white")
    ax.axvline(0.15, color=PALETTE["red"], ls="--", lw=1.5, label="paper's 15% cutoff")
    style(ax, f"Unit demand violated per category\n{(cm.multi_share > 0.15).sum()} of "
              f"{len(cm)} categories exceed the cutoff",
          "share of category-trips buying >1 item", "categories")
    ax.legend(fontsize=8)

    ax = axes[2]
    vc2 = bas.n_cats.clip(upper=25).value_counts().sort_index()
    ax.bar(vc2.index, vc2.values, color=PALETTE["green"], width=.9)
    style(ax, f"Categories per basket\n{(bas.n_cats >= 2).mean():.0%} span 2 or more",
          "distinct categories (clipped at 25)", "baskets")
    fig.suptitle("1. The unit-demand assumption fails in the majority of baskets", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "basket_eda_unit_demand.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- figure B: interaction structure
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    ax.hist(np.log2(L.lift.clip(0.05, 20)), bins=60, color=PALETTE["blue"], edgecolor="white")
    ax.axvline(0, color="k", lw=1.2)
    style(ax, "Sub-commodity co-purchase lift\n(basket size held between 5 and 20)",
          "log2 lift  (0 = independence)", "pairs")

    ax = axes[1]
    top = L.head(12).iloc[::-1]
    lbl = [f"{x[:20]} + {y[:20]}" for x, y in zip(top.a, top.b)]
    ax.barh(range(len(top)), top.lift, color=PALETTE["green"])
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(lbl, fontsize=7)
    style(ax, "Strongest complements\n(bought together far more than chance)", "lift")

    ax = axes[2]
    bot = L.tail(12)
    lbl = [f"{x[:20]} + {y[:20]}" for x, y in zip(bot.a, bot.b)]
    ax.barh(range(len(bot)), bot.lift, color=PALETTE["red"])
    ax.set_yticks(range(len(bot)))
    ax.set_yticklabels(lbl, fontsize=7)
    ax.axvline(1, color="k", ls="--", lw=1)
    style(ax, "Strongest avoidance\n(substitutes and mutually exclusive needs)", "lift")
    fig.suptitle("3. Products interact: a model of independent categories cannot express this",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "basket_eda_interaction.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- figure C: state
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    ax.hist(gaps.clip(upper=120), bins=60, color=PALETTE["blue"], edgecolor="white")
    style(ax, f"Days between repeat purchases\nmedian {gaps.median():.0f} days",
          "days since the same household last bought it", "events")

    ax = axes[1]
    if len(H):
        ax.plot(range(len(hz)), hz["mean"].values, "o-", color=PALETTE["blue"], lw=2)
        ax.set_xticks(range(len(hz)))
        ax.set_xticklabels(hz.index, rotation=45, ha="right", fontsize=8)
        style(ax, f"Repurchase hazard by recency\nmax/min = "
                  f"{hz['mean'].max() / max(hz['mean'].min(), 1e-9):.1f}x",
              "days since last purchase", "P(buy on this trip)")

    ax = axes[2]
    tb = tx.groupby("household_key").DAY.nunique()
    ax.hist(tb.clip(upper=250), bins=50, color=PALETTE["amber"], edgecolor="white")
    style(ax, f"Trips per household\nmedian {tb.median():.0f} over {tx.DAY.nunique()} days",
          "shopping days", "households")
    fig.suptitle("4. Purchase timing carries information the paper's model has no place for",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "basket_eda_state.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- figure D: catalogue
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    ax.plot(C["min_lines"], C["items"], "o-", color=PALETTE["blue"], label="items")
    ax.plot(C["min_lines"], C["items_in_testable_subs"], "s--", color=PALETTE["green"],
            label="items in a sub-commodity with 2+ items")
    ax.axhline(560, color=PALETTE["red"], ls=":", lw=1.5, label="items the paper's port modelled")
    ax.set_xscale("log"); ax.set_yscale("log")
    style(ax, "Catalogue available to a basket model", "minimum purchase lines per item", "items")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(C["min_lines"], C["sub_commodities"], "o-", color=PALETTE["blue"], label="sub-commodities")
    ax.plot(C["min_lines"], C["sub_commodities_with_2plus_items"], "s--", color=PALETTE["green"],
            label="testable (2+ items)")
    ax.set_xscale("log")
    style(ax, "Ground truth available for the embedding test",
          "minimum purchase lines per item", "sub-commodities")
    ax.legend(fontsize=8)
    fig.suptitle("5. Dropping unit demand expands the catalogue by an order of magnitude",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "basket_eda_catalogue.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    log("wrote out/basket_eda.json, 3 csv tables and 4 figures")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--min-lines", type=int, default=100,
                   help="minimum purchase lines for an item to enter the analysis")
    p.add_argument("--min-pair", type=int, default=30,
                   help="minimum co-occurrences before a pair's lift is reported")
    p.add_argument("--n-state-subs", type=int, default=60,
                   help="how many of the most widely bought sub-commodities to use "
                        "for the repurchase hazard")
    main(p.parse_args())
