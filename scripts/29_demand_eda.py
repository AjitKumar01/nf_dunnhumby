"""
Stage 29 -- The exploration that should have come first: demand response to price,
quantity, stores, and the base rates every model head needs.

21_basket_eda.py investigated exactly the three assumptions I had already decided to
attack -- unit demand, category independence, no state -- and was silent on everything
else.  That is justification, not exploration, and it cost real time: every gap below
was eventually discovered by a *bug*, not by looking at the data.

  what was missing                     how it surfaced instead
  -----------------------------------  ------------------------------------------
  does demand respond to price at all?  never measured; the elasticity first appeared
                                        inside the placebo script, weeks later
  units per purchase line               only checked when challenged on quantity
  store price dispersion, assortment    only checked when challenged on stores
  category incidence base rate          only after the generator produced 58
                                        categories per basket against a real 6.5

That last one is the sharpest.  The base rate is one line of pandas.  Had it been in
the exploration, the incidence sampler would never have been trained on a 50/50 split.

So this script asks the questions a price study should ask before anything is fitted:

  1. does demand actually move when price moves, and by how much?
  2. how does that response vary across categories -- what range should a model produce?
  3. what does a promotion do to demand, as a raw event study?
  4. how many units per line, and does quantity respond to price separately?
  5. how much do stores differ, in price and in assortment?
  6. base rates for every event a model head will have to predict

Writes out/demand_eda.json, out/demand_eda_*.csv and figures/demand_eda_*.png.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
IN = os.path.join(HERE, "..", "basket_input")
OUT = os.path.join(HERE, "..", "out")
FIG = os.path.join(HERE, "..", "figures")

PAL = {"blue": "#2d6cdf", "grey": "#9aa5b1", "red": "#d1495b",
       "green": "#2a9d8f", "amber": "#e9c46a"}


def log(m):
    print(f"[29] {m}", flush=True)


def style(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(alpha=.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return ax


def within_slope(df, y, x, group):
    """OLS slope of y on x after removing group means -- a within-group elasticity."""
    d = df[[y, x, group]].dropna()
    if len(d) < 20:
        return np.nan, 0
    yd = d[y] - d.groupby(group)[y].transform("mean")
    xd = d[x] - d.groupby(group)[x].transform("mean")
    den = float((xd ** 2).sum())
    return (float((xd * yd).sum()) / den if den > 1e-12 else np.nan), len(d)


def main(a):
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = {}
    bk = pd.read_parquet(os.path.join(IN, "baskets.parquet"))
    items = pd.read_parquet(os.path.join(IN, "items.parquet"))
    meta = json.load(open(os.path.join(IN, "meta.json")))
    logp = np.load(os.path.join(IN, "log_price.npy"))
    log(f"{len(bk):,} purchase rows, {meta['n_items']:,} items, "
        f"{meta['n_commodities']} categories, {meta['n_stores']} stores")

    # ---------------------------------------------- item x week demand and price
    d2w = bk[["DAY", "WEEK_NO"]].drop_duplicates().set_index("DAY").WEEK_NO
    wk_of_day = d2w.reindex(np.arange(logp.shape[1])).ffill().bfill().to_numpy().astype(int)
    weeks = np.sort(bk.WEEK_NO.unique())
    price_rows = []
    for w in weeks:
        cols = np.flatnonzero(wk_of_day == w)
        if len(cols):
            price_rows.append(pd.DataFrame({
                "item_id": np.arange(logp.shape[0]), "WEEK_NO": w,
                "logp": logp[:, cols].mean(axis=1)}))
    P = pd.concat(price_rows, ignore_index=True)

    dem = (bk.groupby(["item_id", "WEEK_NO"])
           .agg(buyers=("units", "size"), units=("units", "sum")).reset_index())
    trips = bk.groupby("WEEK_NO").BASKET_ID.nunique().rename("trips")
    panel = (P.merge(dem, on=["item_id", "WEEK_NO"], how="left")
             .fillna({"buyers": 0, "units": 0})
             .merge(trips, on="WEEK_NO", how="left")
             .merge(items[["item_id", "cat_id", "COMMODITY_DESC"]], on="item_id"))
    panel["lbuy"] = np.log((panel.buyers + 0.5) / panel.trips)
    panel["lunits"] = np.log((panel.units + 0.5) / panel.trips)
    panel["upb"] = panel.units / panel.buyers.replace(0, np.nan)

    # ============================================== 1. does demand respond at all?
    b_buy, n1 = within_slope(panel, "lbuy", "logp", "item_id")
    b_uni, _ = within_slope(panel, "lunits", "logp", "item_id")
    sub = panel[panel.buyers >= 3].copy()
    sub["lupb"] = np.log(sub.upb)
    b_upb, n3 = within_slope(sub, "lupb", "logp", "item_id")
    r["overall"] = {
        "item_weeks": int(n1),
        "elasticity_buyers": b_buy,
        "elasticity_units": b_uni,
        "elasticity_units_per_buyer": b_upb,
        "share_of_units_elasticity_from_quantity": b_upb / b_uni if b_uni else np.nan,
    }
    log("")
    log("1. does demand respond to price?  (within-item, log-log)")
    log(f"   buyers per trip        {b_buy:+.4f}")
    log(f"   units per trip         {b_uni:+.4f}")
    log(f"   units per buyer        {b_upb:+.4f}   <- the quantity margin, "
        f"{abs(b_upb / b_uni):.0%} of the total")

    # ============================================= 2. how does it vary by category?
    rows = []
    for cat, g in panel.groupby("COMMODITY_DESC"):
        if g.item_id.nunique() < 3 or len(g) < 200:
            continue
        e, n = within_slope(g, "lbuy", "logp", "item_id")
        gg = g[g.buyers >= 3].copy()
        gg["lupb"] = np.log(gg.upb)
        eq, _ = within_slope(gg, "lupb", "logp", "item_id")
        rows.append({"category": cat, "items": int(g.item_id.nunique()),
                     "item_weeks": int(n), "elasticity_buyers": e,
                     "elasticity_units_per_buyer": eq})
    E = pd.DataFrame(rows).sort_values("elasticity_buyers")
    E.to_csv(os.path.join(OUT, "demand_eda_by_category.csv"), index=False)
    r["by_category"] = {
        "categories": int(len(E)),
        "median": float(E.elasticity_buyers.median()),
        "p10": float(E.elasticity_buyers.quantile(.1)),
        "p90": float(E.elasticity_buyers.quantile(.9)),
        "share_negative": float((E.elasticity_buyers < 0).mean()),
        "most_elastic": E.head(5).category.tolist(),
        "least_elastic": E.tail(5).category.tolist(),
    }
    rb = r["by_category"]
    log("")
    log(f"2. across {len(E)} categories: median {rb['median']:+.3f}, "
        f"p10 {rb['p10']:+.3f}, p90 {rb['p90']:+.3f}; "
        f"{rb['share_negative']:.0%} negative")
    log(f"   most elastic:  {', '.join(E.head(3).category)}")
    log(f"   least elastic: {', '.join(E.tail(3).category)}")

    # ================================================ 3. promotions, as an event study
    # Demand around a price cut, without a model: line up every item-week where price
    # fell at least `cut` in logs and average demand in the weeks around it.
    panel = panel.sort_values(["item_id", "WEEK_NO"])
    panel["dlogp"] = panel.groupby("item_id").logp.diff()
    panel["norm_buy"] = panel.buyers / panel.groupby("item_id").buyers.transform("mean").replace(0, np.nan)
    ev = panel[panel.dlogp <= -a.cut]
    idx = {(i, w): k for k, (i, w) in enumerate(zip(panel.item_id, panel.WEEK_NO))}
    prof = {k: [] for k in range(-3, 4)}
    arr_i = panel.item_id.to_numpy(); arr_w = panel.WEEK_NO.to_numpy()
    arr_n = panel.norm_buy.to_numpy()
    for i, w in zip(ev.item_id.to_numpy(), ev.WEEK_NO.to_numpy()):
        for k in prof:
            j = idx.get((i, w + k))
            if j is not None and np.isfinite(arr_n[j]):
                prof[k].append(arr_n[j])
    profile = {k: float(np.mean(v)) for k, v in prof.items() if len(v) > 30}
    r["promotion_event_study"] = {
        "events": int(len(ev)), "min_log_price_cut": a.cut,
        "profile_relative_to_item_mean": profile,
        "lift_at_cut": profile.get(0, np.nan) / profile.get(-1, np.nan)
        if profile.get(-1) else np.nan,
    }
    log("")
    log(f"3. promotion event study: {len(ev):,} price cuts of >= {a.cut} in logs")
    log("   demand relative to the item's own mean, by week around the cut:")
    log("     " + "  ".join(f"{k:+d}:{v:.2f}" for k, v in sorted(profile.items())))
    if profile.get(-1):
        log(f"   -> lift at the cut week: {profile[0] / profile[-1]:.2f}x the week before")

    # ============================================================ 4. quantity
    u = bk.units.to_numpy()
    r["quantity"] = {
        "rows": int(len(u)), "mean_units": float(u.mean()),
        "share_gt1": float((u > 1).mean()),
        "share_of_units_in_multi_rows": float(u[u > 1].sum() / u.sum()),
        "distribution": {int(k): float(v) for k, v in
                         pd.Series(u).clip(upper=6).value_counts(normalize=True)
                         .sort_index().items()},
    }
    rq = r["quantity"]
    log("")
    log(f"4. quantity: {rq['share_gt1']:.1%} of rows buy >1 unit, carrying "
        f"{rq['share_of_units_in_multi_rows']:.1%} of all units (mean {rq['mean_units']:.2f})")

    # ============================================================== 5. stores
    tx = pd.read_parquet(os.path.join(DATA, "tx.parquet"),
                         columns=["PRODUCT_ID", "STORE_ID", "WEEK_NO", "unit_price"])
    tx = tx[tx.PRODUCT_ID.isin(set(items.PRODUCT_ID))]
    sw = tx.groupby(["PRODUCT_ID", "STORE_ID", "WEEK_NO"]).unit_price.median().rename("ps").reset_index()
    cw = tx.groupby(["PRODUCT_ID", "WEEK_NO"]).unit_price.median().rename("pc").reset_index()
    mm = sw.merge(cw, on=["PRODUCT_ID", "WEEK_NO"])
    carried = tx.groupby("STORE_ID").PRODUCT_ID.nunique()
    r["stores"] = {
        "stores": int(tx.STORE_ID.nunique()),
        "share_store_weeks_over_1c_from_chain": float((mm.ps - mm.pc).abs().gt(0.01).mean()),
        "sd_across_stores_within_item_week": float(
            mm.groupby(["PRODUCT_ID", "WEEK_NO"]).ps.std().mean()),
        "median_catalogue_share_carried": float(carried.median() / len(items)),
        "p10_catalogue_share_carried": float(carried.quantile(.1) / len(items)),
    }
    rs = r["stores"]
    log("")
    log(f"5. stores: {rs['stores']}; {rs['share_store_weeks_over_1c_from_chain']:.1%} of "
        f"store-item-weeks differ >1c from the chain price "
        f"(sd ${rs['sd_across_stores_within_item_week']:.3f}); median store carries "
        f"{rs['median_catalogue_share_carried']:.0%} of the catalogue")

    # ====================================================== 6. base rates
    bk2 = bk.merge(items[["item_id", "cat_id"]], on="item_id")
    per_basket_cat = bk2.groupby("BASKET_ID").cat_id.nunique()
    n_baskets = bk.BASKET_ID.nunique()
    r["base_rates"] = {
        "categories_per_basket": float(per_basket_cat.mean()),
        "n_categories": int(meta["n_commodities"]),
        "category_incidence_rate": float(per_basket_cat.mean() / meta["n_commodities"]),
        "items_per_basket": float(bk.groupby("BASKET_ID").size().mean()),
        "n_items": int(meta["n_items"]),
        "item_purchase_rate": float(bk.groupby("BASKET_ID").size().mean() / meta["n_items"]),
        "units_per_basket": float(bk.groupby("BASKET_ID").units.sum().mean()),
    }
    rr = r["base_rates"]
    log("")
    log("6. base rates every model head has to reproduce:")
    log(f"   category incidence  {rr['categories_per_basket']:.2f} of {rr['n_categories']} "
        f"= {rr['category_incidence_rate']:.2%}   <- an incidence head trained on a "
        f"balanced sample is calibrated to 50%, not to this")
    log(f"   item purchase       {rr['items_per_basket']:.2f} of {rr['n_items']:,} "
        f"= {rr['item_purchase_rate']:.3%}")
    log(f"   units per basket    {rr['units_per_basket']:.2f}")

    with open(os.path.join(OUT, "demand_eda.json"), "w") as f:
        json.dump(r, f, indent=2, default=float)

    # ================================================================== figures
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    ax = axes[0]
    s = panel.dropna(subset=["dlogp"])
    s = s[s.dlogp.abs() > 0.01]
    s = s.assign(dlbuy=s.groupby("item_id").lbuy.diff())
    s = s.dropna(subset=["dlbuy"])
    bins = pd.qcut(s.dlogp, 12, duplicates="drop")
    prof2 = s.groupby(bins, observed=True).agg(x=("dlogp", "mean"), y=("dlbuy", "mean"))
    ax.plot(prof2.x, prof2.y, "o-", color=PAL["blue"], lw=2)
    ax.axhline(0, color="k", lw=.8); ax.axvline(0, color="k", lw=.8)
    style(ax, f"Demand moves against price\nwithin-item slope {b_buy:+.2f}",
          "Δ log price, week on week", "Δ log buyers per trip")

    ax = axes[1]
    ax.hist(E.elasticity_buyers, bins=40, color=PAL["blue"], edgecolor="white")
    ax.axvline(0, color="k", lw=1)
    ax.axvline(E.elasticity_buyers.median(), color=PAL["red"], ls="--", lw=1.5,
               label=f"median {E.elasticity_buyers.median():+.2f}")
    style(ax, f"Elasticity varies across {len(E)} categories\n"
              f"{(E.elasticity_buyers < 0).mean():.0%} negative",
          "within-item elasticity of buyers", "categories")
    ax.legend(fontsize=8)

    ax = axes[2]
    if profile:
        ks = sorted(profile)
        ax.plot(ks, [profile[k] for k in ks], "o-", color=PAL["green"], lw=2)
        ax.axvline(0, color=PAL["red"], ls="--", lw=1.5, label="week of the price cut")
        ax.axhline(1, color="k", lw=.8)
        style(ax, f"Promotion event study\n{len(ev):,} cuts of ≥{a.cut} in logs",
              "weeks relative to the cut", "demand ÷ the item's own mean")
        ax.legend(fontsize=8)
    fig.suptitle("Demand response to price — the exploration that should have come first",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "demand_eda_price.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    ax = axes[0]
    dist = r["quantity"]["distribution"]
    ax.bar(list(dist), list(dist.values()), color=PAL["blue"])
    style(ax, f"Units per purchase line\n{rq['share_gt1']:.0%} buy >1, carrying "
              f"{rq['share_of_units_in_multi_rows']:.0%} of volume",
          "units (clipped at 6)", "share of rows")

    ax = axes[1]
    dev = (mm.ps - mm.pc)
    ax.hist(dev.clip(-1, 1), bins=60, color=PAL["amber"], edgecolor="white")
    ax.axvline(0, color="k", lw=1)
    style(ax, f"Store price − chain price\n{rs['share_store_weeks_over_1c_from_chain']:.0%} "
              f"differ by >1c", "$ deviation (clipped)", "store-item-weeks")

    ax = axes[2]
    ax.hist(carried / len(items), bins=40, color=PAL["green"], edgecolor="white")
    ax.axvline(rs["median_catalogue_share_carried"], color=PAL["red"], ls="--", lw=1.5,
               label=f"median {rs['median_catalogue_share_carried']:.0%}")
    style(ax, "Assortment: share of the catalogue a store carries",
          "share of 5,455 items", "stores")
    ax.legend(fontsize=8)
    fig.suptitle("Quantity and stores — both were assumed away, neither should have been",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "demand_eda_quantity_stores.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    log("")
    log("wrote out/demand_eda.json, demand_eda_by_category.csv and 2 figures")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cut", type=float, default=0.15,
                   help="minimum log price fall that counts as a promotion")
    main(p.parse_args())
