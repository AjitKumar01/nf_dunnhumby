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
DATA = os.path.join(HERE, "..", "..", "data")
IN = os.path.join(HERE, "..", "..", "basket_input")
OUT = os.path.join(HERE, "..", "..", "out")
FIG = os.path.join(HERE, "..", "..", "figures")

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
    ev_all = panel[panel.dlogp <= -a.cut]
    idx = {(i, w): k for k, (i, w) in enumerate(zip(panel.item_id, panel.WEEK_NO))}
    arr_n = panel.norm_buy.to_numpy()

    # An item is typically cut many times -- 37,132 events across 4,712 items, so
    # about 8 each -- and 57.7% of consecutive cuts for the same item fall less than
    # 7 weeks apart.  Their +/-3 windows therefore overlap: a week labelled "-1" for
    # one event can be the "+2" of the previous one, so the pre-period is measured on
    # weeks already lifted by an earlier promotion and the post-period decay can be
    # the next promotion arriving rather than this one persisting.
    #
    # Both profiles are reported.  "all" keeps every event; "clean" keeps only events
    # with no other cut of the same item within +/- window weeks, which costs most of
    # the sample but leaves an uncontaminated shape.
    # The demand tail at +1..+3 has two readings: the promotion ended and buying
    # persisted, or the price is simply still down.  Only the price path separates
    # them, so it is measured on the same windows.  arr_p is logp centred on the
    # item's own mean, so 0 = the item at its usual price and -0.20 = 20 log-points
    # (about 18%) below it.
    arr_p = (panel.logp - panel.groupby("item_id").logp.transform("mean")).to_numpy()

    def build(ev, label, arr=None):
        arr = arr_n if arr is None else arr
        prof = {k: [] for k in range(-3, 4)}
        for i, w in zip(ev.item_id.to_numpy(), ev.WEEK_NO.to_numpy()):
            for k in prof:
                j = idx.get((i, w + k))
                if j is not None and np.isfinite(arr[j]):
                    prof[k].append(arr[j])
        pr = {k: float(np.mean(v)) for k, v in prof.items() if len(v) > 30}
        n = {k: int(len(v)) for k, v in prof.items()}
        pre = float(np.mean([pr[k] for k in (-3, -2, -1) if k in pr]))
        return {"events": int(len(ev)), "profile": pr, "n_per_offset": n,
                "pre_period_mean": pre,
                "lift_vs_pre_period": pr.get(0, np.nan) / pre if pre else np.nan,
                "lift_vs_week_minus_1": pr.get(0, np.nan) / pr.get(-1, np.nan)
                if pr.get(-1) else np.nan}

    # isolated events: no other cut of the same item within +/- a.window weeks
    ev_s = ev_all.sort_values(["item_id", "WEEK_NO"])
    gap_prev = ev_s.groupby("item_id").WEEK_NO.diff()
    gap_next = ev_s.groupby("item_id").WEEK_NO.diff(-1).abs()
    isolated = ((gap_prev.isna()) | (gap_prev > a.window)) & \
               ((gap_next.isna()) | (gap_next > a.window))
    ev_clean = ev_s[isolated]

    prof_all = build(ev_all, "all")
    prof_clean = build(ev_clean, "clean")
    # Neither 1.0 nor the pre-period is a clean baseline.  norm_buy averages exactly
    # 1.0 over ALL of an item's weeks, promotion weeks included, so a quiet week sits
    # below 1.0.  And the pre-period is selected: an event requires a price DROP, so
    # the weeks before one are mechanically at an above-average price.  The honest
    # reference is a quiet week -- one more than `window` weeks from any cut of that
    # item -- so it is measured directly.
    near = np.zeros(len(panel), bool)
    for i, w in zip(ev_all.item_id.to_numpy(), ev_all.WEEK_NO.to_numpy()):
        for k in range(-a.window, a.window + 1):
            j = idx.get((i, w + k))
            if j is not None:
                near[j] = True
    qn, qp = arr_n[~near], arr_p[~near]
    quiet = {"weeks": int((~near).sum()),
             "norm_buy": float(np.nanmean(qn)),
             "price_vs_item_mean": float(np.nanmean(qp))}
    r["promotion_event_study"] = r.get("promotion_event_study", {})
    log("")
    log(f"3q. quiet weeks (>{a.window} wks from any cut of that item): "
        f"{quiet['weeks']:,} item-weeks, demand {quiet['norm_buy']:.3f} of the item "
        f"mean, price {quiet['price_vs_item_mean']:+.3f} in logs")

    prof_all["price_path_vs_item_mean"] = build(ev_all, "all", arr_p)["profile"]
    prof_clean["price_path_vs_item_mean"] = build(ev_clean, "clean", arr_p)["profile"]
    prof_all["quiet_week_baseline"] = prof_clean["quiet_week_baseline"] = quiet
    for pr in (prof_all, prof_clean):
        pr["lift_vs_quiet_week"] = pr["profile"][0] / quiet["norm_buy"]
        pr["residual_at_plus3_vs_quiet_week"] = pr["profile"][3] / quiet["norm_buy"]
    r["promotion_event_study"] = {
        "min_log_price_cut": a.cut, "isolation_window_weeks": a.window,
        "all_events": prof_all, "isolated_events": prof_clean,
        "share_events_isolated": float(len(ev_clean) / max(len(ev_all), 1)),
        # kept for backward compatibility with earlier references
        "events": int(len(ev_all)),
        "profile_relative_to_item_mean": prof_all["profile"],
        "lift_at_cut": prof_all["lift_vs_week_minus_1"],
    }
    profile = prof_all["profile"]
    ev = ev_all
    log("")
    log(f"3. promotion event study: {len(ev_all):,} price cuts of >= {a.cut} in logs, "
        f"of which {len(ev_clean):,} ({len(ev_clean)/len(ev_all):.1%}) are isolated "
        f"(no other cut of the same item within +/-{a.window} weeks)")
    for lab, pr in [("all events    ", prof_all), ("isolated only ", prof_clean)]:
        log(f"   {lab}: " + "  ".join(f"{k:+d}:{v:.2f}" for k, v in sorted(pr["profile"].items())))
        log(f"   {' ' * len(lab)}  pre-period {pr['pre_period_mean']:.3f}, "
            f"lift vs pre {pr['lift_vs_pre_period']:.2f}x")
        log(f"   {' ' * len(lab)}  vs quiet week: peak {pr['lift_vs_quiet_week']:.2f}x, "
            f"+3 residual {pr['residual_at_plus3_vs_quiet_week']:.2f}x")
        log(f"   {' ' * len(lab)}  price vs item mean (logs): "
            + "  ".join(f"{k:+d}:{v:+.3f}" for k, v in
                        sorted(pr["price_path_vs_item_mean"].items())))

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

    # ====================================== 4b. breadth: how WIDE is a category buy?
    # The model has a breadth head -- distinct items per purchased category -- and
    # nothing in the exploration established it.  That is the same blank-row failure
    # section 10 describes, caught by the coverage table this time rather than by a
    # bug.  Breadth is a separate quantity from units: three different yogurts is
    # breadth 3, units 3; one yogurt bought three times is breadth 1, units 3.
    bkc = bk.merge(items[["item_id", "cat_id"]], on="item_id")
    cv = bkc.groupby(["BASKET_ID", "cat_id"]).agg(
        breadth=("item_id", "size"), units=("units", "sum"))
    r["breadth"] = {
        "category_visits": int(len(cv)),
        "mean_distinct_items": float(cv.breadth.mean()),
        "share_over_1": float((cv.breadth > 1).mean()),
        "mean_units": float(cv.units.mean()),
        "distribution": {int(k): float(v) for k, v in
                         cv.breadth.clip(upper=5).value_counts(normalize=True)
                         .sort_index().items()},
        "corr_breadth_units": float(cv.corr().iloc[0, 1]),
    }
    # Does a promotion widen the basket, or only deepen it?
    #
    # Two corrections over the first version of this.  (1) The price variable was the
    # mean price of the items the shopper actually BOUGHT, which is chosen rather than
    # faced -- a shopper who adds a second item changes the average by choosing it.
    # The faced price is the mean over EVERY item in the category that week.  (2) The
    # breadth regression is conditioned on the category having been bought, so it says
    # nothing about incidence, and units never enter it, so it says nothing about
    # depth.  Claiming three margins from it was wrong.  All three are estimated here
    # instead, from one (category, week) panel with one estimator and one price.
    faced = (panel.groupby(["cat_id", "WEEK_NO"]).logp.mean()
             .rename("lp_faced").reset_index())

    cw = bkc.groupby(["BASKET_ID", "cat_id", "WEEK_NO"]).agg(
        breadth=("item_id", "size"), units=("units", "sum")).reset_index()
    cwf = cw.merge(faced, on=["cat_id", "WEEK_NO"])
    cwf["lb"] = np.log(cwf.breadth)
    b_br, n_br = within_slope(cwf, "lb", "lp_faced", "cat_id")
    r["breadth"]["elasticity_of_breadth_wrt_price"] = b_br
    r["breadth"]["breadth_regression_visits"] = int(n_br)

    # the three margins, on one panel: a category-week is kept if at least 5 baskets
    # bought the category, so the ratios below are not one-basket artefacts
    nbask = bkc.groupby("WEEK_NO").BASKET_ID.nunique().rename("baskets")
    cwk = (cw.groupby(["cat_id", "WEEK_NO"])
           .agg(visits=("BASKET_ID", "size"), units=("units", "sum"),
                lines=("breadth", "sum")).reset_index()
           .merge(faced, on=["cat_id", "WEEK_NO"]).merge(nbask, on="WEEK_NO"))
    cwk = cwk[cwk.visits >= 5]
    cwk["l_incidence"] = np.log(cwk.visits / cwk.baskets)   # P(category in a basket)
    cwk["l_breadth"] = np.log(cwk.lines / cwk.visits)       # item lines per visit
    cwk["l_depth"] = np.log(cwk.units / cwk.lines)          # units per line
    margins = {}
    for k in ("incidence", "breadth", "depth"):
        s, n = within_slope(cwk, f"l_{k}", "lp_faced", "cat_id")
        margins[k] = s
    margins["category_weeks"] = int(len(cwk))
    r["breadth"]["margins"] = margins

    rb2 = r["breadth"]
    log("")
    log(f"4b. breadth: {rb2['mean_distinct_items']:.3f} distinct items per purchased "
        f"category; {rb2['share_over_1']:.1%} of category visits buy more than one")
    log("    distribution: " + ", ".join(f"{k}:{v:.1%}" for k, v in
                                         rb2["distribution"].items()))
    log(f"    breadth elasticity wrt the FACED price {b_br:+.4f} on {n_br:,} visits "
        f"-> a promotion {'widens' if b_br < 0 else 'does not widen'} the basket")
    log(f"    three margins on {margins['category_weeks']:,} category-weeks: "
        f"incidence {margins['incidence']:+.4f}, breadth {margins['breadth']:+.4f}, "
        f"depth {margins['depth']:+.4f}")

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
        ax.plot(ks, [profile[k] for k in ks], "o-", color=PAL["grey"], lw=2,
                label=f"all {len(ev_all):,} cuts")
        kc = sorted(prof_clean["profile"])
        ax.plot(kc, [prof_clean["profile"][k] for k in kc], "s-", color=PAL["green"],
                lw=2, label=f"isolated only ({len(ev_clean):,})")
        ax.axvline(0, color=PAL["red"], ls="--", lw=1.5)
        ax.axhline(quiet["norm_buy"], color="k", lw=1, ls=":",
                   label=f"quiet week ({quiet['norm_buy']:.2f})")
        style(ax, "Promotion event study\nisolated = no other cut of that item within ±3 wks",
              "weeks relative to the cut", "demand ÷ the item's own mean")
        ax.legend(fontsize=7, loc="upper left")
        # the price path on a twin axis: it shows how much of the post-cut tail is
        # simply the promotion still running rather than demand persisting
        ax2 = ax.twinx()
        pp = prof_clean["price_path_vs_item_mean"]
        kp = sorted(pp)
        ax2.plot(kp, [pp[k] for k in kp], "^--", color=PAL["red"], lw=1.4, ms=5,
                 alpha=.75, label="price (isolated)")
        ax2.axhline(0, color=PAL["red"], lw=.6, alpha=.4)
        ax2.set_ylabel("log price − the item's mean log price", fontsize=8,
                       color=PAL["red"])
        ax2.tick_params(axis="y", labelcolor=PAL["red"], labelsize=8)
        ax2.legend(fontsize=7, loc="upper right")
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
    p.add_argument("--window", type=int, default=3,
                   help="an event is 'isolated' if no other cut of the same item "
                        "falls within this many weeks either side")
    main(p.parse_args())
