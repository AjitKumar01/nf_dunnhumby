"""
Stage 12 -- Figures for PREPROCESSING.md.

Every figure is regenerated from the parquet/CSV artefacts produced by stages 01-04,
so the report and the pipeline cannot drift apart.

  fig 1  price_definitions.png      (produced by 10_price_definition_audit.py)
  fig 2  week_structure.png         where prices change; weekday demand profile
  fig 3  cross_store_prices.png     how much prices differ between stores
  fig 4  random_weight_screen.png   price discreteness, and what it removes
  fig 5  holiday_weeks.png          weekly spend and the excluded weeks
  fig 6  category_funnel.png        the filter waterfall, and unit-demand diagnostics
  fig 7  sample_profile.png         trips, basket size, purchase rates
  fig 8  price_panel.png            observed vs carried-forward prices; variation kept
  fig 9  placebo.png                (produced by 11_placebo_tests.py)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data")
MI = os.path.join(HERE, "..", "..", "model_input")
OUT = os.path.join(HERE, "..", "..", "out")
FIG = os.path.join(HERE, "..", "..", "figures")
# Raw dunnhumby CSVs.  Defaults to a sibling of the repository; override with
# NF_RAW_DIR if the download lives somewhere else.
RAW = os.path.join(os.environ.get(
    "NF_RAW_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                 "dunnhumby_The-Complete-Journey",
                 "dunnhumby_The-Complete-Journey CSV")), "")

BLUE, RED, GREY, GREEN = "#2d6cdf", "#c1432c", "#9aa5b1", "#2e8b6f"
DAYNAMES = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]


def log(m):
    print(f"[12] {m}", flush=True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"wrote figures/{name}")


def modal_share(v, nd=2):
    c = np.round(np.asarray(v, float), nd)
    if not len(c):
        return np.nan
    _, k = np.unique(c, return_counts=True)
    return k.max() / len(c)


# ---------------------------------------------------------------- fig 2
def fig_week_structure(tx):
    pdm = tx.groupby(["PRODUCT_ID", "DAY"]).unit_price.agg(["median", "size"]).reset_index()
    dmap = tx[["DAY", "WEEK_NO", "weekday"]].drop_duplicates()
    g = pdm[pdm["size"] >= 5].merge(dmap, on="DAY").sort_values(["PRODUCT_ID", "DAY"])
    for c, src in [("prev", "median"), ("prevday", "DAY"), ("prevwk", "WEEK_NO"),
                   ("prevwd", "weekday")]:
        g[c] = g.groupby("PRODUCT_ID")[src].shift()
    g = g[(g.DAY - g.prevday) == 1]
    g["chg"] = (g["median"] - g["prev"]).abs() > 0.02

    by_wd = g.groupby("weekday").chg.agg(["mean", "size"])
    trips_wd = tx.groupby("weekday").BASKET_ID.nunique()

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))

    ax = axes[0]
    cols = [RED if d == 1 else GREY for d in by_wd.index]
    ax.bar(by_wd.index, by_wd["mean"], color=cols)
    ax.set_xticks(range(7)); ax.set_xticklabels(DAYNAMES)
    ax.set_ylabel("P(price moves > 2c from the previous day)")
    ax.set_title("Prices reset on Monday\n(the first day of WEEK_NO)", fontsize=10)
    for d, v, n in zip(by_wd.index, by_wd["mean"], by_wd["size"]):
        ax.text(d, v + 0.012, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_ylim(0, by_wd["mean"].max() * 1.22)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    haz = g.groupby(g.WEEK_NO != g.prevwk).chg.agg(["mean", "size"])
    ax.bar([0, 1], [haz.loc[False, "mean"], haz.loc[True, "mean"]], color=[GREY, RED], width=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["inside a week", "across the\nSun$\\rightarrow$Mon boundary"])
    ax.set_ylabel("P(price change)")
    for x, k in zip([0, 1], [False, True]):
        ax.text(x, haz.loc[k, "mean"] + 0.012,
                f"{haz.loc[k,'mean']:.3f}\n(n={int(haz.loc[k,'size']):,})", ha="center", fontsize=9)
    ax.set_ylim(0, haz["mean"].max() * 1.3)
    ax.set_title("The identification window", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    cols = [RED if d in (0, 1) else GREY for d in trips_wd.index]
    ax.bar(trips_wd.index, trips_wd.values / 1000, color=cols)
    ax.set_xticks(range(7)); ax.set_xticklabels(DAYNAMES)
    ax.set_ylabel("shopping trips (thousands)")
    share = trips_wd.loc[[0, 1]].sum() / trips_wd.sum()
    ax.set_title(f"Sunday + Monday = {share:.1%} of trips\n(paper's Tue+Wed: 30.1%)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    save(fig, "week_structure.png")
    return {"hazard_within": float(haz.loc[False, "mean"]),
            "hazard_boundary": float(haz.loc[True, "mean"]),
            "sun_mon_trip_share": float(share)}


# ---------------------------------------------------------------- fig 3
def fig_cross_store(tx):
    top = tx.groupby("PRODUCT_ID").size().sort_values(ascending=False).index[:400]
    sub = tx[tx.PRODUCT_ID.isin(top)]
    pw = sub.groupby(["PRODUCT_ID", "WEEK_NO", "STORE_ID"]).unit_price.median().reset_index()
    agg = pw.groupby(["PRODUCT_ID", "WEEK_NO"]).unit_price.agg(["median", "std", "size", "min", "max"])
    agg = agg[agg["size"] >= 5]
    cv = (agg["std"] / agg["median"]).dropna()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.hist(np.clip(cv, 0, 0.5), bins=50, color=BLUE, alpha=0.85, edgecolor="white")
    ax.axvline(cv.median(), color=RED, ls="--", lw=1.5,
               label=f"median {cv.median():.3f}")
    ax.set_xlabel("coefficient of variation of price across stores,\nwithin product $\\times$ week")
    ax.set_ylabel("product-weeks")
    ax.legend(fontsize=9)
    ax.set_title(f"Prices are close to chain-uniform\n({len(agg):,} product-weeks, $\\geq$5 stores)",
                 fontsize=10)

    ax = axes[1]
    rng = (agg["max"] - agg["min"])
    xs = np.linspace(0, 1.0, 100)
    ax.plot(xs, [(rng <= x).mean() for x in xs], color=BLUE, lw=2)
    for t in [0.05, 0.10, 0.25]:
        ax.axvline(t, color=GREY, ls=":", lw=1)
        ax.text(t, 0.04, f" {(rng <= t).mean():.0%} within ${t:.2f}", fontsize=8, rotation=90)
    ax.set_xlabel("max $-$ min price across stores, within product $\\times$ week (\\$)")
    ax.set_ylabel("cumulative share of product-weeks")
    ax.set_ylim(0, 1)
    ax.set_title("Spread of store prices", fontsize=10)
    ax.grid(alpha=0.3)
    save(fig, "cross_store_prices.png")
    return {"cv_median": float(cv.median()), "cells": int(len(agg))}


# ---------------------------------------------------------------- fig 4
def fig_random_weight(tx, prod):
    # exactly the screen the pipeline applies, imported rather than reimplemented
    from importlib import import_module
    ms = import_module("02_select_sample").price_discreteness(tx).rename("modal_share")
    ms = ms.to_frame().join(prod.set_index("PRODUCT_ID")[["COMMODITY_DESC", "DEPARTMENT"]])

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    ax = axes[0]
    ax.hist(ms.modal_share, bins=40, color=BLUE, alpha=0.85, edgecolor="white")
    ax.axvline(0.60, color=RED, ls="--", lw=1.6, label="screen at 0.60")
    ax.set_xlabel("share of a product-week's buyers at the modal cent value")
    ax.set_ylabel("products")
    ax.legend(fontsize=9)
    ax.set_title(f"Price discreteness separates posted prices from scale items\n"
                 f"{(ms.modal_share < 0.60).mean():.0%} of {len(ms):,} priced products fail",
                 fontsize=10)

    ax = axes[1]
    cat = ms.groupby("COMMODITY_DESC").modal_share.agg(["mean", "size"])
    cat = cat[cat["size"] >= 8].sort_values("mean")
    show = pd.concat([cat.head(10), cat.tail(10)])
    cols = [RED if v < 0.6 else GREEN for v in show["mean"]]
    ax.barh(range(len(show)), show["mean"], color=cols)
    ax.set_yticks(range(len(show)))
    ax.set_yticklabels([s[:26] for s in show.index], fontsize=7.5)
    ax.axvline(0.60, color="0.3", ls="--", lw=1.2)
    ax.set_xlabel("mean modal-price share")
    ax.set_xlim(0, 1)
    ax.set_title("Ten lowest and ten highest commodities\n"
                 "(low = weight $\\times$ price-per-pound, not a posted price)", fontsize=10)
    save(fig, "random_weight_screen.png")
    return {"fail_share": float((ms.modal_share < 0.60).mean()), "products": int(len(ms))}


# ---------------------------------------------------------------- fig 5
def fig_holidays(tx, sessions):
    ws = tx.groupby("WEEK_NO").SALES_VALUE.sum()
    base = ws.rolling(9, center=True, min_periods=3).median()
    dev = (ws - base) / base
    flagged = set(dev[dev.abs() > 0.12].index)
    kept = set(sessions.WEEK_NO)

    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.plot(ws.index, ws.values / 1000, color=GREY, lw=1.2, label="weekly spend")
    ax.plot(base.index, base.values / 1000, color="0.35", ls="--", lw=1,
            label="local 9-week median")
    f = sorted(flagged)
    ax.scatter(f, ws.loc[f].values / 1000, color=RED, zorder=5, s=34,
               label=f"flagged (|deviation| > 12%): {len(f)} weeks")
    for w in f:
        ax.annotate(str(w), (w, ws.loc[w] / 1000), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7, color=RED)
    ax.set_xlabel("WEEK_NO")
    ax.set_ylabel("chain spend, \\$000")
    ax.set_title("dunnhumby days are anonymised, so holiday weeks are flagged from "
                 "aggregate spend rather than a calendar", fontsize=10)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    save(fig, "holiday_weeks.png")
    return {"flagged_weeks": sorted(int(x) for x in flagged)}


# ---------------------------------------------------------------- fig 6
def fig_funnel(audit):
    audit = audit.copy()
    audit["drop_reason"] = audit.drop_reason.fillna("KEPT").replace("", "KEPT")
    order = ["fewer than 5 items or <200 category-trips",
             "unit demand violated (>15% multi-item or >10% multi-top-item trips)",
             "top 15% most seasonal (Herfindahl of daily demand)",
             "insufficient price variation (<2 items changing, or no item with "
             ">=10% of pair-weeks moving >=$0.10)"]
    counts = audit.drop_reason.value_counts()
    order = [o for o in order if o in counts.index]
    total = len(audit)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    ax = axes[0]
    labels = ["all commodities"] + [o.split("(")[0].strip() for o in order] + ["kept"]
    vals = [total]
    run = total
    for o in order:
        run -= counts[o]
        vals.append(run)
    vals.append(counts.get("KEPT", run))
    cols = [GREY] + [RED] * len(order) + [GREEN]
    ax.bar(range(len(vals)), vals, color=cols)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels([l if len(l) < 26 else l[:24] + "…" for l in labels],
                       rotation=28, ha="right", fontsize=7.5)
    for i, v in enumerate(vals):
        ax.text(i, v + 3, str(int(v)), ha="center", fontsize=9)
    ax.set_ylabel("categories remaining")
    ax.set_title("Category filter waterfall (paper app. 8.1 order)", fontsize=10)

    ax = axes[1]
    kept = audit.drop_reason == "KEPT"
    ax.scatter(audit.loc[~kept, "multi_any"], audit.loc[~kept, "multi_top"],
               s=14, color=GREY, alpha=0.6, label="dropped")
    ax.scatter(audit.loc[kept, "multi_any"], audit.loc[kept, "multi_top"],
               s=20, color=GREEN, label="kept")
    ax.axvline(0.15, color=RED, ls="--", lw=1.2)
    ax.axhline(0.10, color=RED, ls="--", lw=1.2)
    ax.set_xlabel("share of category-trips buying $\\geq$2 distinct items")
    ax.set_ylabel("share buying $\\geq$2 of the top 10")
    ax.set_xlim(0, 0.6); ax.set_ylim(0, 0.35)
    ax.legend(fontsize=9)
    ax.set_title("Unit-demand screen", fontsize=10)
    ax.grid(alpha=0.3)
    save(fig, "category_funnel.png")


# ---------------------------------------------------------------- fig 7
def fig_sample(trips_all, sample_trips, obs, items):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0))
    ax = axes[0]
    n = trips_all.groupby("household_key").size()
    ax.hist(np.clip(n, 0, 350), bins=45, color=GREY, alpha=0.9, edgecolor="white")
    ax.axvline(20, color=RED, ls="--", lw=1.4)
    ax.axvline(300, color=RED, ls="--", lw=1.4)
    ax.set_xlabel("shopping trips over the 2 years")
    ax.set_ylabel("households")
    ax.set_title(f"Household screen: 20-300 trips\nkeeps {((n>=20)&(n<=300)).sum():,} "
                 f"of {len(n):,}", fontsize=10)

    ax = axes[1]
    b = sample_trips.groupby(["household_key"]).n_lines.mean()
    ax.hist(np.clip(b, 0, 60), bins=40, color=BLUE, alpha=0.85, edgecolor="white")
    ax.set_xlabel("mean lines per trip (Sun/Mon sample)")
    ax.set_ylabel("households")
    ax.set_title("Basket size", fontsize=10)

    ax = axes[2]
    n_trips = obs.drop_duplicates(["user_id", "session_id"]).shape[0]
    rate = obs.groupby("item_id").size() / n_trips
    cat_rate = obs.merge(items[["item_id", "group_id"]], on="item_id") \
        .groupby("group_id").size() / n_trips
    ax.hist(np.log10(np.clip(rate, 1e-5, None)), bins=35, color=GREY, alpha=0.8,
            label=f"UPC (mean {rate.mean():.4f})", edgecolor="white")
    ax.hist(np.log10(np.clip(cat_rate, 1e-5, None)), bins=25, color=GREEN, alpha=0.75,
            label=f"category (mean {cat_rate.mean():.4f})", edgecolor="white")
    ax.set_xlabel("$\\log_{10}$ purchase rate per trip")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    ax.set_title("Choice is sparse\n(paper: 3.7% per category, 0.36% per UPC)", fontsize=10)
    save(fig, "sample_profile.png")
    return {"mean_cat_rate": float(cat_rate.mean()), "mean_upc_rate": float(rate.mean())}


# ---------------------------------------------------------------- fig 8
def fig_price_panel(price_panel, events, items):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))

    ax = axes[0]
    obs_share = price_panel.groupby("PRODUCT_ID").price_obs.mean()
    ax.hist(obs_share, bins=30, color=BLUE, alpha=0.85, edgecolor="white")
    ax.axvline(obs_share.mean(), color=RED, ls="--", lw=1.5,
               label=f"mean {obs_share.mean():.2f}")
    ax.set_xlabel("share of the item's sessions with a directly observed price")
    ax.set_ylabel("items")
    ax.legend(fontsize=9)
    ax.set_title("Prices are inferred from transactions;\nthe rest is carried forward",
                 fontsize=10)

    ax = axes[1]
    dp = events.dp
    ax.hist(np.clip(dp[dp.abs() > 0.005], -1.5, 1.5), bins=60, color=BLUE,
            alpha=0.85, edgecolor="white")
    ax.axvline(0, color="0.3", lw=1)
    ax.set_xlabel("Sunday $\\rightarrow$ Monday price change (\\$)")
    ax.set_ylabel("item $\\times$ pair-weeks")
    ax.set_title(f"{int((dp.abs() >= 0.10).sum()):,} moves of $\\geq$\\$0.10\n"
                 f"out of {len(dp):,} item $\\times$ pair-weeks", fontsize=10)

    ax = axes[2]
    per_item = events.groupby("item_id").own_price_change.sum()
    ax.hist(per_item, bins=30, color=GREEN, alpha=0.85, edgecolor="white")
    ax.set_xlabel("pair-weeks with an own-price move $\\geq$\\$0.10")
    ax.set_ylabel("items")
    ax.set_title(f"Every retained item has variation\n(median {int(per_item.median())} "
                 f"of {events.pair_week.nunique()} pair-weeks)", fontsize=10)
    save(fig, "price_panel.png")


def main():
    os.makedirs(FIG, exist_ok=True)
    tx = pd.read_parquet(os.path.join(DATA, "tx.parquet"))
    prod = pd.read_csv(RAW + "product.csv")
    trips_all = pd.read_parquet(os.path.join(DATA, "trips.parquet"))
    sample_trips = pd.read_parquet(os.path.join(DATA, "sample_trips.parquet"))
    price_panel = pd.read_parquet(os.path.join(DATA, "price_panel.parquet"))
    sessions = pd.read_parquet(os.path.join(DATA, "sessions.parquet"))
    audit = pd.read_csv(os.path.join(DATA, "filter_audit.csv"))
    obs = pd.read_csv(os.path.join(MI, "id_maps", "observations.csv"))
    items = pd.read_csv(os.path.join(MI, "id_maps", "items.csv"))
    events = pd.read_csv(os.path.join(MI, "events.csv"))

    stats = {}
    stats.update(fig_week_structure(tx))
    stats.update(fig_cross_store(tx))
    stats.update(fig_random_weight(tx, prod))
    stats.update(fig_holidays(tx, sessions))
    fig_funnel(audit)
    stats.update(fig_sample(trips_all, sample_trips, obs, items))
    fig_price_panel(price_panel, events, items)

    with open(os.path.join(OUT, "figure_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    log("wrote out/figure_stats.json")


if __name__ == "__main__":
    main()
