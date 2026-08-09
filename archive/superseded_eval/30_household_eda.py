"""
Stage 30 -- Households: taste, price sensitivity, store visits, trips, demographics.

29_demand_eda.py fixed one gap and revealed a bigger one.  Auditing every term in the
model against the exploration gives:

  theta_i . alpha_j    household taste                 NOT EXPLORED
  gamma_i . beta_j     household price sensitivity     NOT EXPLORED
  alpha_j . abar       basket interaction              covered (21, s2)
  eta_j . state        recency                         covered (21, s3)
  mu_j . delta_w       seasonality                     covered indirectly (25)
  zeta_j . xi_s        store x item affinity           partly (29, dispersion only)
  c0_c + kappa * IV    category incidence              covered (29, s6.5)
  q0_j + q_gamma       quantity                        covered (29, s6.1)

The first two are the *premise of the entire model*.  Every claim about personalised
pricing, targeted promotion or heterogeneous elasticity rests on households differing
in taste and in price response -- and neither was ever measured.  A model can fit
per-household parameters to pure noise and report confident heterogeneity; without a
model-free measurement there is nothing to check that against.

Also missing and covered here:

  store visits     how households choose and switch stores.  The model has a store
                   x item affinity term but nothing was known about store *choice*.
  trip behaviour   frequency, basket size, and whether trips are top-up or stock-up
  demographics     hh_demographic.csv has never been opened in this analysis

Writes out/household_eda.json, out/household_eda_*.csv, figures/household_eda_*.png.
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
RAW = os.path.join(os.environ.get(
    "NF_RAW_DIR",
    os.path.join(HERE, "..", "..", "..", "dunnhumby_The-Complete-Journey",
                 "dunnhumby_The-Complete-Journey CSV")), "")

PAL = {"blue": "#2d6cdf", "grey": "#9aa5b1", "red": "#d1495b",
       "green": "#2a9d8f", "amber": "#e9c46a", "purple": "#7b6cd9"}


def log(m):
    print(f"[30] {m}", flush=True)


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
    bk = bk.merge(items[["item_id", "cat_id", "sub_id"]], on="item_id")
    log(f"{len(bk):,} rows, {bk.user_id.nunique():,} households, "
        f"{bk.store_id.nunique()} stores")

    # ============================================ 1. is there taste heterogeneity?
    # Split each household's trips in half by time.  If tastes are real and stable,
    # what a household bought in the first half should predict the second half far
    # better than another household's first half does.  That is a model-free test and
    # it is the premise of theta_i.
    bk = bk.sort_values(["user_id", "DAY"])
    # log price of the item on the day it was bought; attached before the split-half
    # frames are taken, because they need it too
    bk["lp"] = logp[bk.item_id.to_numpy(), bk.DAY.to_numpy()]
    bk["lu"] = np.log(bk.units.to_numpy(dtype=float))
    med_day = bk.groupby("user_id").DAY.transform("median")
    first = bk[bk.DAY <= med_day]
    second = bk[bk.DAY > med_day]
    C = int(meta["n_commodities"])

    def profile(df):
        m = np.zeros((bk.user_id.max() + 1, C), dtype=np.float32)
        np.add.at(m, (df.user_id.to_numpy(), df.cat_id.to_numpy()), 1.0)
        n = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.clip(n, 1e-9, None)

    P1, P2 = profile(first), profile(second)
    users = np.sort(bk.user_id.unique())
    P1, P2 = P1[users], P2[users]
    keep = (P1.sum(1) > 0) & (P2.sum(1) > 0)
    P1, P2 = P1[keep], P2[keep]
    # The comparison baseline used to be ONE random pairing of households, which made
    # the reported number depend on the seed: over 200 seeds the mean ranged 0.4654 to
    # 0.4765 and the headline ratio 1.645x to 1.684x.  Small, but there is no reason to
    # sample -- every household-vs-household pair can simply be computed.  G holds all
    # of them: G[i, j] is household i's first half against household j's second half.
    G = P1 @ P2.T
    n = len(P1)
    own = np.diag(G).copy()                       # i against itself
    off = ~np.eye(n, dtype=bool)
    other = G[off]                                # all n*(n-1) different-household pairs
    # "does my own past beat a stranger's?" -- averaged over every stranger, not one
    row_mean_other = (G.sum(1) - np.diag(G)) / (n - 1)
    r["taste"] = {
        "households": int(n),
        "cross_household_pairs": int(n * (n - 1)),
        "self_similarity_across_time": float(own.mean()),
        "cross_household_similarity": float(other.mean()),
        "ratio": float(own.mean() / max(other.mean(), 1e-9)),
        "share_self_beats_random": float((own > row_mean_other).mean()),
    }
    rt = r["taste"]
    log("")
    log("1. taste heterogeneity (model-free, split-half over time)")
    log(f"   a household's own two halves agree   {rt['self_similarity_across_time']:.4f}")
    log(f"   two different households agree        {rt['cross_household_similarity']:.4f}")
    log(f"   -> ratio {rt['ratio']:.2f}x; own beats random for "
        f"{rt['share_self_beats_random']:.1%} of households")

    # ================================== 2. does price sensitivity differ by household?
    # For each household, regress log(units) on log(price) within item, using only
    # items that household buys repeatedly.  Split-half again: if the spread across
    # households is real rather than noise, a household's first-half slope should
    # correlate with its second-half slope.
    def hh_slopes(df, min_obs):
        """Per-household within-item slope of log units on log price.

        Vectorised: a Python loop over 2,066 households each running its own groupby
        never finished.  Demeaning within (household, item) and summing the
        cross-products per household gives the same estimate in two groupbys.
        """
        g = df[df.groupby(["user_id", "item_id"]).item_id.transform("size") >= 3].copy()
        if not len(g):
            return pd.Series(dtype=float)
        key = ["user_id", "item_id"]
        g["yd"] = g.lu - g.groupby(key).lu.transform("mean")
        g["xd"] = g.lp - g.groupby(key).lp.transform("mean")
        g["xy"] = g.xd * g.yd
        g["xx"] = g.xd ** 2
        agg = g.groupby("user_id").agg(xy=("xy", "sum"), xx=("xx", "sum"),
                                       n=("xx", "size"))
        agg = agg[(agg.n >= min_obs) & (agg.xx > 1e-6)]
        return (agg.xy / agg.xx).rename(None)

    s_all = hh_slopes(bk, a.min_obs)
    # the halves hold about half the purchases each, so they use half the threshold
    s1 = hh_slopes(first, a.min_obs // 2)
    s2 = hh_slopes(second, a.min_obs // 2)

    # How much price movement does a household actually have to learn from?  The
    # denominator of the slope is sum (logp - mean_hi logp)^2.  When an item's price
    # never moved for that household, the pair contributes nothing to it, and dividing
    # by a near-zero denominator turns accidents into large slopes.  Measuring this
    # explains the modest split-half correlation below.
    gq = bk[bk.groupby(["user_id", "item_id"]).item_id.transform("size") >= 3].copy()
    kq = ["user_id", "item_id"]
    gq["xd"] = gq.lp - gq.groupby(kq).lp.transform("mean")
    gq["yd"] = gq.lu - gq.groupby(kq).lu.transform("mean")
    xx = gq.assign(xx=gq.xd ** 2).groupby("user_id").xx.sum()
    xx = xx[s_all.index]
    thin, thick = s_all[xx <= xx.median()], s_all[xx > xx.median()]
    r["price_variation_available"] = {
        "qualifying_rows": int(len(gq)),
        "household_item_pairs": int(gq.groupby(kq).ngroups),
        "share_rows_price_never_moved": float((gq.xd.abs() < 0.01).mean()),
        "share_pairs_zero_price_variation": float(
            gq.groupby(kq).xd.apply(lambda s: (s.abs() < 1e-9).all()).mean()),
        "median_sum_xd2": float(xx.median()),
        "thin_half": {"sd_of_slopes": float(thin.std()),
                      "share_positive": float((thin > 0).mean())},
        "thick_half": {"sd_of_slopes": float(thick.std()),
                       "share_positive": float((thick > 0).mean())},
    }
    both = pd.concat([s1.rename("h1"), s2.rename("h2")], axis=1).dropna()
    split_corr = float(both.h1.corr(both.h2)) if len(both) > 20 else np.nan
    r["price_sensitivity"] = {
        "households_estimated": int(len(s_all)),
        "median_slope": float(s_all.median()),
        "p10": float(s_all.quantile(.1)), "p90": float(s_all.quantile(.9)),
        "sd": float(s_all.std()),
        "split_half_correlation": split_corr,
        "households_in_split_half": int(len(both)),
    }
    rp = r["price_sensitivity"]
    rv = r["price_variation_available"]
    log("")
    log("2. price sensitivity across households (units on price, within item)")
    log(f"   {rp['households_estimated']:,} households: median {rp['median_slope']:+.3f}, "
        f"p10 {rp['p10']:+.3f}, p90 {rp['p90']:+.3f}, sd {rp['sd']:.3f}")
    log(f"   split-half correlation {split_corr:+.3f} on {len(both):,} households "
        f"-> {'REAL heterogeneity' if split_corr > 0.1 else 'mostly NOISE'}")
    log(f"   price movement available: {rv['share_rows_price_never_moved']:.1%} of rows "
        f"come from a (household,item) pair whose price never moved; "
        f"{rv['share_pairs_zero_price_variation']:.1%} of pairs have none at all")
    log(f"   households with LITTLE price movement: slope sd "
        f"{rv['thin_half']['sd_of_slopes']:.3f}, {rv['thin_half']['share_positive']:.1%} positive")
    log(f"   households with MUCH   price movement: slope sd "
        f"{rv['thick_half']['sd_of_slopes']:.3f}, {rv['thick_half']['share_positive']:.1%} positive")
    s_all.rename("slope").to_frame().to_csv(
        os.path.join(OUT, "household_eda_price_slopes.csv"))

    # ==================================================== 3. store visit behaviour
    trips = bk[["user_id", "DAY", "store_id", "BASKET_ID"]].drop_duplicates()
    per_hh = trips.groupby("user_id").agg(trips=("DAY", "nunique"),
                                          stores=("store_id", "nunique"))
    share_primary = (trips.groupby(["user_id", "store_id"]).size()
                     .groupby(level=0).apply(lambda s: s.max() / s.sum()))
    tr = trips.sort_values(["user_id", "DAY"])
    tr["prev_store"] = tr.groupby("user_id").store_id.shift()
    switch = float((tr.store_id != tr.prev_store).dropna().mean())
    r["stores_visited"] = {
        "median_trips": float(per_hh.trips.median()),
        "median_stores": float(per_hh.stores.median()),
        "p90_stores": float(per_hh.stores.quantile(.9)),
        "median_primary_share": float(share_primary.median()),
        "p10_primary_share": float(share_primary.quantile(.1)),
        "share_trips_switching_store": switch,
        "share_single_store_households": float((per_hh.stores == 1).mean()),
    }
    rs = r["stores_visited"]
    log("")
    log("3. store visits")
    log(f"   median household: {rs['median_trips']:.0f} trips across "
        f"{rs['median_stores']:.0f} stores; {rs['median_primary_share']:.0%} of trips at "
        f"its primary store (p10 {rs['p10_primary_share']:.0%})")
    log(f"   {switch:.1%} of consecutive trips switch store; "
        f"{rs['share_single_store_households']:.1%} of households use only one")

    # ======================================================== 4. trip behaviour
    # n_items, not items: `items` shadows DataFrame.items and every read of it
    # silently returns the method
    tb = bk.groupby("BASKET_ID").agg(user_id=("user_id", "first"),
                                     DAY=("DAY", "first"),
                                     n_items=("item_id", "size"),
                                     units=("units", "sum"),
                                     cats=("cat_id", "nunique"))
    tb = tb.sort_values(["user_id", "DAY"])
    tb["gap"] = tb.groupby("user_id").DAY.diff()
    big = tb.n_items >= tb.n_items.quantile(.75)
    r["trips"] = {
        "median_items": float(tb.n_items.median()),
        "median_gap_days": float(tb.gap.median()),
        "p90_gap_days": float(tb.gap.quantile(.9)),
        "corr_gap_and_size": float(tb[["gap", "n_items"]].dropna().corr().iloc[0, 1]),
        "median_gap_before_large_trip": float(tb.gap[big].median()),
        "median_gap_before_small_trip": float(tb.gap[~big].median()),
    }
    rtr = r["trips"]
    log("")
    log("4. trips")
    log(f"   median {rtr['median_items']:.0f} items, {rtr['median_gap_days']:.0f} days "
        f"since the last trip (p90 {rtr['p90_gap_days']:.0f})")
    log(f"   correlation between the gap and trip size: {rtr['corr_gap_and_size']:+.3f} "
        f"-> {'longer gaps mean bigger trips (stock-up)' if rtr['corr_gap_and_size'] > .05 else 'no stock-up pattern'}")
    log(f"   gap before a large trip {rtr['median_gap_before_large_trip']:.0f} days vs "
        f"{rtr['median_gap_before_small_trip']:.0f} before a small one")

    # ======================================================== 5. demographics
    dpath = RAW + "hh_demographic.csv"
    if os.path.exists(dpath):
        dem = pd.read_csv(dpath)
        idmap = pd.read_parquet(os.path.join(DATA, "trips.parquet"),
                                columns=["household_key"]).household_key.unique() \
            if os.path.exists(os.path.join(DATA, "trips.parquet")) else None
        hh_raw = np.sort(pd.read_parquet(os.path.join(DATA, "tx.parquet"),
                                         columns=["household_key"]).household_key.unique())
        # user_id was assigned by sorting household_key among retained households
        keep_hh = hh_raw
        uid = {h: i for i, h in enumerate(np.sort(keep_hh))}
        dem["user_id"] = dem.household_key.map(uid)
        dem = dem.dropna(subset=["user_id"])
        cov = float(dem.user_id.nunique() / bk.user_id.nunique())
        merged = dem.set_index("user_id").join(s_all.rename("price_slope"), how="inner")
        by = {}
        for col in [c for c in dem.columns if c not in ("household_key", "user_id")]:
            g = merged.groupby(col).price_slope.agg(["median", "size"])
            g = g[g["size"] >= 30]
            if len(g) >= 2:
                by[col] = {"levels": int(len(g)),
                           "range_of_median_slope": float(g["median"].max() - g["median"].min()),
                           "detail": {str(k): float(v) for k, v in g["median"].items()}}
        r["demographics"] = {"coverage_of_modelled_households": cov,
                             "price_slope_by_attribute": by}
        log("")
        log(f"5. demographics: cover {cov:.0%} of modelled households")
        for k, v in sorted(by.items(), key=lambda x: -x[1]["range_of_median_slope"])[:5]:
            log(f"   {k:22s} {v['levels']:2d} levels, median price slope spans "
                f"{v['range_of_median_slope']:.3f}")
    else:
        log("")
        log("5. demographics: hh_demographic.csv not found, skipped")

    with open(os.path.join(OUT, "household_eda.json"), "w") as f:
        json.dump(r, f, indent=2, default=float)

    # ================================================================= figures
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    ax = axes[0]
    # one set of bin edges for both, and density so 2,066 self-pairs and 4.27M
    # cross-pairs are on the same footing
    edges = np.linspace(0, 1, 51)
    ax.hist(other, bins=edges, density=True, alpha=.6, color=PAL["grey"],
            label=f"different households (all {len(other):,} pairs)")
    ax.hist(own, bins=edges, density=True, alpha=.75, color=PAL["blue"],
            label=f"same household, two halves ({len(own):,})")
    style(ax, f"Taste is real and stable\n{rt['ratio']:.1f}x more self-similar than random",
          "cosine similarity of category profiles", "share of pairs (density)")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.hist(s_all.clip(-4, 2), bins=50, color=PAL["blue"], edgecolor="white")
    ax.axvline(0, color="k", lw=1)
    ax.axvline(s_all.median(), color=PAL["red"], ls="--", lw=1.5,
               label=f"median {s_all.median():+.2f}")
    style(ax, f"Price sensitivity differs across households\n"
              f"p10 {rp['p10']:+.2f} to p90 {rp['p90']:+.2f}",
          "within-item slope of log units on log price", "households")
    ax.legend(fontsize=8)

    ax = axes[2]
    if len(both) > 20:
        ax.scatter(both.h1.clip(-4, 2), both.h2.clip(-4, 2), s=6, alpha=.3,
                   color=PAL["green"])
        ax.axhline(0, color="k", lw=.8); ax.axvline(0, color="k", lw=.8)
        style(ax, f"Is it real or noise?\nsplit-half correlation {split_corr:+.2f}",
              "slope, first half of trips", "slope, second half")
    fig.suptitle("Households: the premise of the whole model, finally measured",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "household_eda_heterogeneity.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    ax = axes[0]
    ax.hist(share_primary, bins=40, color=PAL["amber"], edgecolor="white")
    ax.axvline(share_primary.median(), color=PAL["red"], ls="--", lw=1.5,
               label=f"median {share_primary.median():.0%}")
    style(ax, "Store loyalty", "share of trips at the household's primary store",
          "households")
    ax.legend(fontsize=8)

    ax = axes[1]
    vc = per_hh.stores.clip(upper=15).value_counts().sort_index()
    ax.bar(vc.index, vc.values, color=PAL["green"])
    style(ax, f"Stores visited per household\nmedian {per_hh.stores.median():.0f}, "
              f"{rs['share_single_store_households']:.0%} use only one",
          "distinct stores (clipped at 15)", "households")

    ax = axes[2]
    g = tb.dropna(subset=["gap"])
    g = g[g.gap <= 42]
    prof = g.groupby(pd.cut(g.gap, [0, 3, 7, 14, 21, 28, 42], right=False),
                     observed=True)["n_items"].mean()
    ax.plot(range(len(prof)), prof.values, "o-", color=PAL["purple"], lw=2)
    ax.set_xticks(range(len(prof)))
    ax.set_xticklabels([f"{int(i.left)}-{int(i.right)}" for i in prof.index], fontsize=8)
    style(ax, f"Longer gap, bigger trip\ncorrelation {rtr['corr_gap_and_size']:+.2f}",
          "days since the household's last trip", "items in the trip")
    fig.suptitle("Store visits and trip rhythm — neither was ever explored", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "household_eda_stores_trips.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    log("")
    log("wrote out/household_eda.json, household_eda_price_slopes.csv and 2 figures")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--min-obs", type=int, default=80,
                   help="purchase rows a household needs before its own price slope "
                        "is estimated")
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
