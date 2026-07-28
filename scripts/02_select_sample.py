"""
Stage 2 -- Sample construction and the paper's category filters.

Design decisions, and how they map onto Donnelly et al. (2023) sec. 4.1 / app. 8.1
---------------------------------------------------------------------------------
* Identification window.  In the paper almost all price changes happen at midnight
  on Tuesday, so Tuesday/Wednesday straddle the change and the estimation sample is
  restricted to those two days.  In dunnhumby, WEEK_NO runs Monday->Sunday and the
  price-change hazard at the Sunday->Monday boundary is 51.9% vs 26.3% for a
  within-week day pair (see report).  The exact analogue is therefore the
  (Sunday of week w, Monday of week w+1) pair, which we call a *pair-week*.
* Households: keep those with 20-300 trips overall (paper: 2068 households).
* Category = COMMODITY_DESC (the paper's "category"); SUB_COMMODITY_DESC, BRAND and
  MANUFACTURER are deliberately withheld from the model so they can be used, as in
  paper sec. 6.4.1, to test whether cross-price elasticities are higher inside a
  subclass than across subclasses.
* Session = calendar day.  bemb_loc requires prices to be common to all users in a
  session, and it stores a dense Nitems x Nsessions price matrix, so a chain-level
  daily price is used (cross-store dispersion within a product-week has median
  CV = 1.1%).  A store-restricted variant is available with --stores.

Outputs (in ../data):
    sample_trips.parquet     retained trips (household, day) with session ids
    sample_choices.parquet   one row per (trip, category) with the chosen item
    categories.parquet       retained categories + diagnostics
    items.parquet            retained items with category / subclass / brand
    filter_audit.csv         why every category was kept or dropped
    price_panel.parquet      item x session price (complete, carried forward)
"""
import argparse
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
# Raw dunnhumby CSVs.  Defaults to a sibling of the repository; override with
# NF_RAW_DIR if the download lives somewhere else.
RAW = os.path.join(os.environ.get(
    "NF_RAW_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                 "dunnhumby_The-Complete-Journey",
                 "dunnhumby_The-Complete-Journey CSV")), "")

# Departments that are not shoppable merchandise with a unit price.
DROP_DEPARTMENTS = {
    "MISC. TRANS.", "MISC SALES TRAN", "COUP/STR & MFG", "KIOSK-GAS", "POSTAL CENTER",
    "CNTRL/STORE SUP", "PHOTO", "RX", "GM MERCH EXP", "VIDEO", "VIDEO RENTAL",
    "TRAVEL & LEISUR", "RESTAURANT", "GARDEN CENTER", "AUTOMOTIVE", "CHARITABLE CONT",
    "PROD-WHS SALES", "HBC", "TOYS", "ELECT &PLUMBING", "CNTRL/STORE SUP",
}
DROP_COMMODITY_PAT = (
    "NO COMMODITY DESCRIPTION|COUPON|MISCELLANEOUS|GASOLINE|LOTTERY|MONEY ORDER|"
    "POSTAGE|WESTERN UNION|TOBACCO|CIGAR|SERVICE|RENTAL|FUEL"
)


def log(m):
    print(f"[02] {m}", flush=True)


def herfindahl(v):
    v = np.asarray(v, dtype=float)
    s = v.sum()
    if s <= 0:
        return np.nan
    p = v / s
    return float((p ** 2).sum())


def price_discreteness(tx, thresholds=(5, 3, 2)):
    """Share of transactions at the modal cent value, averaged over product-weeks.

    Close to 1 for a posted shelf price; close to 0 for random-weight items, where
    the recorded 'unit price' is really weight x price-per-pound and so varies
    continuously across shoppers facing the same shelf.

    The measure must be taken *within* a week.  Pooling a product's transactions
    across the whole panel and asking what share sit at the modal value conflates
    two completely different things: a scale item, and an ordinary posted-price item
    whose price simply changed a few times over two years.  Doing that flagged 54%
    of thin products and 1,989 high-volume products as random-weight when they were
    nothing of the kind.  Products too thin for the strictest threshold fall back to
    a looser one, never to a cross-week pool.
    """
    d = tx[["PRODUCT_ID", "WEEK_NO", "unit_price"]].copy()
    d["cents"] = np.round(d.unit_price, 2)
    g = d.groupby(["PRODUCT_ID", "WEEK_NO"])
    w = pd.concat([g.size().rename("n"),
                   g.cents.agg(lambda s: s.value_counts().iat[0]).rename("modal")], axis=1)
    out = None
    for t in thresholds:
        sub = w[w.n >= t]
        if not len(sub):
            continue
        share = (sub.modal / sub.n).groupby(level="PRODUCT_ID").mean()
        out = share if out is None else out.reindex(out.index.union(share.index)).fillna(share)
    return out.dropna() if out is not None else pd.Series(dtype=float)


def main(cfg):
    tx = pd.read_parquet(os.path.join(DATA, "tx.parquet"))
    trips = pd.read_parquet(os.path.join(DATA, "trips.parquet"))
    prod = pd.read_csv(RAW + "product.csv")
    prod.columns = [c.strip() for c in prod.columns]

    # ------------------------------------------------------------- pair-weeks
    # weekday: 0=Sun (last day of WEEK_NO w), 1=Mon (first day of w+1).
    # pair_week id = the WEEK_NO of the Sunday, so a Sunday in week w and the
    # Monday in week w+1 share pair_week = w.
    day_map = tx[["DAY", "WEEK_NO", "weekday"]].drop_duplicates().sort_values("DAY")
    day_map["pair_week"] = np.where(day_map.weekday == 0, day_map.WEEK_NO, day_map.WEEK_NO - 1)
    day_map = day_map[day_map.weekday.isin([0, 1])]
    # keep only complete pairs (both a Sunday and a Monday present)
    ok = day_map.groupby("pair_week").weekday.nunique() == 2
    day_map = day_map[day_map.pair_week.isin(ok[ok].index)]
    log(f"pair-weeks with both days present: {day_map.pair_week.nunique()}")

    # ------------------------------------------------- holiday-week exclusion
    # The paper drops the weeks before Halloween/Thanksgiving/Christmas/July 4th/
    # Labor Day.  dunnhumby days are anonymised, so flag them from the data:
    # weeks whose chain-wide spend deviates strongly from a local 9-week median.
    wk_spend = tx.groupby("WEEK_NO").SALES_VALUE.sum()
    base = wk_spend.rolling(9, center=True, min_periods=3).median()
    dev = (wk_spend - base) / base
    holiday_weeks = set(dev[dev.abs() > cfg.holiday_dev].index)
    # a pair-week straddles two calendar weeks; drop it if either is flagged
    pw_bad = {w for w in day_map.pair_week.unique() if w in holiday_weeks or (w + 1) in holiday_weeks}
    log(f"flagged holiday/anomalous weeks: {sorted(holiday_weeks)}")
    log(f"pair-weeks dropped for holidays: {len(pw_bad)}")
    day_map = day_map[~day_map.pair_week.isin(pw_bad)]
    keep_days = set(day_map.DAY)
    log(f"retained pair-weeks: {day_map.pair_week.nunique()}   sessions (days): {len(keep_days)}")

    # ------------------------------------------------------------- households
    ntrips = trips.groupby("household_key").size()
    hh = set(ntrips[(ntrips >= cfg.min_trips) & (ntrips <= cfg.max_trips)].index)
    log(f"households with {cfg.min_trips}-{cfg.max_trips} trips: {len(hh)}")

    # --------------------------------------------------------- sample of lines
    tx = tx.merge(prod[["PRODUCT_ID", "DEPARTMENT", "COMMODITY_DESC",
                        "SUB_COMMODITY_DESC", "BRAND", "MANUFACTURER",
                        "CURR_SIZE_OF_PRODUCT"]], on="PRODUCT_ID", how="left")
    tx = tx[~tx.DEPARTMENT.isin(DROP_DEPARTMENTS)]
    tx = tx[~tx.COMMODITY_DESC.str.upper().str.contains(DROP_COMMODITY_PAT, na=True)]

    # ---- drop pair-weeks with a near-empty day
    # dunnhumby has two calendar days (278 and 643) carrying a single basket
    # chain-wide: they are holes in the panel, not quiet trading days.  A pair-week
    # whose Sunday is empty cannot support a Sunday-to-Monday comparison at all, and
    # leaving it in silently puts a session in the price grid that no observation
    # ever references.  (Found by running the authors' C++ binary on the emitted
    # files, which warns about item-sessions it cannot match.)
    day_baskets = tx.groupby("DAY").BASKET_ID.nunique()
    thin_days = set(day_baskets[day_baskets < cfg.min_day_baskets].index)
    thin_pw = set(day_map.loc[day_map.DAY.isin(thin_days), "pair_week"])
    if thin_pw:
        log(f"dropping {len(thin_pw)} pair-week(s) containing a near-empty day "
            f"(<{cfg.min_day_baskets} baskets chain-wide): days "
            f"{sorted(thin_days & set(day_map.DAY))}")
        day_map = day_map[~day_map.pair_week.isin(thin_pw)]
        keep_days = set(day_map.DAY)
        log(f"retained pair-weeks: {day_map.pair_week.nunique()}   "
            f"sessions (days): {len(keep_days)}")

    smp = tx[tx.DAY.isin(keep_days) & tx.household_key.isin(hh)].copy()
    smp = smp.merge(day_map[["DAY", "pair_week"]], on="DAY", how="left")
    log(f"sample lines: {len(smp):,}   trips: {smp.groupby(['household_key','DAY']).ngroups:,}   "
        f"households: {smp.household_key.nunique():,}")

    # ------------------------------------------- random-weight ("scale") screen
    # dunnhumby-specific, with no counterpart in the paper.  For random-weight
    # items (loose produce, service-counter meat/seafood) QUANTITY is the number
    # of scans, so SALES_VALUE/QUANTITY is weight x price-per-pound, not a posted
    # price: it varies continuously across shoppers facing the same shelf price.
    # Such items would inject enormous measurement error into the price
    # coefficients, so screen them out on price discreteness -- the share of
    # transactions at the modal cent value within a product-week.
    disc = price_discreteness(tx)
    n_scale = (disc < cfg.min_modal_share).sum()
    log(f"random-weight screen: {n_scale:,} of {len(disc):,} priced items have modal "
        f"price share < {cfg.min_modal_share} and are excluded")
    scale_items = set(disc.index[disc < cfg.min_modal_share])
    smp = smp[~smp.PRODUCT_ID.isin(scale_items)]
    tx = tx[~tx.PRODUCT_ID.isin(scale_items)]

    # ------------------------------------------------------------ top-J items
    # Item popularity = number of sample trips containing the item.
    item_trips = smp.groupby(["COMMODITY_DESC", "PRODUCT_ID"]).apply(
        lambda d: d.groupby(["household_key", "DAY"]).ngroups, include_groups=False
    ).rename("n_trips").reset_index()
    item_trips = item_trips.sort_values(["COMMODITY_DESC", "n_trips"], ascending=[True, False])
    item_trips["rank"] = item_trips.groupby("COMMODITY_DESC").cumcount() + 1
    top = item_trips[item_trips["rank"] <= cfg.top_j].copy()

    cat_stats = item_trips.groupby("COMMODITY_DESC").agg(
        n_items_all=("PRODUCT_ID", "size"), trips_all=("n_trips", "sum")).reset_index()
    cat_stats = cat_stats.merge(
        top.groupby("COMMODITY_DESC").agg(n_top=("PRODUCT_ID", "size"),
                                          trips_top=("n_trips", "sum")).reset_index(),
        on="COMMODITY_DESC")
    log(f"categories before filters: {len(cat_stats)}")

    audit = cat_stats.set_index("COMMODITY_DESC").copy()
    audit["drop_reason"] = ""

    def mark(mask_index, reason):
        idx = [c for c in mask_index if audit.loc[c, "drop_reason"] == ""]
        audit.loc[idx, "drop_reason"] = reason
        return set(idx)

    # ---- filter 0: enough items and enough purchase volume to be modelled
    thin = audit.index[(audit.n_items_all < cfg.min_items) | (audit.trips_top < cfg.min_cat_trips)]
    mark(thin, f"fewer than {cfg.min_items} items or <{cfg.min_cat_trips} category-trips")

    # ---- filter 2 (paper app. 8.1 #2): unit demand
    smp_top = smp.merge(top[["PRODUCT_ID", "COMMODITY_DESC"]].assign(is_top=1),
                        on=["PRODUCT_ID", "COMMODITY_DESC"], how="left")
    trip_cat = smp_top.groupby(["household_key", "DAY", "COMMODITY_DESC"]).agg(
        n_distinct=("PRODUCT_ID", "nunique"),
        n_distinct_top=("is_top", "sum"),
        n_units=("QUANTITY", "sum"),
    ).reset_index()
    ud = trip_cat.groupby("COMMODITY_DESC").agg(
        cat_trips=("n_distinct", "size"),
        multi_any=("n_distinct", lambda s: (s > 1).mean()),
        multi_top=("n_distinct_top", lambda s: (s > 1).mean()),
        multi_units=("n_units", lambda s: (s > 1).mean()),
    )
    audit = audit.join(ud)
    viol = audit.index[(audit.multi_any > cfg.max_multi_any) | (audit.multi_top > cfg.max_multi_top)]
    mark(viol, f"unit demand violated (>{cfg.max_multi_any:.0%} multi-item or "
               f">{cfg.max_multi_top:.0%} multi-top-item trips)")

    # ------------------------------------------------- price panel (item x day)
    # Chain-level weekly median transacted unit price, carried forward across
    # weeks with no transactions, then mapped onto the retained sessions.
    pw = pd.read_parquet(os.path.join(DATA, "price_week.parquet"))
    top_ids = top.PRODUCT_ID.unique()
    pw = pw[pw.PRODUCT_ID.isin(top_ids)]
    weeks = np.arange(int(tx.WEEK_NO.min()), int(tx.WEEK_NO.max()) + 1)
    grid = pd.MultiIndex.from_product([top_ids, weeks], names=["PRODUCT_ID", "WEEK_NO"]).to_frame(index=False)
    pw = grid.merge(pw[["PRODUCT_ID", "WEEK_NO", "price", "n_tx"]], on=["PRODUCT_ID", "WEEK_NO"], how="left")
    pw = pw.sort_values(["PRODUCT_ID", "WEEK_NO"])
    pw["price_obs"] = pw.price.notna()
    pw["price"] = pw.groupby("PRODUCT_ID").price.ffill()
    pw["price"] = pw.groupby("PRODUCT_ID").price.bfill()
    log(f"price panel: {pw.price_obs.mean():.3f} of item-weeks directly observed")

    price_sess = day_map[["DAY", "WEEK_NO", "weekday", "pair_week"]].merge(
        pw[["PRODUCT_ID", "WEEK_NO", "price", "price_obs"]], on="WEEK_NO", how="left")
    price_sess = price_sess.dropna(subset=["price"])

    # ---- filter 3 (app. 8.1 #3): within-category price co-movement
    def mean_abs_corr(df):
        m = df.pivot_table(index="DAY", columns="PRODUCT_ID", values="price")
        if m.shape[1] < 2:
            return np.nan
        c = m.corr().values
        iu = np.triu_indices_from(c, k=1)
        v = np.abs(c[iu])
        v = v[~np.isnan(v)]
        return float(v.mean()) if v.size else np.nan

    ps = price_sess.merge(top[["PRODUCT_ID", "COMMODITY_DESC"]], on="PRODUCT_ID")
    corr = ps.groupby("COMMODITY_DESC")[["DAY", "PRODUCT_ID", "price"]].apply(
        mean_abs_corr, include_groups=False).rename("mean_abs_price_corr")
    audit = audit.join(corr)
    co = audit.index[audit.mean_abs_price_corr > cfg.max_price_corr]
    mark(co, f"within-category price correlation > {cfg.max_price_corr}")

    # ---- filter 4 (app. 8.1 #4): enough Sunday->Monday price variation
    sun = ps[ps.weekday == 0][["PRODUCT_ID", "COMMODITY_DESC", "pair_week", "price"]]
    mon = ps[ps.weekday == 1][["PRODUCT_ID", "pair_week", "price"]].rename(columns={"price": "price_mon"})
    chg = sun.merge(mon, on=["PRODUCT_ID", "pair_week"], how="inner")
    chg["dp"] = chg.price_mon - chg.price
    it = chg.groupby(["COMMODITY_DESC", "PRODUCT_ID"]).agg(
        any_change=("dp", lambda s: (s.abs() > 0.011).any()),
        share_big=("dp", lambda s: (s.abs() >= 0.10).mean()),
        n_big=("dp", lambda s: int((s.abs() >= 0.10).sum())),
    ).reset_index()
    pv = it.groupby("COMMODITY_DESC").agg(
        n_items_with_change=("any_change", "sum"),
        max_share_big=("share_big", "max"),
        n_big_changes=("n_big", "sum"),
    )
    audit = audit.join(pv)
    nopv = audit.index[(audit.n_items_with_change.fillna(0) < 2) |
                       (audit.max_share_big.fillna(0) < cfg.min_share_big)]
    mark(nopv, f"insufficient price variation (<2 items changing, or no item with "
               f">={cfg.min_share_big:.0%} of pair-weeks moving >=$0.10)")

    # ---- filter 5 (app. 8.1 #5): seasonality
    daily = smp[smp.PRODUCT_ID.isin(top_ids)].groupby(["PRODUCT_ID", "DAY"]).size().reset_index(name="q")
    hh_idx = daily.groupby("PRODUCT_ID").q.apply(herfindahl).rename("hhi")
    hh_idx = hh_idx.to_frame()
    hh_idx["pct"] = hh_idx.hhi.rank(pct=True)
    cat_seas = top.merge(hh_idx, left_on="PRODUCT_ID", right_index=True, how="left") \
                  .groupby("COMMODITY_DESC").pct.mean().rename("seasonality_pct")
    audit = audit.join(cat_seas)
    alive = audit.index[audit.drop_reason == ""]
    if len(alive):
        cut = audit.loc[alive, "seasonality_pct"].quantile(1 - cfg.seasonal_drop)
        seas = [c for c in alive if audit.loc[c, "seasonality_pct"] > cut]
        mark(seas, f"top {cfg.seasonal_drop:.0%} most seasonal (Herfindahl of daily demand)")

    # ---- optional: drop categories that fail the price-endogeneity placebo tests
    # (11_placebo_tests.py -> 13_placebo_followup.py).  Off by default because the
    # placebo results are themselves produced from a sample built by this script;
    # turn it on for a second pass.
    if cfg.exclude_placebo_failures:
        path = os.path.join(HERE, "..", "out", "placebo_category_status.csv")
        if os.path.exists(path):
            st = pd.read_csv(path)
            col = ("fails_any" if cfg.placebo_rule == "any" else "fails_random_only_tests")
            bad = set(st.loc[st[col], "COMMODITY_DESC"])
            mark([c for c in audit.index if c in bad],
                 f"fails the price placebo test ({cfg.placebo_rule})")
            log(f"dropped {len(bad & set(audit.index))} categories failing the "
                f"'{cfg.placebo_rule}' placebo rule")
        else:
            log(f"[WARN] --exclude-placebo-failures set but {path} is missing; skipping")

    kept = sorted(audit.index[audit.drop_reason == ""])
    log(f"categories retained: {len(kept)}")
    log("\n" + audit.drop_reason.replace("", "KEPT").value_counts().to_string())

    # -------------------------------------------------------------- build sets
    items = top[top.COMMODITY_DESC.isin(kept)].merge(
        prod[["PRODUCT_ID", "SUB_COMMODITY_DESC", "BRAND", "MANUFACTURER",
              "DEPARTMENT", "CURR_SIZE_OF_PRODUCT"]], on="PRODUCT_ID", how="left")
    log(f"items retained: {len(items):,} across {items.COMMODITY_DESC.nunique()} categories")

    # choices: one row per (trip, category) in which something was bought
    ch = smp[smp.PRODUCT_ID.isin(set(items.PRODUCT_ID))].copy()
    ch = ch.groupby(["household_key", "DAY", "COMMODITY_DESC", "PRODUCT_ID"], as_index=False).agg(
        units=("QUANTITY", "sum"), spend=("SALES_VALUE", "sum"),
        coupon_used=("coupon_used", "max"), store_id=("STORE_ID", "first"))
    # paper: when several items from a category are bought on one trip, keep one at random
    rng = np.random.default_rng(cfg.seed)
    ch["_r"] = rng.random(len(ch))
    ch = ch.sort_values("_r").groupby(["household_key", "DAY", "COMMODITY_DESC"], as_index=False).first()
    ch = ch.drop(columns="_r")
    log(f"category-purchase observations (product-choice stage): {len(ch):,}")

    sess = day_map[["DAY", "WEEK_NO", "weekday", "pair_week"]].copy()
    sess = sess.sort_values("DAY").reset_index(drop=True)
    sess["session_id"] = np.arange(len(sess))

    smp_trips = smp.groupby(["household_key", "DAY"], as_index=False).agg(
        n_lines=("PRODUCT_ID", "size"), spend=("SALES_VALUE", "sum"),
        store_id=("STORE_ID", "first"), hour=("hour", "median"))
    smp_trips = smp_trips.merge(sess, on="DAY", how="left")

    price_out = price_sess[price_sess.PRODUCT_ID.isin(set(items.PRODUCT_ID))] \
        .merge(sess[["DAY", "session_id"]], on="DAY", how="left")

    os.makedirs(DATA, exist_ok=True)
    smp_trips.to_parquet(os.path.join(DATA, "sample_trips.parquet"), index=False)
    ch.to_parquet(os.path.join(DATA, "sample_choices.parquet"), index=False)
    items.to_parquet(os.path.join(DATA, "items.parquet"), index=False)
    audit.reset_index().to_csv(os.path.join(DATA, "filter_audit.csv"), index=False)
    price_out.to_parquet(os.path.join(DATA, "price_panel.parquet"), index=False)
    sess.to_parquet(os.path.join(DATA, "sessions.parquet"), index=False)
    audit.loc[kept].reset_index().to_parquet(os.path.join(DATA, "categories.parquet"), index=False)
    log("wrote sample_trips / sample_choices / items / categories / price_panel / sessions / filter_audit")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--top-j", type=int, default=10)
    p.add_argument("--min-trips", type=int, default=20)
    p.add_argument("--max-trips", type=int, default=300)
    p.add_argument("--min-items", type=int, default=5)
    p.add_argument("--min-cat-trips", type=int, default=200)
    p.add_argument("--max-multi-any", type=float, default=0.15)
    p.add_argument("--max-multi-top", type=float, default=0.10)
    p.add_argument("--max-price-corr", type=float, default=0.75)
    p.add_argument("--min-share-big", type=float, default=0.10)
    p.add_argument("--seasonal-drop", type=float, default=0.15)
    p.add_argument("--min-modal-share", type=float, default=0.60,
                   help="random-weight screen: minimum share of transactions at the modal price")
    p.add_argument("--holiday-dev", type=float, default=0.12)
    p.add_argument("--min-day-baskets", type=int, default=50,
                   help="drop a pair-week if either of its days has fewer than this "
                        "many baskets chain-wide (panel holes, not quiet days)")
    p.add_argument("--exclude-placebo-failures", action="store_true",
                   help="drop categories that fail the price-endogeneity placebo tests "
                        "(requires out/placebo_category_status.csv)")
    p.add_argument("--placebo-rule", choices=["any", "random"], default="random",
                   help="'any' drops a category failing any placebo; 'random' drops only "
                        "those failing the fully decorrelated placebo")
    p.add_argument("--seed", type=int, default=20230808)
    main(p.parse_args())
