"""
Stage 3 -- Emit the Nested Factorization product-choice inputs.

File formats are taken from src/bemb_loc/emb_io.hpp (they are read with fscanf,
so they must be exactly tab separated, integer ids, no ragged lines):

  train.tsv / validation.tsv / test.tsv   user_id \t item_id \t session_id \t units
  item_sess_price.tsv                     item_id \t session_id \t price
                                          (must be complete: Nitems x Nsessions)
  itemGroup.tsv                           item_id \t group_id        (= category)
  userGroup.tsv                           user_id \t group_id        (hpf -days)
  sess_days.tsv                           session_id \t day_id \t weekday_id \t hour
                                          ("day_id" is the paper's week trend index;
                                           here it is the pair-week)
  obsUser.tsv                             user_id \t w_1 ... w_UC
  obsItem.tsv                             item_id \t x_1 ... x_IC

Held-out design (paper sec. 6): the split is at the household x week level, so a
household is observed in training but its behaviour in a held-out week -- including
the Sunday/Monday days that straddle a price change -- has to be predicted.

Also written:
  events.csv        per (item, pair-week) flags for the three counterfactual events
  id_maps/*.csv     original dunnhumby ids <-> model ids
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


def log(m):
    print(f"[03] {m}", flush=True)


def build_user_obs(users, cfg):
    """Household demographics -> numeric design matrix.

    Only 801 of the 2500 dunnhumby households carry demographics, so a
    'demographics observed' indicator is included and missing values are coded 0.
    That keeps the paper's W_i^T rho_j term well defined for every household
    (the paper's data had demographics for all households).
    """
    d = pd.read_csv(RAW + "hh_demographic.csv")
    # this release ships generic column names; map them to their meanings
    ren = {"classification_1": "age", "classification_2": "marital",
           "classification_3": "income", "classification_4": "hh_size",
           "classification_5": "hh_comp", "HOMEOWNER_DESC": "homeowner",
           "KID_CATEGORY_DESC": "kids"}
    d = d.rename(columns=ren)

    def ordinal(s, pat):
        return pd.to_numeric(s.astype(str).str.extract(pat, expand=False), errors="coerce")

    out = pd.DataFrame({"household_key": d.household_key})
    out["age"] = ordinal(d.age, r"(\d+)")                    # Age Group1..6
    out["income"] = ordinal(d.income, r"(\d+)")              # Level1..12
    out["hh_size"] = ordinal(d.hh_size.astype(str).str.replace("+", "", regex=False), r"(\d+)")
    out["hh_comp"] = ordinal(d.hh_comp, r"(\d+)")            # Group1..5
    out["kids"] = ordinal(d.kids, r"(\d+)").fillna(0)        # 1,2,3, None/Unknown
    out["married"] = (d.marital.astype(str).str.upper() == "X").astype(float)
    out["single"] = (d.marital.astype(str).str.upper() == "Y").astype(float)
    out["homeowner"] = (d.homeowner.astype(str).str.upper() == "HOMEOWNER").astype(float)

    W = pd.DataFrame({"household_key": users})
    W = W.merge(out, on="household_key", how="left")
    W["has_demo"] = W.age.notna().astype(float)
    for c in ["age", "income", "hh_size", "hh_comp", "kids", "married", "single", "homeowner"]:
        v = W[c]
        mu, sd = v.mean(), v.std()
        W[c] = ((v - mu) / (sd if sd and sd > 0 else 1.0)).fillna(0.0)
    return W


def parse_size(s):
    """CURR_SIZE_OF_PRODUCT -> a rough numeric size in a common unit, else NaN."""
    if not isinstance(s, str):
        return np.nan
    s = s.strip().upper()
    num = pd.to_numeric(pd.Series([s]).str.extract(r"([\d.]+)", expand=False), errors="coerce").iat[0]
    if np.isnan(num):
        return np.nan
    if "LB" in s:
        return num * 16.0
    if "OZ" in s or "FZ" in s:
        return num
    if "GA" in s:
        return num * 128.0
    if "QT" in s:
        return num * 32.0
    if "PT" in s:
        return num * 16.0
    if "LT" in s or "LTR" in s:
        return num * 33.8
    if "ML" in s:
        return num * 0.0338
    return num


def build_item_obs(items):
    X = pd.DataFrame({"PRODUCT_ID": items.PRODUCT_ID})
    X["private_label"] = (items.BRAND.astype(str).str.upper() == "PRIVATE").astype(float).values
    size = items.CURR_SIZE_OF_PRODUCT.map(parse_size)
    # size is only comparable within a category, so standardise within category
    size = size.groupby(items.COMMODITY_DESC.values).transform(
        lambda v: (v - v.mean()) / (v.std() if v.std() and v.std() > 0 else 1.0))
    X["size_z"] = size.fillna(0.0).values
    X["size_missing"] = items.CURR_SIZE_OF_PRODUCT.map(parse_size).isna().astype(float).values
    # manufacturer scale: how many of the retained items the manufacturer supplies
    mfr = items.groupby("MANUFACTURER").PRODUCT_ID.transform("size")
    X["mfr_breadth"] = ((mfr - mfr.mean()) / (mfr.std() if mfr.std() > 0 else 1.0)).values
    return X


def main(cfg):
    outdir = os.path.join(HERE, "..", cfg.outdir)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(os.path.join(outdir, "id_maps"), exist_ok=True)

    items = pd.read_parquet(os.path.join(DATA, "items.parquet"))
    choices = pd.read_parquet(os.path.join(DATA, "sample_choices.parquet"))
    trips = pd.read_parquet(os.path.join(DATA, "sample_trips.parquet"))
    price = pd.read_parquet(os.path.join(DATA, "price_panel.parquet"))
    sess = pd.read_parquet(os.path.join(DATA, "sessions.parquet"))

    choices = choices[choices.PRODUCT_ID.isin(set(items.PRODUCT_ID))].copy()
    choices = choices.merge(sess[["DAY", "session_id", "pair_week", "weekday"]], on="DAY", how="left")

    # ------------------------------------------------------------------- ids
    users = np.sort(choices.household_key.unique())
    uid = pd.Series(np.arange(len(users)), index=users, name="user_id")
    items = items.sort_values(["COMMODITY_DESC", "n_trips"], ascending=[True, False]).reset_index(drop=True)
    items["item_id"] = np.arange(len(items))
    iid = items.set_index("PRODUCT_ID").item_id
    cats = pd.Series(np.arange(items.COMMODITY_DESC.nunique()),
                     index=np.sort(items.COMMODITY_DESC.unique()), name="group_id")
    items["group_id"] = items.COMMODITY_DESC.map(cats).values

    choices["user_id"] = choices.household_key.map(uid).values
    choices["item_id"] = choices.PRODUCT_ID.map(iid).values
    log(f"users {len(users)}  items {len(items)}  categories {len(cats)}  "
        f"sessions {len(sess)}  observations {len(choices):,}")

    # ------------------------------------------------------ train/val/test split
    # Split household x pair-week cells.  Every item and every user must appear in
    # train (emb_io warns and mis-indexes otherwise), so repair afterwards.
    rng = np.random.default_rng(cfg.seed)
    cells = choices[["user_id", "pair_week"]].drop_duplicates().reset_index(drop=True)
    r = rng.random(len(cells))
    cells["split"] = np.where(r < cfg.p_train, "train",
                       np.where(r < cfg.p_train + cfg.p_val, "validation", "test"))
    choices = choices.merge(cells, on=["user_id", "pair_week"], how="left")

    missing_items = set(items.item_id) - set(choices.loc[choices.split == "train", "item_id"])
    missing_users = set(range(len(users))) - set(choices.loc[choices.split == "train", "user_id"])
    if missing_items or missing_users:
        need = choices[choices.item_id.isin(missing_items) | choices.user_id.isin(missing_users)]
        move = need.groupby(["item_id"], as_index=False).head(1).index.union(
               need.groupby(["user_id"], as_index=False).head(1).index)
        choices.loc[move, "split"] = "train"
        log(f"moved {len(move)} observations into train so every user/item is represented")
    log(choices.split.value_counts().to_string())

    # ------------------------------------------------------------------ writers
    def w(name, df, fmt):
        p = os.path.join(outdir, name)
        with open(p, "w") as f:
            for row in df.itertuples(index=False):
                f.write(fmt(row))
        log(f"wrote {name}  ({len(df):,} lines)")

    for split, fname in [("train", "train.tsv"), ("validation", "validation.tsv"), ("test", "test.tsv")]:
        d = choices.loc[choices.split == split, ["user_id", "item_id", "session_id", "units"]]
        d = d.astype({"user_id": int, "item_id": int, "session_id": int, "units": int})
        d["units"] = 1  # unit demand: the model wants a 0/1 choice indicator
        w(fname, d.sort_values(["session_id", "user_id", "item_id"]),
          lambda r: f"{r.user_id}\t{r.item_id}\t{r.session_id}\t{r.units}\n")

    # ---- complete item x session price grid
    pr = price[["PRODUCT_ID", "session_id", "price"]].copy()
    pr["item_id"] = pr.PRODUCT_ID.map(iid)
    pr = pr.dropna(subset=["item_id"])
    grid = pd.MultiIndex.from_product([items.item_id.values, sess.session_id.values],
                                      names=["item_id", "session_id"]).to_frame(index=False)
    pr = grid.merge(pr[["item_id", "session_id", "price"]], on=["item_id", "session_id"], how="left")
    n_missing = pr.price.isna().sum()
    if n_missing:
        med = pr.groupby("item_id").price.transform("median")
        pr["price"] = pr.price.fillna(med)
        log(f"filled {n_missing:,} missing item-session prices with the item median")
    assert pr.price.notna().all(), "price grid still incomplete"
    w("item_sess_price.tsv", pr.astype({"item_id": int, "session_id": int}),
      lambda r: f"{r.item_id}\t{r.session_id}\t{r.price:.4f}\n")

    # ---- groups, sessions, observables
    w("itemGroup.tsv", items[["item_id", "group_id"]].astype(int),
      lambda r: f"{r.item_id}\t{r.group_id}\n")

    hourly = trips.groupby("DAY").hour.median()
    sd = sess.copy()
    sd["day_id"] = pd.factorize(sd.pair_week, sort=True)[0]
    sd["weekday_id"] = sd.weekday.astype(int)          # 0 = Sunday, 1 = Monday
    sd["hour"] = sd.DAY.map(hourly).fillna(13.0)
    w("sess_days.tsv", sd[["session_id", "day_id", "weekday_id", "hour"]],
      lambda r: f"{int(r.session_id)}\t{int(r.day_id)}\t{int(r.weekday_id)}\t{r.hour:.3f}\n")

    W = build_user_obs(users, cfg)
    wcols = [c for c in W.columns if c != "household_key"]
    W.insert(0, "user_id", np.arange(len(W)))
    w("obsUser.tsv", W[["user_id"] + wcols],
      lambda r: f"{int(r.user_id)}\t" + "\t".join(f"{getattr(r, c):.6f}" for c in wcols) + "\n")
    log(f"obsUser columns (UC={len(wcols)}): {wcols}")

    X = build_item_obs(items)
    xcols = [c for c in X.columns if c != "PRODUCT_ID"]
    X.insert(0, "item_id", items.item_id.values)
    w("obsItem.tsv", X[["item_id"] + xcols],
      lambda r: f"{int(r.item_id)}\t" + "\t".join(f"{getattr(r, c):.6f}" for c in xcols) + "\n")
    log(f"obsItem columns (IC={len(xcols)}): {xcols}")

    # userGroup: needed only by the hpf binary's -days option; use spend quintiles
    spend = trips.groupby("household_key").spend.sum().reindex(users).fillna(0)
    ug = pd.DataFrame({"user_id": np.arange(len(users)),
                       "group_id": pd.qcut(spend.rank(method="first"), 5, labels=False).values})
    w("userGroup.tsv", ug.astype(int), lambda r: f"{r.user_id}\t{r.group_id}\n")

    # ---------------------------------------------------- counterfactual events
    # (a) own-price change, (b) another item in the category changes price,
    # (c) an item enters/leaves the choice set.  dunnhumby has no stock-out feed,
    # so (c) is proxied by an item having no recorded chain-wide sale in the week.
    ps = pr.merge(sess[["session_id", "weekday", "pair_week"]], on="session_id")
    ps = ps.merge(items[["item_id", "group_id"]], on="item_id")
    sun = ps[ps.weekday == 0][["item_id", "group_id", "pair_week", "price"]]
    mon = ps[ps.weekday == 1][["item_id", "pair_week", "price"]].rename(columns={"price": "price_mon"})
    ev = sun.merge(mon, on=["item_id", "pair_week"], how="inner")
    ev["dp"] = (ev.price_mon - ev.price).round(4)
    ev["own_price_change"] = (ev.dp.abs() >= 0.10).astype(int)
    grp = ev.groupby(["group_id", "pair_week"]).own_price_change.transform("sum")
    ev["cross_price_change"] = ((grp - ev.own_price_change) > 0).astype(int)
    ev.to_csv(os.path.join(outdir, "events.csv"), index=False)
    log(f"counterfactual events: own-price {ev.own_price_change.sum():,} item-pairweeks, "
        f"cross-price {ev.cross_price_change.sum():,}")

    # ---------------------------------------------------------------- id maps
    mp = os.path.join(outdir, "id_maps")
    pd.DataFrame({"user_id": np.arange(len(users)), "household_key": users}).to_csv(
        os.path.join(mp, "users.csv"), index=False)
    items[["item_id", "PRODUCT_ID", "COMMODITY_DESC", "SUB_COMMODITY_DESC", "BRAND",
           "MANUFACTURER", "group_id", "n_trips"]].to_csv(os.path.join(mp, "items.csv"), index=False)
    cats.rename("group_id").reset_index().rename(columns={"index": "COMMODITY_DESC"}).to_csv(
        os.path.join(mp, "categories.csv"), index=False)
    sd[["session_id", "DAY", "WEEK_NO", "pair_week", "weekday", "day_id"]].to_csv(
        os.path.join(mp, "sessions.csv"), index=False)
    choices[["user_id", "item_id", "session_id", "pair_week", "weekday", "split"]].to_csv(
        os.path.join(mp, "observations.csv"), index=False)
    log(f"done -> {outdir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="model_input")
    p.add_argument("--p-train", type=float, default=0.70)
    p.add_argument("--p-val", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=7)
    main(p.parse_args())
