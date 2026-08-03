"""
Stage 22 -- Build the dataset for a basket model with household state.

This replaces 02/03 rather than extending them.  Those scripts exist to serve the
paper's model, and every one of their major filters is there to protect an
assumption this model drops:

  paper's pipeline                        here
  --------------------------------------  ------------------------------------------
  at most one item per category per trip  a basket is a set; buy as many as you like
  -> 132 of 307 categories dropped        -> no unit-demand filter at all
  categories independent                  items interact through a shared embedding
  Sunday + Monday only (172 sessions)     all 711 days, because state needs time
  -> 560 items, 56 categories             -> ~5,500 items, ~750 sub-commodities
  price = the only time-varying input     price + household state per sub-commodity

What it writes to basket_input/:

  baskets.parquet   one row per (basket, item) purchase, with split label
  items.parquet     item_id <-> PRODUCT_ID and the held-out labels (sub-commodity,
                    brand, manufacturer, department) that the model never sees
  prices.npy        [J, D] log price by item and day, carried forward
  state.npz         the structure that answers "how long since this household last
                    bought this sub-commodity, as of this day"
  meta.json         sizes, split boundaries, and the frequency cut

The state lookup deserves a note.  The obvious implementations are both wrong at
this scale: materialising (household x day x sub-commodity) is ~32 million rows, and
a per-sample Python lookup is ~35 million dict hits per epoch.  Instead the purchase
days are stored once as a globally sorted key array, key = group_id * 1024 + day,
where group_id indexes a (household, sub-commodity) pair.  A single vectorised
np.searchsorted then answers the question for a whole batch, and because the offset
is larger than any day the ordering never crosses a group boundary.

Splitting is by household x week, as the paper does, so a household contributes to
several splits and the model is never asked to predict a household it has not seen.
The last weeks are held out rather than a random slice: a model meant to answer
"what happens next" should be tested on what happened next.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "basket_input")
RAW = os.path.join(os.environ.get(
    "NF_RAW_DIR",
    os.path.join(HERE, "..", "..", "dunnhumby_The-Complete-Journey",
                 "dunnhumby_The-Complete-Journey CSV")), "")

DAY_STRIDE = 1024          # > 711, so group_id * DAY_STRIDE + day never collides


def log(m):
    print(f"[22] {m}", flush=True)


def main(a):
    os.makedirs(OUT, exist_ok=True)
    tx = pd.read_parquet(os.path.join(DATA, "tx.parquet"),
                         columns=["household_key", "DAY", "WEEK_NO", "PRODUCT_ID",
                                  "BASKET_ID", "QUANTITY", "STORE_ID",
                                  "unit_price", "base_price"])
    prod = pd.read_csv(RAW + "product.csv",
                       usecols=["PRODUCT_ID", "COMMODITY_DESC", "SUB_COMMODITY_DESC",
                                "BRAND", "MANUFACTURER", "DEPARTMENT"])
    tx = tx.merge(prod, on="PRODUCT_ID", how="left").dropna(subset=["COMMODITY_DESC"])
    log(f"raw: {len(tx):,} lines, {tx.PRODUCT_ID.nunique():,} products, "
        f"{tx.BASKET_ID.nunique():,} baskets")

    # ------------------------------------------------------------- item universe
    # The only filter that survives: an item needs enough purchases for its own
    # embedding to mean anything.  Nothing is dropped for violating unit demand,
    # for co-moving prices, or for seasonality.
    freq = tx.groupby("PRODUCT_ID").size()
    keep = freq[freq >= a.min_lines].index
    tx = tx[tx.PRODUCT_ID.isin(set(keep))].copy()
    log(f"kept {len(keep):,} items with >={a.min_lines} lines -> {len(tx):,} lines "
        f"({tx.BASKET_ID.nunique():,} baskets, {tx.SUB_COMMODITY_DESC.nunique():,} "
        f"sub-commodities, {tx.COMMODITY_DESC.nunique()} commodities)")

    # Households need enough trips for a taste vector to be estimable.  This is the
    # paper's own 20-300 trip rule, kept because it is about statistical support
    # rather than about the model's structure.
    trips_per_hh = tx.groupby("household_key").DAY.nunique()
    hh_keep = trips_per_hh[(trips_per_hh >= a.min_trips) & (trips_per_hh <= a.max_trips)].index
    tx = tx[tx.household_key.isin(set(hh_keep))].copy()
    log(f"kept {len(hh_keep):,} households with {a.min_trips}-{a.max_trips} trips "
        f"-> {len(tx):,} lines")

    # ------------------------------------------------------------------- ids
    items = (tx.groupby("PRODUCT_ID")
             .agg(n_lines=("QUANTITY", "size"),
                  n_households=("household_key", "nunique"))
             .reset_index()
             .merge(prod, on="PRODUCT_ID", how="left")
             .sort_values("PRODUCT_ID").reset_index(drop=True))
    items["item_id"] = np.arange(len(items))
    iid = items.set_index("PRODUCT_ID").item_id
    for col, name in [("SUB_COMMODITY_DESC", "sub_id"), ("COMMODITY_DESC", "cat_id"),
                      ("BRAND", "brand_id"), ("MANUFACTURER", "mfr_id"),
                      ("DEPARTMENT", "dept_id")]:
        items[name] = pd.factorize(items[col])[0]
    users = np.sort(tx.household_key.unique())
    uid = pd.Series(np.arange(len(users)), index=users)

    tx["item_id"] = tx.PRODUCT_ID.map(iid).astype(np.int32)
    tx["user_id"] = tx.household_key.map(uid).astype(np.int32)
    tx["sub_id"] = tx.item_id.map(items.set_index("item_id").sub_id).astype(np.int32)

    J, N, D = len(items), len(users), int(tx.DAY.max()) + 1
    S = int(items.sub_id.max()) + 1
    log(f"ids: {N:,} households, {J:,} items, {S:,} sub-commodities, {D} days")

    # ------------------------------------------------------------ basket table
    # One row per (basket, item).  QUANTITY is kept but the model treats purchase as
    # binary: dunnhumby's quantity field is unreliable for weighed goods and the
    # paper makes the same choice.
    bk = (tx.groupby(["BASKET_ID", "user_id", "DAY", "WEEK_NO", "item_id", "sub_id"],
                     as_index=False)
          .agg(units=("QUANTITY", "sum"), price=("unit_price", "median")))
    bk = bk.sort_values(["user_id", "DAY", "BASKET_ID", "item_id"]).reset_index(drop=True)
    log(f"basket table: {len(bk):,} (basket, item) rows over {bk.BASKET_ID.nunique():,} baskets")

    # ----------------------------------------------------------------- splits
    # Hold out the final weeks, not a random slice.  A model whose point is to answer
    # "what happens next week if we change price" should be scored on later weeks,
    # and a random split would let it see the future of every household.
    wmax = int(bk.WEEK_NO.max())
    test_from = wmax - a.test_weeks + 1
    val_from = test_from - a.val_weeks
    bk["split"] = np.where(bk.WEEK_NO >= test_from, "test",
                           np.where(bk.WEEK_NO >= val_from, "validation", "train"))
    log(f"splits by week: train <{val_from}, validation {val_from}-{test_from - 1}, "
        f"test >={test_from} (of {wmax})")
    log("  " + "  ".join(f"{k} {v:,}" for k, v in bk.split.value_counts().items()))

    # An item or household seen only after the split boundary has no fitted vector.
    # Drop those rows from the held-out sets rather than scoring a cold start the
    # model was never given a chance at -- and say how many.
    tr = bk[bk.split == "train"]
    known_i, known_u = set(tr.item_id), set(tr.user_id)
    bad = (~bk.item_id.isin(known_i) | ~bk.user_id.isin(known_u)) & (bk.split != "train")
    log(f"  dropping {int(bad.sum()):,} held-out rows whose item or household never "
        f"appears in training ({bad.sum() / max((bk.split != 'train').sum(), 1):.2%} of held out)")
    bk = bk[~bad].reset_index(drop=True)

    # ------------------------------------------------------------- price panel
    # Daily log price per item, carried forward from the last observed day and back
    # for the start of the panel.  Held at the chain level, as elsewhere in the repo.
    pd_day = (tx.groupby(["item_id", "DAY"]).unit_price.median().rename("p").reset_index())
    grid = pd.MultiIndex.from_product([np.arange(J), np.arange(D)],
                                      names=["item_id", "DAY"]).to_frame(index=False)
    pg = grid.merge(pd_day, on=["item_id", "DAY"], how="left").sort_values(["item_id", "DAY"])
    pg["p"] = pg.groupby("item_id").p.ffill()
    pg["p"] = pg.groupby("item_id").p.bfill()
    obs_share = float(pd_day.shape[0] / (J * D))
    price = pg.p.to_numpy(dtype=np.float32).reshape(J, D)
    # An item with no price anywhere cannot happen (it has >=100 purchases), but be
    # explicit rather than propagating a NaN into a log.
    assert np.isfinite(price).all(), "price grid has holes"
    log_price = np.log(np.clip(price, 1e-3, None)).astype(np.float32)
    # Centre per item: the level is absorbed by the item's own intercept, and what
    # identifies price response is deviation from that item's normal price.
    log_price_dev = (log_price - log_price.mean(axis=1, keepdims=True)).astype(np.float32)
    log(f"price panel: {J} x {D}, {obs_share:.1%} of item-days directly observed, "
        f"sd of within-item log price {log_price_dev.std():.4f}")

    # --------------------------------------------------------------- state
    # For every (household, sub-commodity) pair, the sorted days on which that
    # household bought that sub-commodity.  Encoded as one globally sorted key array
    # so a batch query is a single searchsorted (see the module docstring).
    ev = tx[["user_id", "sub_id", "DAY"]].drop_duplicates()
    ev = ev.sort_values(["user_id", "sub_id", "DAY"]).reset_index(drop=True)
    ev["group"] = ev.user_id.astype(np.int64) * S + ev.sub_id
    keys = (ev.group.to_numpy() * DAY_STRIDE + ev.DAY.to_numpy()).astype(np.int64)
    assert (np.diff(keys) > 0).all(), "state keys must be strictly increasing"
    # group -> first index, so a query can tell "no earlier purchase in this group"
    # from "the previous key belongs to a different group".
    gstart = {}
    uniq, first = np.unique(ev.group.to_numpy(), return_index=True)
    gstart_keys, gstart_vals = uniq.astype(np.int64), first.astype(np.int64)
    log(f"state: {len(ev):,} (household, sub-commodity, day) purchase events across "
        f"{len(uniq):,} (household, sub-commodity) pairs")

    # Median repurchase gap per sub-commodity: the natural time scale for that
    # sub-commodity, used to normalise recency so "a long time" means the right
    # number of days for milk and for shampoo.
    g = ev.copy()
    g["prev"] = g.groupby(["user_id", "sub_id"]).DAY.shift()
    gaps = (g.DAY - g.prev).dropna()
    gap_by_sub = (g.assign(gap=g.DAY - g.prev).dropna(subset=["gap"])
                  .groupby("sub_id").gap.median())
    sub_gap = np.full(S, float(gaps.median()), dtype=np.float32)
    sub_gap[gap_by_sub.index.to_numpy()] = gap_by_sub.to_numpy()
    sub_gap = np.clip(sub_gap, 3.0, 180.0)
    log(f"  repurchase gap: overall median {gaps.median():.0f} days; per sub-commodity "
        f"median ranges {sub_gap.min():.0f}-{sub_gap.max():.0f}")

    # ------------------------------------------------------------------ write
    bk.to_parquet(os.path.join(OUT, "baskets.parquet"), index=False)
    items.to_parquet(os.path.join(OUT, "items.parquet"), index=False)
    np.save(os.path.join(OUT, "log_price_dev.npy"), log_price_dev)
    np.save(os.path.join(OUT, "log_price.npy"), log_price)
    np.savez(os.path.join(OUT, "state.npz"),
             keys=keys, gstart_keys=gstart_keys, gstart_vals=gstart_vals,
             sub_gap=sub_gap, item_sub=items.sub_id.to_numpy().astype(np.int32))
    meta = {
        "n_users": int(N), "n_items": int(J), "n_subs": int(S), "n_days": int(D),
        "n_baskets": int(bk.BASKET_ID.nunique()),
        "n_rows": int(len(bk)),
        "min_lines": a.min_lines, "min_trips": a.min_trips, "max_trips": a.max_trips,
        "val_from_week": int(val_from), "test_from_week": int(test_from),
        "day_stride": DAY_STRIDE,
        "split_rows": {k: int(v) for k, v in bk.split.value_counts().items()},
        "n_commodities": int(items.cat_id.nunique()),
        "price_obs_share": obs_share,
        "median_repurchase_gap_days": float(gaps.median()),
    }
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log(f"wrote basket_input/ -> {len(bk):,} rows, {J:,} items, {N:,} households, "
        f"{S:,} sub-commodities")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--min-lines", type=int, default=100,
                   help="minimum purchase lines before an item gets its own embedding")
    p.add_argument("--min-trips", type=int, default=20)
    p.add_argument("--max-trips", type=int, default=300)
    p.add_argument("--val-weeks", type=int, default=8)
    p.add_argument("--test-weeks", type=int, default=12)
    main(p.parse_args())
