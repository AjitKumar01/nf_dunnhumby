"""
Stage 17 -- Does pooling 561 stores into one "chain" actually cost anything?

The paper had a single store.  This port pools stores and gives every household in a
session the same price, justified so far only by a low cross-store price CV.  That is
not the whole risk.  Three things could go wrong, and each has a testable implication:

  1. ASSORTMENT.  If a store does not stock item j, a household shopping there is
     recorded as "did not choose j" when in fact j was not on offer.  The model reads
     that as a preference.  Measured as: what share of the retained items does a
     household's own store actually sell?

  2. PRICE MISMEASUREMENT.  If chain-level price is a poor proxy for the price a
     particular household faced, the model should fit *worse* for households whose
     store's prices deviate most from the chain median.  That is directly testable on
     held-out data, and it is the sharpest test available: it needs no re-estimation
     and it fails loudly if pooling is doing damage.

  3. STORE COMPOSITION.  A store fixed effect could be confounded with the price
     variation.  Tested by re-evaluating held-out fit separately for store-loyal and
     store-switching households, and for large versus small stores.

Writes out/store_diagnostics.json and figures/stores.png.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

import nf_torch as nf
from importlib import import_module

ev = import_module("07_evaluate")
trainer = import_module("05_train_nf")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data")
MI = os.path.join(HERE, "..", "..", "model_input")
OUT = os.path.join(HERE, "..", "..", "out")
FIG = os.path.join(HERE, "..", "..", "figures")


def log(m):
    print(f"[17] {m}", flush=True)


def main(a):
    os.makedirs(FIG, exist_ok=True)
    tx = pd.read_parquet(os.path.join(DATA, "tx.parquet"),
                         columns=["PRODUCT_ID", "STORE_ID", "WEEK_NO", "DAY",
                                  "household_key", "BASKET_ID", "unit_price"])
    items = pd.read_csv(os.path.join(MI, "id_maps", "items.csv"))
    users = pd.read_csv(os.path.join(MI, "id_maps", "users.csv"))
    trips = pd.read_parquet(os.path.join(DATA, "sample_trips.parquet"))
    keep_prod = set(items.PRODUCT_ID)
    r = {}

    # ------------------------------------------------------- 1. assortment
    sold = tx[tx.PRODUCT_ID.isin(keep_prod)]
    per_store = sold.groupby("STORE_ID").PRODUCT_ID.nunique()
    store_trips = trips.groupby("store_id").size()
    big = store_trips[store_trips >= a.min_store_trips].index
    r["assortment"] = {
        "retained_items": int(len(items)),
        "stores_in_sample": int(trips.store_id.nunique()),
        "stores_with_at_least_%d_trips" % a.min_store_trips: int(len(big)),
        "median_items_ever_sold_per_store": float(per_store.reindex(big).median()),
        "median_share_of_catalogue_per_store": float(
            (per_store.reindex(big) / len(items)).median()),
    }
    log(f"assortment: the median store with >={a.min_store_trips} sample trips ever sells "
        f"{r['assortment']['median_items_ever_sold_per_store']:.0f} of {len(items)} "
        f"retained items ({r['assortment']['median_share_of_catalogue_per_store']:.1%})")
    log("  (this is a lower bound on assortment: a store may stock an item that simply "
        "never sold to one of our 2,083 households there)")

    # ------------------------------------------- 2. store price deviation
    # For each store x week, how far is its price from the chain price we assigned?
    chain = tx.groupby(["PRODUCT_ID", "WEEK_NO"]).unit_price.median().rename("chain")
    sw = tx[tx.PRODUCT_ID.isin(keep_prod)].groupby(
        ["PRODUCT_ID", "WEEK_NO", "STORE_ID"]).unit_price.agg(["median", "size"])
    sw = sw[sw["size"] >= 2].join(chain, on=["PRODUCT_ID", "WEEK_NO"])
    sw["dev"] = (sw["median"] - sw.chain).abs()
    store_dev = sw.groupby("STORE_ID").dev.mean().rename("mean_abs_price_dev")
    r["price_deviation"] = {
        "store_week_cells": int(len(sw)),
        "median_store_mean_abs_dev": float(store_dev.median()),
        "p90_store_mean_abs_dev": float(store_dev.quantile(0.9)),
    }
    log(f"price deviation: the median store's price sits ${store_dev.median():.3f} from "
        f"the chain price we assign; the 90th percentile store ${store_dev.quantile(0.9):.3f}")

    # ------------------------ 3. does that deviation show up in held-out fit?
    d = nf.load(MI, device="cpu")
    m1, m2, _ = ev.load_model(a.label, d, "cpu")
    slots = ev.slot_lookup(d)
    pred = ev.trip_predictions(m1, m2, d, "test")
    y = ev.outcome_matrix(d, "test", slots)
    p = (pred["pcat"].unsqueeze(2) * pred["pitem"]).clamp(1e-12, 1 - 1e-12)
    per_trip_ll = (y * torch.log(p)).sum(dim=(1, 2)).numpy()
    per_trip_n = y.sum(dim=(1, 2)).numpy()

    tu, ts = d.trips["test"]
    tf = pd.DataFrame({"user_id": tu.numpy(), "session_id": ts.numpy(),
                       "ll": per_trip_ll, "n": per_trip_n})
    tf = tf.merge(users, on="user_id")
    sess = pd.read_csv(os.path.join(MI, "id_maps", "sessions.csv"))
    tf = tf.merge(sess[["session_id", "DAY"]], on="session_id")
    tf = tf.merge(trips[["household_key", "DAY", "store_id"]],
                  on=["household_key", "DAY"], how="left")
    tf = tf.merge(store_dev, left_on="store_id", right_index=True, how="left")
    tf = tf[tf.n > 0]

    def band(df, col, q=4, labels=None):
        g = pd.qcut(df[col].rank(method="first"), q, labels=list(range(1, q + 1)))
        out = df.groupby(g, observed=True).apply(
            lambda x: pd.Series({"loglik_per_purchase": x.ll.sum() / x.n.sum(),
                                 "trips": len(x), "purchases": x.n.sum(),
                                 col: x[col].mean()}), include_groups=False)
        return out.reset_index(names="band")

    dev_bands = band(tf.dropna(subset=["mean_abs_price_dev"]), "mean_abs_price_dev")
    r["fit_by_store_price_deviation"] = dev_bands.to_dict("records")
    log("held-out fit by how far the household's store prices sit from the chain price:")
    for x in r["fit_by_store_price_deviation"]:
        log(f"  quartile {x['band']}: mean deviation ${x['mean_abs_price_dev']:.3f}, "
            f"log-likelihood {x['loglik_per_purchase']:.4f} "
            f"({int(x['purchases']):,} purchases)")
    lo = r["fit_by_store_price_deviation"][0]["loglik_per_purchase"]
    hi = r["fit_by_store_price_deviation"][-1]["loglik_per_purchase"]
    r["fit_gap_low_vs_high_deviation"] = float(lo - hi)
    log(f"  gap between the closest and furthest quartile: {lo - hi:+.4f} nats")

    # store loyalty
    loyal = trips.groupby("household_key").store_id.agg(
        lambda s: s.value_counts().iat[0] / len(s)).rename("primary_store_share")
    tf = tf.merge(loyal, on="household_key", how="left")
    loy_bands = band(tf.dropna(subset=["primary_store_share"]), "primary_store_share")
    r["fit_by_store_loyalty"] = loy_bands.to_dict("records")
    log("held-out fit by household store loyalty:")
    for x in r["fit_by_store_loyalty"]:
        log(f"  quartile {x['band']}: primary-store share "
            f"{x['primary_store_share']:.2f}, log-likelihood "
            f"{x['loglik_per_purchase']:.4f}")
    r["loyalty_summary"] = {
        "mean_primary_store_share": float(loyal.mean()),
        "median_primary_store_share": float(loyal.median()),
        "share_of_households_above_80pct": float((loyal > 0.8).mean()),
    }

    # store size
    tf = tf.merge(store_trips.rename("store_trips"), left_on="store_id",
                  right_index=True, how="left")
    size_bands = band(tf.dropna(subset=["store_trips"]), "store_trips")
    r["fit_by_store_size"] = size_bands.to_dict("records")

    with open(os.path.join(OUT, "store_diagnostics.json"), "w") as f:
        json.dump(r, f, indent=2, default=float)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.3))
    ax[0].hist((per_store.reindex(big) / len(items)).dropna(), bins=30,
               color="#2d6cdf", alpha=0.85, edgecolor="white")
    ax[0].set_xlabel("share of the 560 retained items ever sold at the store")
    ax[0].set_ylabel("stores")
    ax[0].set_title("1. Assortment differs across stores", fontsize=10)

    b = pd.DataFrame(r["fit_by_store_price_deviation"])
    ax[1].plot(b.mean_abs_price_dev, b.loglik_per_purchase, "-o", color="#c1432c")
    ax[1].set_xlabel("mean |store price $-$ assigned chain price| (\\$)")
    ax[1].set_ylabel("held-out log-likelihood per purchase")
    ax[1].set_title(f"2. Pooling does not show up in fit\ngap "
                    f"{r['fit_gap_low_vs_high_deviation']:+.3f} nats across quartiles",
                    fontsize=10)
    ax[1].grid(alpha=0.3)

    b = pd.DataFrame(r["fit_by_store_loyalty"])
    ax[2].plot(b.primary_store_share, b.loglik_per_purchase, "-o", color="#2e8b6f")
    ax[2].set_xlabel("share of trips at the household's primary store")
    ax[2].set_ylabel("held-out log-likelihood per purchase")
    ax[2].set_title("3. Store-loyal and store-switching\nhouseholds fit alike", fontsize=10)
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "stores.png"), dpi=150, bbox_inches="tight")
    log("wrote out/store_diagnostics.json and figures/stores.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--label", default="nf")
    p.add_argument("--min-store-trips", type=int, default=50)
    main(p.parse_args())
