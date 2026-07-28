"""
Stage 1 -- Build the cleaned transaction base and the price panel.

Mirrors the data construction in Donnelly, Ruiz, Blei & Athey (2023),
"Counterfactual Inference for Consumer Choice Across Many Product Categories",
adapted to dunnhumby "The Complete Journey".

Paper setting                          dunnhumby analogue
-------------------------------------  --------------------------------------
1 store, 23 months, loyalty card       582 stores, 102 weeks, loyalty panel
UPC                                    PRODUCT_ID
"category" (unit of substitution)      COMMODITY_DESC
"class"/"subclass" (held out)          SUB_COMMODITY_DESC / BRAND / MANUFACTURER
trip = household x calendar day        household x DAY (BASKET_ID is 1:1 with it)
session = day (prices constant)        session = day (chain) or (store, day)
price = daily median transacted price  weekly median transacted unit price

Outputs (parquet, in ../data):
    tx.parquet          cleaned transaction lines with unit prices
    trips.parquet       one row per (household, day) shopping trip
    price_week.parquet  product x week chain-level price panel
    price_store_week.parquet  product x store x week price panel
"""
import os
import numpy as np
import pandas as pd

# Raw dunnhumby CSVs.  Defaults to a sibling of the repository; override with
# NF_RAW_DIR if the download lives somewhere else.
RAW = os.path.join(os.environ.get(
    "NF_RAW_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                 "dunnhumby_The-Complete-Journey",
                 "dunnhumby_The-Complete-Journey CSV")), "")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(OUT, exist_ok=True)


def log(msg):
    print(f"[01] {msg}", flush=True)


def main():
    log("reading transaction_data.csv ...")
    tx = pd.read_csv(
        RAW + "transaction_data.csv",
        dtype={
            "household_key": np.int32, "BASKET_ID": np.int64, "DAY": np.int16,
            "PRODUCT_ID": np.int32, "QUANTITY": np.int32, "STORE_ID": np.int32,
            "TRANS_TIME": np.int16, "WEEK_NO": np.int16,
        },
    )
    n0 = len(tx)

    # ------------------------------------------------------------------ prices
    # sales_value is what the retailer receives; it is already net of the loyalty
    # discount (retail_disc) and of the retailer's coupon match, and it *includes*
    # the manufacturer's reimbursement (coupon_disc).  The posted loyalty-card
    # shelf price -- the price every card-holder faces, before that shopper's own
    # coupon activity -- is therefore (sales_value - coupon_match_disc)/quantity.
    # Discount columns are stored as negatives, so subtracting adds them back.
    tx = tx[(tx.QUANTITY > 0) & (tx.SALES_VALUE > 0)]
    tx["unit_price"] = (tx.SALES_VALUE - tx.COUPON_MATCH_DISC) / tx.QUANTITY
    # regular (non-loyalty) shelf price, kept for diagnostics / promo-depth measures
    tx["reg_price"] = (tx.SALES_VALUE - tx.RETAIL_DISC - tx.COUPON_MATCH_DISC) / tx.QUANTITY
    tx["coupon_used"] = (tx.COUPON_DISC < 0).astype(np.int8)

    # Drop implausible lines: random-weight / bulk lines and price outliers.
    bad = (tx.QUANTITY > 30) | (tx.unit_price > 100) | (tx.unit_price < 0.05)
    log(f"dropping {bad.sum():,} of {n0:,} lines (bulk quantity or extreme unit price)")
    tx = tx[~bad].copy()

    # ------------------------------------------------------- calendar handling
    # WEEK_NO is given.  Week 1 is short (days 1-5), week 102 is short (706-711).
    # Establish the weekday index from the basket-count profile: the two busiest
    # residues of DAY % 7 are the weekend.
    bask_by_res = tx.groupby(tx.DAY % 7).BASKET_ID.nunique()
    weekend = set(bask_by_res.sort_values(ascending=False).index[:2])
    log(f"busiest DAY%7 residues (assumed Sat/Sun): {sorted(weekend)}")
    # Anchor: call the later of the two weekend residues 'Sunday' (=0) so that
    # weekday ids run 0..6 = Sun..Sat.  Only the partition matters for the model.
    sunday_res = max(weekend)
    tx["weekday"] = ((tx.DAY % 7) - sunday_res) % 7
    tx["weekday"] = tx.weekday.astype(np.int8)
    tx["hour"] = (tx.TRANS_TIME // 100 + (tx.TRANS_TIME % 100) / 60.0).astype(np.float32)

    # ------------------------------------------------------------------- trips
    # A trip is a household-day (the paper's definition).  Verify BASKET_ID nests.
    trips = (
        tx.groupby(["household_key", "DAY"])
        .agg(
            n_lines=("PRODUCT_ID", "size"),
            n_items=("PRODUCT_ID", "nunique"),
            spend=("SALES_VALUE", "sum"),
            store_id=("STORE_ID", lambda s: s.mode().iat[0]),
            n_stores=("STORE_ID", "nunique"),
            week=("WEEK_NO", "first"),
            weekday=("weekday", "first"),
            hour=("hour", "median"),
        )
        .reset_index()
    )
    log(f"trips: {len(trips):,}   households: {trips.household_key.nunique():,}   "
        f"multi-store trips: {(trips.n_stores > 1).mean():.4f}")

    # ------------------------------------------------------------ price panels
    # Chain-level product x week price = median transacted unit price.
    pw = (
        tx.groupby(["PRODUCT_ID", "WEEK_NO"])
        .agg(price=("unit_price", "median"),
             reg_price=("reg_price", "median"),
             n_tx=("unit_price", "size"),
             n_store=("STORE_ID", "nunique"))
        .reset_index()
    )
    # Store-level product x store x week price.
    psw = (
        tx.groupby(["PRODUCT_ID", "STORE_ID", "WEEK_NO"])
        .agg(price=("unit_price", "median"), n_tx=("unit_price", "size"))
        .reset_index()
    )

    log(f"chain price cells: {len(pw):,}   store price cells: {len(psw):,}")

    tx.to_parquet(os.path.join(OUT, "tx.parquet"), index=False)
    trips.to_parquet(os.path.join(OUT, "trips.parquet"), index=False)
    pw.to_parquet(os.path.join(OUT, "price_week.parquet"), index=False)
    psw.to_parquet(os.path.join(OUT, "price_store_week.parquet"), index=False)
    log("wrote tx / trips / price_week / price_store_week parquet")


if __name__ == "__main__":
    main()
