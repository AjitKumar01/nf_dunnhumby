"""
Stage 8 -- Descriptive statistics for the report, all recomputed from the data.

Writes out/data_report.md and out/data_report.json.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
MI = os.path.join(HERE, "..", "model_input")
OUT = os.path.join(HERE, "..", "out")
# Raw dunnhumby CSVs.  Defaults to a sibling of the repository; override with
# NF_RAW_DIR if the download lives somewhere else.
RAW = os.path.join(os.environ.get(
    "NF_RAW_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                 "dunnhumby_The-Complete-Journey",
                 "dunnhumby_The-Complete-Journey CSV")), "")


def md_table(df):
    """Minimal markdown table writer (avoids a tabulate dependency)."""
    if isinstance(df, pd.Series):
        df = df.to_frame("value")
    df = df.reset_index() if df.index.name or not isinstance(df.index, pd.RangeIndex) else df
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        cells = []
        for v in row:
            if isinstance(v, (int, np.integer)) or (isinstance(v, float) and float(v).is_integer()
                                                    and abs(v) >= 1):
                cells.append(f"{int(v):,}")
            elif isinstance(v, (float, np.floating)):
                cells.append(f"{v:,.4g}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    os.makedirs(OUT, exist_ok=True)
    r = {}
    tx = pd.read_parquet(os.path.join(DATA, "tx.parquet"),
                         columns=["PRODUCT_ID", "household_key", "DAY", "WEEK_NO", "STORE_ID",
                                  "BASKET_ID", "unit_price", "weekday", "SALES_VALUE"])
    trips = pd.read_parquet(os.path.join(DATA, "trips.parquet"))

    r["raw"] = {"lines": len(tx), "households": int(tx.household_key.nunique()),
                "products": int(tx.PRODUCT_ID.nunique()), "stores": int(tx.STORE_ID.nunique()),
                "days": int(tx.DAY.nunique()), "weeks": int(tx.WEEK_NO.nunique()),
                "trips": len(trips)}

    # ---- price-change hazard by day pair (the identification check)
    pdm = tx.groupby(["PRODUCT_ID", "DAY"]).unit_price.agg(["median", "size"]).reset_index()
    dmap = tx[["DAY", "WEEK_NO", "weekday"]].drop_duplicates()
    g = pdm[pdm["size"] >= 5].merge(dmap, on="DAY").sort_values(["PRODUCT_ID", "DAY"])
    g["prev"] = g.groupby("PRODUCT_ID")["median"].shift()
    g["prevday"] = g.groupby("PRODUCT_ID").DAY.shift()
    g["prevwk"] = g.groupby("PRODUCT_ID").WEEK_NO.shift()
    g = g[(g.DAY - g.prevday) == 1]
    g["chg"] = (g["median"] - g["prev"]).abs() > 0.02
    haz = g.groupby(g.WEEK_NO != g.prevwk).chg.agg(["mean", "size"])
    r["price_change_hazard"] = {
        "within_week": float(haz.loc[False, "mean"]), "within_week_n": int(haz.loc[False, "size"]),
        "week_boundary": float(haz.loc[True, "mean"]), "week_boundary_n": int(haz.loc[True, "size"])}

    # ---- cross-store price dispersion (justifies chain-level session prices)
    top = tx.groupby("PRODUCT_ID").size().sort_values(ascending=False).index[:400]
    sub = tx[tx.PRODUCT_ID.isin(top)]
    pw = sub.groupby(["PRODUCT_ID", "WEEK_NO", "STORE_ID"]).unit_price.median().reset_index()
    agg = pw.groupby(["PRODUCT_ID", "WEEK_NO"]).unit_price.agg(["median", "std", "size"])
    agg = agg[agg["size"] >= 5]
    r["cross_store_price_cv_median"] = float((agg["std"] / agg["median"]).median())
    r["cross_store_cells"] = int(len(agg))

    # ---- retained sample
    items = pd.read_csv(os.path.join(MI, "id_maps", "items.csv"))
    sess = pd.read_csv(os.path.join(MI, "id_maps", "sessions.csv"))
    obs = pd.read_csv(os.path.join(MI, "id_maps", "observations.csv"))
    audit = pd.read_csv(os.path.join(DATA, "filter_audit.csv"))
    st = pd.read_parquet(os.path.join(DATA, "sample_trips.parquet"))
    r["sample"] = {
        "households": int(obs.user_id.nunique()), "items": len(items),
        "categories": int(items.group_id.nunique()), "sessions": len(sess),
        "pair_weeks": int(sess.pair_week.nunique()), "trips": len(st),
        "category_purchases": len(obs),
        "purchase_rate_per_category_trip": len(obs) / (len(st) * items.group_id.nunique()),
        "split": obs.split.value_counts().to_dict(),
    }

    # ---- filter audit
    audit["drop_reason"] = audit.drop_reason.fillna("KEPT").replace("", "KEPT")
    r["filters"] = audit.drop_reason.value_counts().to_dict()

    # ---- price variation inside the retained sample
    ev = pd.read_csv(os.path.join(MI, "events.csv"))
    r["events"] = {
        "item_pairweeks": len(ev),
        "own_price_change": int(ev.own_price_change.sum()),
        "cross_price_change": int(ev.cross_price_change.sum()),
        "mean_abs_dp_when_changed": float(ev.loc[ev.own_price_change == 1, "dp"].abs().mean()),
    }

    # ---- household coverage of demographics
    demo = pd.read_csv(RAW + "hh_demographic.csv")
    users = pd.read_csv(os.path.join(MI, "id_maps", "users.csv"))
    r["demographics_coverage"] = float(users.household_key.isin(demo.household_key).mean())

    with open(os.path.join(OUT, "data_report.json"), "w") as f:
        json.dump(r, f, indent=2)

    cat = pd.read_parquet(os.path.join(DATA, "categories.parquet"))
    lines = ["# dunnhumby sample for Nested Factorization\n",
             "## Raw panel\n",
             md_table(pd.Series(r["raw"])), "",
             "## Identification: where do prices change?\n",
             f"- consecutive days **within** a `WEEK_NO`: P(price move > 2c) = "
             f"{r['price_change_hazard']['within_week']:.3f} "
             f"(n={r['price_change_hazard']['within_week_n']:,})",
             f"- consecutive days **across** the Sunday→Monday week boundary: "
             f"{r['price_change_hazard']['week_boundary']:.3f} "
             f"(n={r['price_change_hazard']['week_boundary_n']:,})",
             f"- median cross-store CV of a product's price within a week: "
             f"{r['cross_store_price_cv_median']:.4f} "
             f"({r['cross_store_cells']:,} product-weeks with >=5 stores)", "",
             "## Retained estimation sample\n",
             md_table(pd.Series(r["sample"])), "",
             "## Category filters (paper app. 8.1)\n",
             md_table(pd.Series(r["filters"])), "",
             "## Quasi-experimental variation available\n",
             md_table(pd.Series(r["events"])), "",
             "## Retained categories\n",
             md_table(cat[["COMMODITY_DESC", "n_items_all", "trips_top", "multi_any",
                           "mean_abs_price_corr", "max_share_big"]].round(3))]
    with open(os.path.join(OUT, "data_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))
    print(f"\n[08] wrote out/data_report.md and out/data_report.json")


if __name__ == "__main__":
    main()
